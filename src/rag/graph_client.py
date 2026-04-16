# src/rag/graph_client.py
"""
Temporal knowledge graph client backed by SurrealDB.

Replaces Graphiti + Neo4j: episodes and extracted facts with vector search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from surrealdb import RecordID

from ..config.experiment_setups import ExperimentSetup
from ..config.settings import GeminiConfig, SurrealDBConfig, get_config
from ..embedders import BaseEmbedder, EmbedderType, create_embedder
from ..embedders.factory import EmbedderConfig
from .surreal.connection import apply_schema, connect_surreal

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _flatten_query(res: Any) -> List[Dict[str, Any]]:
    if res is None:
        return []
    if isinstance(res, list):
        out: List[Dict[str, Any]] = []
        for block in res:
            if isinstance(block, dict) and "result" in block:
                r = block["result"]
                if isinstance(r, list):
                    out.extend([x for x in r if isinstance(x, dict)])
                elif isinstance(r, dict):
                    out.append(r)
            elif isinstance(block, dict):
                out.append(block)
        return out
    return []


def _strip_json_fence(text: str) -> str:
    t = re.sub(r"^```json\s*", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r"^```\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s*```$", "", t, flags=re.MULTILINE)
    return t.strip()


class SearchResult:
    """Standardized search result"""

    def __init__(
        self,
        fact: str,
        score: float,
        entity_name: Optional[str] = None,
        created_at: Optional[datetime] = None,
        valid_at: Optional[datetime] = None,
        source_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.fact = fact
        self.score = score
        self.entity_name = entity_name
        self.created_at = created_at
        self.valid_at = valid_at
        self.source_description = source_description
        self.metadata = metadata or {}


class TemporalGraphClient:
    """
    SurrealDB-backed temporal graph for Agentic RAG.

    - add_episode: stores episode + LLM-extracted facts with embeddings
    - search: cosine similarity on extracted_fact.embedding
    """

    def __init__(
        self,
        surreal_config: Optional[SurrealDBConfig] = None,
        gemini_config: Optional[GeminiConfig] = None,
        group_id: Optional[str] = None,
        setup: Optional[ExperimentSetup] = None,
    ):
        cfg = get_config()
        self.surreal_config = surreal_config or cfg.surreal
        self.gemini_config = gemini_config or cfg.gemini
        self.setup = setup
        if group_id:
            self.group_id = group_id
        elif setup and setup.storage.group_id:
            self.group_id = setup.storage.group_id
        else:
            self.group_id = f"temporal_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # surrealdb.AsyncSurreal is a factory in stubs; use Any for the live connection object.
        self._db: Any = None
        self._embedder: Optional[BaseEmbedder] = None

    @property
    def embedder(self) -> Optional[BaseEmbedder]:
        """Embedder used for fact vectors (shared with dense session passages when unified ingest)."""
        return self._embedder

    @classmethod
    def from_setup(
        cls,
        setup: ExperimentSetup,
        surreal_config: Optional[SurrealDBConfig] = None,
    ) -> "TemporalGraphClient":
        return cls(surreal_config=surreal_config, setup=setup)

    async def initialize(self) -> None:
        self._db = await connect_surreal(self.surreal_config)
        await apply_schema(self._db)

        if self.setup is None:
            logger.info("TemporalGraphClient minimal init (group_id=%s)", self.group_id)
            return

        if self.setup.embedder.provider == "huggingface":
            self._embedder = create_embedder(
                embedder_type=EmbedderType.HUGGINGFACE,
                model_name=self.setup.embedder.name,
            )
        else:
            self._embedder = create_embedder(
                config=EmbedderConfig(
                    embedder_type=EmbedderType.GEMINI,
                    model_name=self.setup.embedder.name,
                    description="",
                ),
                gemini_api_key=self.gemini_config.api_key,
            )
        await self._embedder.initialize()
        logger.info("TemporalGraphClient ready group_id=%s", self.group_id)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
        logger.info("SurrealDB connection closed")

    @property
    def client(self) -> Any:
        """Legacy attribute used by some tests; returns self for search routing."""
        return self

    async def search(
        self,
        query: Optional[str] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Semantic search; accepts Graphiti-style keyword args (query=, group_ids=)."""
        query = query or kwargs.pop("query", None)
        num_results = int(kwargs.pop("num_results", num_results))
        group_ids = group_ids or kwargs.pop("group_ids", None)
        if kwargs:
            logger.debug("search: ignored extra kwargs %s", list(kwargs.keys()))
        if not query:
            raise ValueError("query is required")
        if self._db is None:
            raise RuntimeError("Client not initialized")
        gids = group_ids or [self.group_id]
        gid = gids[0]
        if self._embedder is None:
            raise RuntimeError("Embedder required for search")
        emb = await self._embedder.embed([query])
        if not emb.embeddings:
            raise RuntimeError("Embedding API returned no vectors")
        qv = list(float(x) for x in emb.embeddings[0])
        sql = (
            "SELECT fact_text, entity_names, valid_at, created_at, source_description, "
            "vector::similarity::cosine(embedding, $qv) AS score "
            "FROM extracted_fact WHERE group_id = $gid "
            "ORDER BY score DESC LIMIT $lim"
        )
        res = await self._db.query(sql, {"qv": qv, "gid": gid, "lim": num_results})
        rows = _flatten_query(res)
        out: List[SearchResult] = []
        for r in rows:
            names = r.get("entity_names") or []
            en = names[0] if isinstance(names, list) and names else None
            va = r.get("valid_at")
            ca = r.get("created_at")
            out.append(
                SearchResult(
                    fact=str(r.get("fact_text", "")),
                    score=float(r.get("score", 0.0)),
                    entity_name=str(en) if en else None,
                    created_at=ca if isinstance(ca, datetime) else None,
                    valid_at=va if isinstance(va, datetime) else None,
                    source_description=r.get("source_description"),
                )
            )
        return out

    async def add_episode(
        self,
        content: str,
        name: str,
        source_description: str,
        reference_time: Optional[datetime] = None,
        source_type: Any = None,
        group_id: Optional[str] = None,
        episode_body: Optional[str] = None,
    ) -> str:
        """Accepts both (content, ...) and Graphiti-style episode_body keyword."""
        if self._db is None:
            raise RuntimeError("Client not initialized")
        body = episode_body if episode_body is not None else content
        gid = group_id or self.group_id
        ref = _as_utc(reference_time) if reference_time else _utc_now()
        created_at = _utc_now()

        ep_id = str(uuid.uuid4())
        ep_rid = RecordID("episode", ep_id)
        await self._db.upsert(
            ep_rid,
            {
                "group_id": gid,
                "name": name,
                "body": body,
                "source_description": source_description,
                "reference_time": ref,
                "created_at": created_at,
            },
        )

        facts = await self._extract_facts(body, ref)
        if self._embedder is None:
            logger.warning("No embedder; skipping fact vectors")
            return ep_id

        for item in facts:
            ft = item.get("fact") or item.get("fakta") or ""
            if not str(ft).strip():
                continue
            ents = item.get("entities") or item.get("entitas") or []
            if not isinstance(ents, list):
                ents = []
            va_raw = item.get("valid_at")
            if va_raw:
                parsed_va = _parse_dt(va_raw)
                va = _as_utc(parsed_va) if parsed_va is not None else ref
            else:
                va = ref
            emb_res = await self._embedder.embed([str(ft)])
            if not emb_res.embeddings:
                continue
            fv = list(float(x) for x in emb_res.embeddings[0])
            frid = RecordID("extracted_fact", str(uuid.uuid4()))
            await self._db.upsert(
                frid,
                {
                    "group_id": gid,
                    "episode_name": name,
                    "fact_text": str(ft),
                    "embedding": fv,
                    "entity_names": [str(e) for e in ents],
                    "valid_at": va,
                    "source_description": source_description,
                    "created_at": created_at,
                },
            )
            try:
                await self._db.query(
                    "RELATE $ep->has_fact->$ft",
                    {"ep": ep_rid, "ft": frid},
                )
            except Exception as ex:
                logger.warning("RELATE has_fact (episode→fact): %s", ex)
            for ename in ents:
                safe_e = re.sub(r"[^a-zA-Z0-9_]", "_", f"{gid}_{ename}")[:180]
                er = RecordID("entity", safe_e)
                await self._db.upsert(
                    er,
                    {
                        "group_id": gid,
                        "name": str(ename),
                        "created_at": created_at,
                    },
                )
                try:
                    await self._db.query(
                        "RELATE $ft->fact_involves->$ent",
                        {"ft": frid, "ent": er},
                    )
                except Exception as ex:
                    logger.warning("RELATE fact_involves (fact→entity): %s", ex)
        return ep_id

    async def _extract_facts(self, body: str, reference_time: datetime) -> List[Dict[str, Any]]:
        """Single-pass JSON fact extraction (Indonesian)."""
        prompt = f"""Anda mengekstrak fakta atomik dari percakapan berikut (Bahasa Indonesia).
Waktu referensi (ISO): {reference_time.isoformat()}

Keluarkan HANYA JSON array valid tanpa markdown, bentuk:
[{{"fact": "...", "entities": ["..."], "valid_at": "ISO8601 atau null"}}]

Aturan:
- Setiap elemen array satu fakta mandiri.
- entities berisi nama entitas yang disebutkan untuk fakta tersebut.
- valid_at gunakan null jika tidak jelas.

Teks percakapan:
---
{body[:120_000]}
---
"""
        if self.setup and self.setup.llm_extraction and self.setup.llm_extraction.provider == "novita":
            from openai import AsyncOpenAI

            cfg = get_config().novita
            if not cfg.is_configured():
                raise ValueError("NOVITAAI_API_KEY required for Gemma extraction")
            client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            resp = await client.chat.completions.create(
                model=self.setup.llm_extraction.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=8192,
            )
            text = (resp.choices[0].message.content or "").strip()
        else:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=self.gemini_config.api_key)
            model_name = (
                self.setup.llm_extraction.name
                if self.setup and self.setup.llm_extraction
                else self.gemini_config.model_medium
            )
            loop = asyncio.get_running_loop()

            def _call():
                return client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.2, max_output_tokens=8192
                    ),
                )

            resp = await loop.run_in_executor(None, _call)
            text = (resp.text or "").strip()

        cleaned = _strip_json_fence(text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Fact JSON parse failed; storing raw episode only")
            return [{"fact": body[:2000], "entities": [], "valid_at": None}]
        if isinstance(data, dict):
            for k in ("facts", "items", "data"):
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            return [{"fact": body[:2000], "entities": [], "valid_at": None}]
        return [x for x in data if isinstance(x, dict)]

    async def search_with_temporal_filter(
        self,
        query: str,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        num_results: int = 10,
    ) -> List[SearchResult]:
        results = await self.search(query, num_results=num_results * 2)
        filtered: List[SearchResult] = []
        for r in results:
            if r.valid_at:
                if before and r.valid_at > before:
                    continue
                if after and r.valid_at < after:
                    continue
            filtered.append(r)
        return filtered[:num_results]

    async def get_entity_facts(self, entity_name: str, limit: int = 20) -> List[SearchResult]:
        if self._db is None:
            return []
        sql = (
            "SELECT fact_text, valid_at, created_at FROM extracted_fact "
            "WHERE group_id = $gid AND array::contains(entity_names, $name) LIMIT $lim"
        )
        try:
            res = await self._db.query(
                sql, {"gid": self.group_id, "name": entity_name, "lim": limit}
            )
        except Exception as e:
            logger.warning("get_entity_facts query failed: %s", e)
            return []
        rows = _flatten_query(res)
        return [
            SearchResult(
                fact=str(r.get("fact_text", "")),
                score=1.0,
                entity_name=entity_name,
                created_at=r.get("created_at") if isinstance(r.get("created_at"), datetime) else None,
                valid_at=r.get("valid_at") if isinstance(r.get("valid_at"), datetime) else None,
            )
            for r in rows
        ]

    async def get_stats(self) -> Dict[str, int]:
        if self._db is None:
            return {"entities": 0, "edges": 0, "episodes": 0}
        db = self._db
        gid = self.group_id

        async def _cnt(table: str) -> int:
            q = f"SELECT count() AS c FROM {table} WHERE group_id = $gid GROUP ALL"
            try:
                res = await db.query(q, {"gid": gid})
                rows = _flatten_query(res)
                if rows and rows[0].get("c") is not None:
                    return int(rows[0]["c"])
            except Exception as e:
                logger.debug("count %s: %s", table, e)
            return 0

        ent = await _cnt("entity")
        ep = await _cnt("episode")
        facts = await _cnt("extracted_fact")
        return {"entities": ent, "edges": facts, "episodes": ep, "facts": facts}

    async def clear_group(self) -> None:
        if self._db is None:
            await self.initialize()
        assert self._db is not None
        gid = self.group_id
        try:
            await self._db.query(
                "DELETE FROM has_fact WHERE out IN (SELECT id FROM extracted_fact WHERE group_id = $gid)",
                {"gid": gid},
            )
        except Exception as e:
            logger.debug("delete has_fact: %s", e)
        try:
            await self._db.query(
                "DELETE FROM fact_involves WHERE in IN (SELECT id FROM extracted_fact WHERE group_id = $gid)",
                {"gid": gid},
            )
        except Exception as e:
            logger.debug("delete fact_involves: %s", e)
        for table in ("extracted_fact", "episode", "entity"):
            await self._db.query(f"DELETE FROM {table} WHERE group_id = $gid", {"gid": gid})
        logger.warning("Cleared SurrealDB data for group_id=%s", gid)


def _parse_dt(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        s = str(val).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


async def test_connection() -> bool:
    client = TemporalGraphClient()
    try:
        await client.initialize()
        stats = await client.get_stats()
        print("Connection OK", client.group_id, stats)
        return True
    except Exception as e:
        print("Connection failed", e)
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
