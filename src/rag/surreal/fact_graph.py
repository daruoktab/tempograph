# src/rag/surreal/fact_graph.py
"""
SurrealDB temporal fact graph: episodes, ``extracted_fact`` vectors, ``entity`` nodes,
and graph edges (``has_fact``, ``fact_involves``) for agentic / hybrid retrieval.
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

from ...config.experiment_setups import ExperimentSetup
from ...config.settings import GeminiConfig, SurrealDBConfig, get_config
from ...embedders import BaseEmbedder, EmbedderType, create_embedder
from ...embedders.factory import EmbedderConfig
from .connection import apply_schema, connect_surreal

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

    - add_episode: stores episode + LLM-extracted facts with embeddings; RELATE
      ``has_fact`` (episode→fact) and ``fact_involves`` (fact→entity).
    - search: cosine similarity on ``extracted_fact.embedding``, merged with
      facts reachable from ``entity`` nodes whose names appear in the query
      (``fact_involves`` traversal + optional score boost when also vector-hit).
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

    async def _vector_search_rows(
        self, qv: List[float], gid: str, limit: int
    ) -> List[Dict[str, Any]]:
        sql = (
            "SELECT fact_text, entity_names, valid_at, created_at, source_description, "
            "vector::similarity::cosine(embedding, $qv) AS score "
            "FROM extracted_fact WHERE group_id = $gid "
            "ORDER BY score DESC LIMIT $lim"
        )
        res = await self._db.query(sql, {"qv": qv, "gid": gid, "lim": limit})
        return _flatten_query(res)

    @staticmethod
    def _rows_to_search_results(rows: List[Dict[str, Any]], metadata_extra: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        out: List[SearchResult] = []
        extra = metadata_extra or {}
        for r in rows:
            names = r.get("entity_names") or []
            en = names[0] if isinstance(names, list) and names else None
            va = r.get("valid_at")
            ca = r.get("created_at")
            meta = {**extra, **(r.get("_retrieval_meta") or {})}
            if isinstance(names, list) and names:
                meta = {**meta, "entity_names": [str(x) for x in names]}
            out.append(
                SearchResult(
                    fact=str(r.get("fact_text", "")),
                    score=float(r.get("score", 0.0)),
                    entity_name=str(en) if en else None,
                    created_at=ca if isinstance(ca, datetime) else None,
                    valid_at=va if isinstance(va, datetime) else None,
                    source_description=r.get("source_description"),
                    metadata=meta if meta else None,
                )
            )
        return out

    async def resolve_entities_in_query(self, query: str, group_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return ``entity`` rows whose ``name`` is a case-insensitive substring of ``query``.
        Used to anchor retrieval on ``fact_involves`` edges.
        """
        if self._db is None or not query or not str(query).strip():
            return []
        gid = group_id or self.group_id
        q_lower = str(query).strip().lower()
        sql = (
            "SELECT id, name FROM entity WHERE group_id = $gid "
            "AND string::contains($q, string::lowercase(name)) "
            "AND string::len(name) >= 2 LIMIT 25"
        )
        try:
            res = await self._db.query(sql, {"gid": gid, "q": q_lower})
        except Exception as e:
            logger.debug("resolve_entities_in_query failed: %s", e)
            return []
        return _flatten_query(res)

    async def search_facts_for_entity_ids(
        self,
        entity_ids: List[Any],
        qv: List[float],
        group_id: Optional[str] = None,
        limit: int = 30,
    ) -> List[SearchResult]:
        """
        Facts linked to the given entity record ids via ``fact_involves`` (fact → entity),
        ordered by cosine similarity to ``qv`` on the restricted set.
        """
        if self._db is None or not entity_ids:
            return []
        gid = group_id or self.group_id
        sql = (
            "SELECT fact_text, entity_names, valid_at, created_at, source_description, "
            "vector::similarity::cosine(embedding, $qv) AS score "
            "FROM extracted_fact WHERE group_id = $gid AND id IN ("
            "SELECT in FROM fact_involves WHERE out IN $eids) "
            "ORDER BY score DESC LIMIT $lim"
        )
        try:
            res = await self._db.query(sql, {"qv": qv, "gid": gid, "eids": entity_ids, "lim": limit})
        except Exception as e:
            logger.warning("search_facts_for_entity_ids failed: %s", e)
            return []
        rows = _flatten_query(res)
        return self._rows_to_search_results(rows, {"source": "entity_graph"})

    @staticmethod
    def _merge_vector_and_graph_results(
        vector_hits: List[SearchResult],
        graph_hits: List[SearchResult],
        resolved_entity_names: List[str],
        num_results: int,
    ) -> List[SearchResult]:
        """Dedupe by fact text, boost overlap between vector and graph + name overlap."""
        names_lower = {n.lower() for n in resolved_entity_names if isinstance(n, str) and len(n) >= 2}

        def _name_overlap(sr: SearchResult) -> bool:
            if not names_lower:
                return False
            raw = sr.entity_name or ""
            if isinstance(raw, str) and raw.lower() in names_lower:
                return True
            ens = (sr.metadata or {}).get("entity_names") or []
            for n in ens:
                if isinstance(n, str) and n.lower() in names_lower:
                    return True
            return False

        merged: Dict[str, SearchResult] = {}
        for r in vector_hits:
            if not r.fact:
                continue
            meta = {**(r.metadata or {}), "source": "vector"}
            merged[r.fact] = SearchResult(
                fact=r.fact,
                score=r.score,
                entity_name=r.entity_name,
                created_at=r.created_at,
                valid_at=r.valid_at,
                source_description=r.source_description,
                metadata=meta,
            )

        for r in graph_hits:
            if not r.fact:
                continue
            boost = 0.04 if r.fact in merged else 0.0
            if r.fact in merged:
                old = merged[r.fact]
                new_score = min(1.0, max(old.score, r.score) + 0.03 + boost)
                meta = {**(old.metadata or {}), "source": "vector+graph", "graph_linked": True}
                merged[r.fact] = SearchResult(
                    fact=old.fact,
                    score=new_score,
                    entity_name=old.entity_name or r.entity_name,
                    created_at=old.created_at or r.created_at,
                    valid_at=old.valid_at or r.valid_at,
                    source_description=old.source_description or r.source_description,
                    metadata=meta,
                )
            else:
                meta = {**(r.metadata or {}), "source": "entity_graph"}
                merged[r.fact] = SearchResult(
                    fact=r.fact,
                    score=min(1.0, r.score + boost),
                    entity_name=r.entity_name,
                    created_at=r.created_at,
                    valid_at=r.valid_at,
                    source_description=r.source_description,
                    metadata=meta,
                )

        for fact, sr in list(merged.items()):
            if _name_overlap(sr) and sr.metadata and sr.metadata.get("source") == "vector":
                merged[fact] = SearchResult(
                    fact=sr.fact,
                    score=min(1.0, sr.score + 0.02),
                    entity_name=sr.entity_name,
                    created_at=sr.created_at,
                    valid_at=sr.valid_at,
                    source_description=sr.source_description,
                    metadata={**(sr.metadata or {}), "entity_query_overlap": True},
                )

        ranked = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return ranked[:num_results]

    async def search(
        self,
        query: Optional[str] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Semantic search plus optional entity-graph recall (``fact_involves``)."""
        query = query or kwargs.pop("query", None)
        num_results = int(kwargs.pop("num_results", num_results))
        group_ids = group_ids or kwargs.pop("group_ids", None)
        use_entity_graph = bool(kwargs.pop("use_entity_graph", True))
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

        fetch_lim = max(num_results * 3, num_results + 10)
        v_rows = await self._vector_search_rows(qv, gid, fetch_lim)
        vector_results = self._rows_to_search_results(v_rows, {"source": "vector"})

        if not use_entity_graph:
            return vector_results[:num_results]

        ent_rows = await self.resolve_entities_in_query(query, gid)
        if not ent_rows:
            return vector_results[:num_results]

        eids = [r["id"] for r in ent_rows if r.get("id") is not None]
        resolved_names = [str(r["name"]) for r in ent_rows if r.get("name") is not None]
        graph_results = await self.search_facts_for_entity_ids(
            eids, qv, gid, limit=max(num_results * 2, 20)
        )

        return self._merge_vector_and_graph_results(
            vector_results, graph_results, resolved_names, num_results
        )

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
        """Accepts ``content`` or legacy keyword ``episode_body`` (same text)."""
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
        gid = self.group_id
        rows: List[Dict[str, Any]] = []
        try:
            er = await self._db.query(
                "SELECT id FROM entity WHERE group_id = $gid "
                "AND string::lowercase(name) = string::lowercase($name) LIMIT 1",
                {"gid": gid, "name": entity_name},
            )
            erows = _flatten_query(er)
        except Exception as e:
            logger.debug("get_entity_facts entity lookup failed: %s", e)
            erows = []

        if erows and erows[0].get("id") is not None:
            eid = erows[0]["id"]
            try:
                res = await self._db.query(
                    "SELECT fact_text, entity_names, valid_at, created_at, source_description "
                    "FROM extracted_fact WHERE group_id = $gid AND id IN ("
                    "SELECT in FROM fact_involves WHERE out = $eid) "
                    "ORDER BY created_at DESC LIMIT $lim",
                    {"gid": gid, "eid": eid, "lim": limit},
                )
                rows = _flatten_query(res)
            except Exception as e:
                logger.debug("get_entity_facts graph traverse failed: %s", e)
                rows = []

        if not rows:
            fb = (
                "SELECT fact_text, entity_names, valid_at, created_at, source_description "
                "FROM extracted_fact WHERE group_id = $gid AND array::contains(entity_names, $name) "
                "ORDER BY created_at DESC LIMIT $lim"
            )
            try:
                res = await self._db.query(fb, {"gid": gid, "name": entity_name, "lim": limit})
                rows = _flatten_query(res)
            except Exception as e:
                logger.warning("get_entity_facts fallback failed: %s", e)
                return []
        out: List[SearchResult] = []
        for r in rows:
            names = r.get("entity_names") or []
            en = names[0] if isinstance(names, list) and names else entity_name
            out.append(
                SearchResult(
                    fact=str(r.get("fact_text", "")),
                    score=1.0,
                    entity_name=str(en) if en else entity_name,
                    created_at=r.get("created_at") if isinstance(r.get("created_at"), datetime) else None,
                    valid_at=r.get("valid_at") if isinstance(r.get("valid_at"), datetime) else None,
                    source_description=r.get("source_description"),
                    metadata={"source": "entity_expand"},
                )
            )
        return out

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
