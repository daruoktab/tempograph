# src/rag/surreal/fact_graph.py
"""
SurrealDB temporal fact graph: episodes, ``extracted_fact`` vectors, ``entity`` nodes,
and graph edges (``has_fact``, ``fact_involves``) for agentic / hybrid retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
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
from .connection import apply_schema, connect_surreal
from .ranking import (
    estimate_chars_per_token,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
    should_chunk_by_density,
)
from ..retrieval.trace import append_retrieval_trace, retrieval_span

logger = logging.getLogger(__name__)


def _entity_char_entropy(s: str) -> float:
    """Shannon entropy over characters (lowercase, no spaces) — filters trivial entity strings."""
    import math

    t = re.sub(r"\s+", "", s.lower())
    if len(t) < 2:
        return 0.0
    counts: Dict[str, int] = {}
    for c in t:
        counts[c] = counts.get(c, 0) + 1
    total = len(t)
    h = 0.0
    for cnt in counts.values():
        p = cnt / total
        h -= p * math.log2(p)
    return h


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
        self,
        qv: List[float],
        gid: str,
        limit: int,
        valid_before: Optional[datetime] = None,
        valid_after: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        conds = ["group_id = $gid"]
        params: Dict[str, Any] = {"qv": qv, "gid": gid, "lim": limit}
        if valid_before is not None:
            conds.append("(valid_at IS NONE OR valid_at <= $vb)")
            conds.append("(invalid_at IS NONE OR invalid_at > $vb)")
            params["vb"] = valid_before
        if valid_after is not None:
            conds.append("(valid_at IS NONE OR valid_at >= $va)")
            params["va"] = valid_after
        where_sql = " AND ".join(conds)
        sql = (
            f"SELECT id, fact_text, entity_names, valid_at, created_at, source_description, "
            f"vector::similarity::cosine(embedding, $qv) AS score "
            f"FROM extracted_fact WHERE {where_sql} "
            f"ORDER BY score DESC LIMIT $lim"
        )
        res = await self._db.query(sql, params)
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
            rid = r.get("id")
            if rid is not None:
                meta = {**meta, "fact_record_id": str(rid)}
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
        rows = _flatten_query(res)
        return self._filter_entity_rows(rows, query)

    async def search_facts_for_entity_ids(
        self,
        entity_ids: List[Any],
        qv: List[float],
        group_id: Optional[str] = None,
        limit: int = 30,
        valid_before: Optional[datetime] = None,
        valid_after: Optional[datetime] = None,
    ) -> List[SearchResult]:
        """
        Facts linked to the given entity record ids via ``fact_involves`` (fact → entity),
        ordered by cosine similarity to ``qv`` on the restricted set.
        """
        if self._db is None or not entity_ids:
            return []
        gid = group_id or self.group_id
        conds = [
            "group_id = $gid",
            "id IN (SELECT VALUE in FROM fact_involves WHERE out IN $eids)",
        ]
        params: Dict[str, Any] = {"qv": qv, "gid": gid, "eids": entity_ids, "lim": limit}
        if valid_before is not None:
            conds.append("(valid_at IS NONE OR valid_at <= $vb)")
            conds.append("(invalid_at IS NONE OR invalid_at > $vb)")
            params["vb"] = valid_before
        if valid_after is not None:
            conds.append("(valid_at IS NONE OR valid_at >= $va)")
            params["va"] = valid_after
        where_sql = " AND ".join(conds)
        sql = (
            f"SELECT id, fact_text, entity_names, valid_at, created_at, source_description, "
            f"vector::similarity::cosine(embedding, $qv) AS score "
            f"FROM extracted_fact WHERE {where_sql} "
            f"ORDER BY score DESC LIMIT $lim"
        )
        try:
            res = await self._db.query(sql, params)
        except Exception as e:
            logger.warning("search_facts_for_entity_ids failed: %s", e)
            return []
        rows = _flatten_query(res)
        return self._rows_to_search_results(rows, {"source": "entity_graph"})

    @staticmethod
    def _merge_vector_and_graph_results_heuristic(
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

    @staticmethod
    def _result_fact_id(sr: SearchResult) -> str:
        meta = sr.metadata or {}
        fid = meta.get("fact_record_id")
        if fid:
            return str(fid)
        return hashlib.sha256(sr.fact.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _filter_entity_rows(rows: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        if not rows:
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            name = str(r.get("name", "")).strip()
            if len(name) < 2:
                continue
            ent = _entity_char_entropy(name)
            if len(name) < 6 and ent < 1.0:
                continue
            out.append(r)
        return out if out else rows[:25]

    async def _keyword_ranked_fact_ids(
        self,
        query: str,
        gid: str,
        limit: int,
        valid_before: Optional[datetime] = None,
        valid_after: Optional[datetime] = None,
    ) -> List[str]:
        if self._db is None:
            return []
        toks = list({t.lower() for t in re.split(r"\W+", query) if len(t) > 3})[:5]
        if not toks:
            return []
        scored: Dict[str, int] = {}
        for tok in toks:
            conds = [
                "group_id = $gid",
                "string::contains(string::lowercase(fact_text), $tok)",
            ]
            params: Dict[str, Any] = {"gid": gid, "tok": tok, "lim": max(limit, 20)}
            if valid_before is not None:
                conds.append("(valid_at IS NONE OR valid_at <= $vb)")
                params["vb"] = valid_before
            if valid_after is not None:
                conds.append("(valid_at IS NONE OR valid_at >= $va)")
                params["va"] = valid_after
            where_sql = " AND ".join(conds)
            sql = f"SELECT id FROM extracted_fact WHERE {where_sql} LIMIT $lim"
            try:
                res = await self._db.query(sql, params)
            except Exception as e:
                logger.debug("_keyword_ranked_fact_ids: %s", e)
                continue
            for row in _flatten_query(res):
                rid = row.get("id")
                if rid is None:
                    continue
                k = str(rid)
                scored[k] = scored.get(k, 0) + 1
        return [fid for fid, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:limit]]

    async def _apply_mmr_to_ranked(
        self,
        ranked: List[SearchResult],
        qv: List[float],
        gid: str,
        k: int,
        pool: int,
        lambda_mult: float,
    ) -> List[SearchResult]:
        if not ranked or k <= 0 or self._db is None:
            return ranked[:k]
        pool = min(pool, len(ranked))
        ids: List[str] = []
        id_to_sr: Dict[str, SearchResult] = {}
        for r in ranked[:pool]:
            fid = self._result_fact_id(r)
            if fid not in id_to_sr or r.score > id_to_sr[fid].score:
                id_to_sr[fid] = r
            if fid not in ids:
                ids.append(fid)
        if len(ids) < 2:
            return ranked[:k]
        id_list = list(id_to_sr.keys())
        try:
            res = await self._db.query(
                "SELECT id, embedding FROM extracted_fact WHERE group_id = $gid AND id IN $ids",
                {"gid": gid, "ids": id_list},
            )
        except Exception as e:
            logger.debug("_apply_mmr_to_ranked: %s", e)
            return ranked[:k]
        emb_map: Dict[str, List[float]] = {}
        for row in _flatten_query(res):
            rid = row.get("id")
            emb = row.get("embedding")
            if rid is not None and isinstance(emb, list) and emb:
                emb_map[str(rid)] = [float(x) for x in emb]
        if len(emb_map) < 2:
            return ranked[:k]
        mmr_ids = maximal_marginal_relevance(
            qv, emb_map, k=min(k, len(emb_map)), lambda_mult=lambda_mult
        )
        out: List[SearchResult] = []
        for mid in mmr_ids:
            if mid in id_to_sr:
                out.append(id_to_sr[mid])
        return out if out else ranked[:k]

    async def _fuse_search_results(
        self,
        *,
        vector_results: List[SearchResult],
        graph_results: List[SearchResult],
        resolved_entity_names: List[str],
        num_results: int,
        use_rrf: bool,
        use_keyword_rrf: bool,
        rrf_rank_const: int,
        use_mmr: bool,
        mmr_lambda: float,
        mmr_pool_size: int,
        qv: List[float],
        gid: str,
        query: str,
        valid_before: Optional[datetime],
        valid_after: Optional[datetime],
    ) -> List[SearchResult]:
        if not use_rrf:
            return TemporalGraphClient._merge_vector_and_graph_results_heuristic(
                vector_results, graph_results, resolved_entity_names, num_results
            )
        rv = [self._result_fact_id(r) for r in vector_results if r.fact]
        rg = [self._result_fact_id(r) for r in graph_results if r.fact]
        rankings: List[List[str]] = [rv, rg]
        if use_keyword_rrf:
            kw = await self._keyword_ranked_fact_ids(
                query, gid, max(30, num_results * 4), valid_before, valid_after
            )
            if kw:
                rankings.append(kw)
        fused = reciprocal_rank_fusion(rankings, rank_const=rrf_rank_const)
        by_id: Dict[str, SearchResult] = {}
        for r in vector_results + graph_results:
            if not r.fact:
                continue
            fid = self._result_fact_id(r)
            prev = by_id.get(fid)
            if prev is None or r.score > prev.score:
                by_id[fid] = r
        ordered: List[SearchResult] = []
        for fid, rrf_s in fused:
            sr = by_id.get(fid)
            if sr is None:
                continue
            meta = {**(sr.metadata or {}), "rrf_score": float(rrf_s), "cosine_score": float(sr.score)}
            ordered.append(
                SearchResult(
                    fact=sr.fact,
                    score=float(rrf_s),
                    entity_name=sr.entity_name,
                    created_at=sr.created_at,
                    valid_at=sr.valid_at,
                    source_description=sr.source_description,
                    metadata=meta,
                )
            )
            if len(ordered) >= max(mmr_pool_size, num_results * 4, 30):
                break
        if use_mmr and self._db is not None:
            ordered = await self._apply_mmr_to_ranked(
                ordered, qv, gid, num_results, mmr_pool_size, mmr_lambda
            )
            return ordered[:num_results]
        return ordered[:num_results]

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
        rc = get_config().retrieval
        trace_path = kwargs.pop("trace_path", None) or rc.retrieval_trace_jsonl
        valid_before = kwargs.pop("valid_before", kwargs.pop("before", None))
        valid_after = kwargs.pop("valid_after", kwargs.pop("after", None))
        use_rrf = bool(kwargs.pop("use_rrf", rc.use_rrf))
        use_keyword_rrf = bool(kwargs.pop("use_keyword_rrf", rc.use_keyword_rrf))
        rrf_rank_const = int(kwargs.pop("rrf_rank_const", rc.rrf_rank_const))
        use_mmr = bool(kwargs.pop("use_mmr", rc.use_mmr))
        mmr_lambda = float(kwargs.pop("mmr_lambda", rc.mmr_lambda))
        mmr_pool_size = int(kwargs.pop("mmr_pool_size", rc.mmr_pool_size))
        fetch_mult = float(kwargs.pop("fact_fetch_multiplier", rc.fact_fetch_multiplier))
        fetch_extra = int(kwargs.pop("fact_fetch_min_extra", rc.fact_fetch_min_extra))
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
        with retrieval_span("fact_graph.embed_query", path=trace_path, group_id=gid):
            emb = await self._embedder.embed([query])
        if not emb.embeddings:
            raise RuntimeError("Embedding API returned no vectors")
        qv = list(float(x) for x in emb.embeddings[0])

        fetch_lim = max(int(num_results * fetch_mult), num_results + fetch_extra)
        with retrieval_span("fact_graph.vector_sql", path=trace_path, fetch_lim=fetch_lim):
            v_rows = await self._vector_search_rows(
                qv, gid, fetch_lim, valid_before=valid_before, valid_after=valid_after
            )
        vector_results = self._rows_to_search_results(v_rows, {"source": "vector"})

        if not use_entity_graph:
            out = vector_results[:num_results]
            append_retrieval_trace(
                {
                    "phase": "fact_graph.search_done",
                    "path": trace_path,
                    "vector_hits": len(vector_results),
                    "graph_hits": 0,
                    "returned": len(out),
                },
                path=trace_path,
            )
            return out

        with retrieval_span("fact_graph.resolve_entities", path=trace_path):
            ent_rows = await self.resolve_entities_in_query(query, gid)
        if not ent_rows:
            append_retrieval_trace(
                {
                    "phase": "fact_graph.search_done",
                    "path": trace_path,
                    "vector_hits": len(vector_results),
                    "graph_hits": 0,
                    "returned": min(len(vector_results), num_results),
                },
                path=trace_path,
            )
            return vector_results[:num_results]

        eids = [r["id"] for r in ent_rows if r.get("id") is not None]
        resolved_names = [str(r["name"]) for r in ent_rows if r.get("name") is not None]
        with retrieval_span("fact_graph.entity_graph_sql", path=trace_path):
            graph_results = await self.search_facts_for_entity_ids(
                eids,
                qv,
                gid,
                limit=max(num_results * 2, 20),
                valid_before=valid_before,
                valid_after=valid_after,
            )
        fused = await self._fuse_search_results(
            vector_results=vector_results,
            graph_results=graph_results,
            resolved_entity_names=resolved_names,
            num_results=num_results,
            use_rrf=use_rrf,
            use_keyword_rrf=use_keyword_rrf,
            rrf_rank_const=rrf_rank_const,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            mmr_pool_size=mmr_pool_size,
            qv=qv,
            gid=gid,
            query=query,
            valid_before=valid_before,
            valid_after=valid_after,
        )
        append_retrieval_trace(
            {
                "phase": "fact_graph.search_done",
                "path": trace_path,
                "vector_hits": len(vector_results),
                "graph_hits": len(graph_results),
                "entity_anchors": len(eids),
                "returned": len(fused),
            },
            path=trace_path,
        )
        return fused

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

            # Resolve all entity IDs (including deduplication)
            resolved_entity_ids = []
            for ename in ents:
                er = await self._resolve_entity(ename, gid, created_at)
                resolved_entity_ids.append(er)

            # Resolve contradictions and duplicates
            should_save = await self._resolve_contradictions(
                new_fact_text=str(ft),
                new_embedding=fv,
                entity_ids=resolved_entity_ids,
                entity_names=[str(e) for e in ents],
                gid=gid,
                reference_time=va,
            )

            if not should_save:
                continue

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

            for er in resolved_entity_ids:
                try:
                    await self._db.query(
                        "RELATE $ft->fact_involves->$ent",
                        {"ft": frid, "ent": er},
                    )
                except Exception as ex:
                    logger.warning("RELATE fact_involves (fact→entity): %s", ex)
        return ep_id

    async def _extract_facts(self, body: str, reference_time: datetime) -> List[Dict[str, Any]]:
        """
        LLM JSON fact extraction. Dense sessions are split into overlapping chunks only when
        ``should_chunk_by_density`` indicates high entity-like density (Graphiti-style heuristic).
        """
        work = body[:120_000]
        if should_chunk_by_density(work):
            chunk_chars = 3000 * estimate_chars_per_token()
            overlap = 200 * estimate_chars_per_token()
            merged: List[Dict[str, Any]] = []
            seen: set[str] = set()
            start = 0
            while start < len(work):
                piece = work[start : start + chunk_chars]
                if not piece.strip():
                    break
                batch = await self._extract_facts_once(piece, reference_time)
                for item in batch:
                    ft = str(item.get("fact") or item.get("fakta") or "").strip()
                    if ft and ft not in seen:
                        seen.add(ft)
                        merged.append(item)
                if start + chunk_chars >= len(work):
                    break
                start += chunk_chars - overlap
            return merged if merged else await self._extract_facts_once(work, reference_time)
        return await self._extract_facts_once(work, reference_time)

    async def _call_llm(self, prompt: str, max_output_tokens: int = 2048) -> str:
        """Call configured LLM (Gemini or Novita) with prompt and return raw text."""
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
                max_tokens=max_output_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
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
                        temperature=0.2, max_output_tokens=max_output_tokens
                    ),
                )

            resp = await loop.run_in_executor(None, _call)
            return (resp.text or "").strip()

    async def _extract_facts_once(self, body: str, reference_time: datetime) -> List[Dict[str, Any]]:
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
        text = await self._call_llm(prompt, max_output_tokens=8192)
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

    async def _resolve_entity(
        self,
        ename: str,
        gid: str,
        created_at: datetime,
    ) -> RecordID:
        """Resolve entity name to existing RecordID if it is a duplicate/alias, or create a new one."""
        try:
            er = await self._db.query(
                "SELECT id, name FROM entity WHERE group_id = $gid",
                {"gid": gid},
            )
            existing = _flatten_query(er)
        except Exception as e:
            logger.warning("Failed to query existing entities: %s", e)
            existing = []

        # Check exact match (case-insensitive)
        for r in existing:
            if r["name"].lower() == ename.lower():
                return r["id"]

        # Fuzzy match using rapidfuzz (including partial ratio and substring checks)
        from rapidfuzz import fuzz
        best_match = None
        best_score = 0.0
        for r in existing:
            score_sort = fuzz.token_sort_ratio(ename.lower(), r["name"].lower())
            score_partial = fuzz.partial_ratio(ename.lower(), r["name"].lower())
            score = max(score_sort, score_partial)
            
            # Boost if one is substring of another (e.g. "Aisha" in "Aisha Santoso")
            is_substring = (ename.lower() in r["name"].lower()) or (r["name"].lower() in ename.lower())
            if is_substring:
                score = max(score, 75.0)
                
            if score > best_score:
                best_score = score
                best_match = r

        if best_score >= 85 and best_match:
            matched_id = best_match["id"]
            if len(ename) > len(best_match["name"]):
                try:
                    await self._db.query(
                        "UPDATE entity SET name = $new_name WHERE id = $eid",
                        {"new_name": ename, "eid": matched_id}
                    )
                    logger.info("Updated entity name from %s to %s due to deduplication", best_match["name"], ename)
                except Exception:
                    pass
            return matched_id

        if 60 <= best_score < 85 and best_match:
            prompt = f"""Bandingkan dua entitas berikut dalam konteks profil pengguna chatbot:
Entitas A: "{best_match["name"]}"
Entitas B: "{ename}"

Apakah Entitas A dan Entitas B merujuk pada individu, tempat, organisasi, atau benda yang SAMA?
Keluarkan HANYA objek JSON valid berikut tanpa penjelasan apa pun:
{{"same": true}} atau {{"same": false}}
"""
            try:
                raw_resp = await self._call_llm(prompt)
                cleaned = _strip_json_fence(raw_resp)
                data = json.loads(cleaned)
                if data.get("same") is True:
                    matched_id = best_match["id"]
                    if len(ename) > len(best_match["name"]):
                        await self._db.query(
                            "UPDATE entity SET name = $new_name WHERE id = $eid",
                            {"new_name": ename, "eid": matched_id}
                        )
                    return matched_id
            except Exception as e:
                logger.debug("LLM entity deduplication check failed: %s", e)

        # Create new entity node
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
        return er

    async def _resolve_contradictions(
        self,
        new_fact_text: str,
        new_embedding: List[float],
        entity_ids: List[RecordID],
        entity_names: List[str],
        gid: str,
        reference_time: datetime,
    ) -> bool:
        """
        Compare a new fact with existing facts involving the same entities.
        If a duplicate is found, returns False (indicating the new fact should NOT be saved).
        If a contradiction is found, invalidates the old fact in DB by setting invalid_at = reference_time.
        Returns True if the new fact is compatible (or contradictory, but should be saved).
        """
        import numpy as np

        def _cosine_sim(v1, v2):
            v1_arr = np.array(v1, dtype=np.float64)
            v2_arr = np.array(v2, dtype=np.float64)
            n1 = np.linalg.norm(v1_arr)
            n2 = np.linalg.norm(v2_arr)
            if n1 == 0 or n2 == 0:
                return 0.0
            return float(np.dot(v1_arr, v2_arr) / (n1 * n2))

        should_save = True

        for eid, ename in zip(entity_ids, entity_names):
            try:
                res = await self._db.query(
                    "SELECT id, fact_text, embedding, valid_at FROM extracted_fact "
                    "WHERE group_id = $gid AND invalid_at IS NONE AND id IN ("
                    "SELECT VALUE in FROM fact_involves WHERE out = $eid)",
                    {"gid": gid, "eid": eid},
                )
                old_facts = _flatten_query(res)
            except Exception as e:
                logger.warning("Failed to query facts for entity %s: %s", ename, e)
                continue

            for old in old_facts:
                old_text = old.get("fact_text", "")
                old_emb = old.get("embedding", [])
                old_id = old.get("id")
                old_valid = old.get("valid_at") or reference_time

                # Check semantic similarity
                if not old_emb or not new_embedding:
                    sim = 0.0
                else:
                    sim = _cosine_sim(new_embedding, old_emb)

                logger.info("Comparing new fact: '%s' with old fact: '%s' (similarity: %.4f)", new_fact_text, old_text, sim)

                # If similar, compare using LLM
                if sim >= 0.35:
                    prompt = f"""Bandingkan dua fakta berikut mengenai entitas "{ename}":
Fakta Lama: "{old_text}" (Valid sejak: {old_valid.isoformat() if isinstance(old_valid, datetime) else str(old_valid)})
Fakta Baru: "{new_fact_text}" (Valid sejak: {reference_time.isoformat()})

Tentukan hubungan antara keduanya dari 3 pilihan berikut:
1. "compatible": Kedua fakta saling melengkapi atau tidak bertentangan.
2. "contradictory": Fakta baru bertentangan dengan fakta lama, atau fakta baru memperbarui/menggantikan informasi di fakta lama.
3. "duplicate": Fakta baru mengandung informasi yang sama persis dengan fakta lama.

Keluarkan HANYA objek JSON valid berikut tanpa penjelasan apa pun:
{{"relation": "compatible" | "contradictory" | "duplicate"}}
"""
                    try:
                        raw_resp = await self._call_llm(prompt)
                        cleaned = _strip_json_fence(raw_resp)
                        data = json.loads(cleaned)
                        relation = data.get("relation", "compatible")
                        if relation == "duplicate":
                            logger.info("Skipping duplicate fact: '%s'", new_fact_text)
                            should_save = False
                            break
                        elif relation == "contradictory":
                            await self._db.query(
                                "UPDATE extracted_fact SET invalid_at = $inv WHERE id = $oid",
                                {"inv": reference_time, "oid": old_id},
                            )
                            logger.info("Invalidated contradictory fact: '%s' by new fact: '%s'", old_text, new_fact_text)
                    except Exception as e:
                        logger.debug("LLM contradiction resolution failed: %s", e)

            if not should_save:
                break

        return should_save

    async def build_communities(self, group_id: Optional[str] = None) -> int:
        """
        Run graph community detection on entity relations and generate community summaries.
        Returns the number of communities built.
        """
        if self._db is None:
            raise RuntimeError("Client not initialized")
        gid = group_id or self.group_id
        
        # 1. Fetch all entities
        try:
            er = await self._db.query(
                "SELECT id, name FROM entity WHERE group_id = $gid",
                {"gid": gid},
            )
            entities = _flatten_query(er)
        except Exception as e:
            logger.warning("build_communities failed to fetch entities: %s", e)
            return 0
            
        if not entities:
            return 0
            
        eid_to_name = {str(item["id"]): item["name"] for item in entities}
        eid_to_rid = {str(item["id"]): item["id"] for item in entities}

        # 2. Fetch all fact_involves relations
        try:
            rel = await self._db.query(
                "SELECT in, out FROM fact_involves WHERE in.group_id = $gid",
                {"gid": gid},
            )
            relations = _flatten_query(rel)
        except Exception as e:
            logger.warning("build_communities failed to fetch relations: %s", e)
            return 0

        # Build mapping of fact_id -> list of entity_ids
        from collections import defaultdict
        fact_to_entities = defaultdict(list)
        for r in relations:
            fid = str(r["in"])
            eid = str(r["out"])
            if eid in eid_to_name:
                fact_to_entities[fid].append(eid)

        # Build the graph in NetworkX
        import networkx as nx
        G = nx.Graph()
        
        for eid in eid_to_name:
            G.add_node(eid)
            
        for fid, eids in fact_to_entities.items():
            if len(eids) >= 2:
                for i in range(len(eids)):
                    for j in range(i + 1, len(eids)):
                        G.add_edge(eids[i], eids[j])

        # Get communities using Louvain algorithm
        try:
            communities = list(nx.community.louvain_communities(G))
        except Exception:
            communities = list(nx.connected_components(G))

        # Clean up old communities for this group
        try:
            await self._db.query(
                "DELETE has_member WHERE in.group_id = $gid; "
                "DELETE community WHERE group_id = $gid;",
                {"gid": gid}
            )
        except Exception as e:
            logger.warning("Failed to clean up old communities: %s", e)

        community_count = 0
        created_at = _utc_now()

        for c_idx, node_set in enumerate(communities):
            if len(node_set) < 2:
                continue # Skip single-node communities
                
            c_eids = list(node_set)
            c_names = [eid_to_name[eid] for eid in c_eids]
            
            # Fetch facts related to these entities
            try:
                fact_res = await self._db.query(
                    "SELECT fact_text FROM extracted_fact WHERE group_id = $gid AND invalid_at IS NONE AND id IN ("
                    "SELECT in FROM fact_involves WHERE out IN $eids)",
                    {"gid": gid, "eids": [eid_to_rid[eid] for eid in c_eids]}
                )
                fact_rows = _flatten_query(fact_res)
                c_facts = [r["fact_text"] for r in fact_rows if r.get("fact_text")]
            except Exception:
                c_facts = []

            # Summarize using LLM
            facts_bullet = "\n".join(f"- {f}" for f in c_facts[:15])
            prompt = f"""Berikut adalah entitas dan fakta yang saling berhubungan erat dalam graf sosial/konseptual pengguna:
Entitas: {", ".join(c_names)}
Fakta-fakta:
{facts_bullet}

Tolong bantu membuat:
1. Nama singkat (maksimal 3 kata) yang menggambarkan lingkaran sosial/konseptual ini (misal: "Proyek Skincare", "Teman Kuliah Bandung", dsb).
2. Ringkasan singkat (1-3 kalimat) dalam Bahasa Indonesia yang merangkum hubungan dan konteks kelompok ini.

Keluarkan HANYA JSON objek valid berikut tanpa penjelasan:
{{"name": "Nama Kelompok", "summary": "Ringkasan penjelasan kelompok"}}
"""
            try:
                raw_resp = await self._call_llm(prompt)
                cleaned = _strip_json_fence(raw_resp)
                data = json.loads(cleaned)
                c_name = data.get("name", f"Kelompok {c_idx + 1}")
                c_summary = data.get("summary", "Anggota saling terhubung.")
            except Exception as e:
                logger.warning("LLM community summary failed: %s", e)
                c_name = f"Kelompok {c_idx + 1}"
                c_summary = f"Kelompok berisi: {', '.join(c_names[:5])}"

            # Save community node
            c_uuid = str(uuid.uuid4())
            c_rid = RecordID("community", c_uuid)
            try:
                await self._db.upsert(
                    c_rid,
                    {
                        "group_id": gid,
                        "name": c_name,
                        "summary": c_summary,
                        "created_at": created_at,
                    }
                )
                
                # Relate to members
                for eid in c_eids:
                    await self._db.query(
                        "RELATE $comm->has_member->$ent",
                        {"comm": c_rid, "ent": eid_to_rid[eid]}
                    )
                community_count += 1
                logger.info("Successfully built community: '%s' with %d members", c_name, len(c_eids))
            except Exception as e:
                logger.error("Failed to save community node: %s", e)

        return community_count

    async def search_with_temporal_filter(
        self,
        query: str,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        num_results: int = 10,
    ) -> List[SearchResult]:
        """Filter ``valid_at`` in SurrealQL (via ``search``) instead of post-filtering in Python."""
        return await self.search(
            query=query,
            num_results=num_results,
            valid_before=before,
            valid_after=after,
        )

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
                    "SELECT id, fact_text, entity_names, valid_at, created_at, source_description "
                    "FROM extracted_fact WHERE group_id = $gid AND invalid_at IS NONE AND id IN ("
                    "SELECT VALUE in FROM fact_involves WHERE out = $eid) "
                    "ORDER BY created_at DESC LIMIT $lim",
                    {"gid": gid, "eid": eid, "lim": limit},
                )
                rows = _flatten_query(res)
            except Exception as e:
                logger.debug("get_entity_facts graph traverse failed: %s", e)
                rows = []

        if not rows:
            fb = (
                "SELECT id, fact_text, entity_names, valid_at, created_at, source_description "
                "FROM extracted_fact WHERE group_id = $gid AND invalid_at IS NONE AND array::contains(entity_names, $name) "
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
            meta = {"source": "entity_expand"}
            rid = r.get("id")
            if rid is not None:
                meta["fact_record_id"] = str(rid)
            out.append(
                SearchResult(
                    fact=str(r.get("fact_text", "")),
                    score=1.0,
                    entity_name=str(en) if en else entity_name,
                    created_at=r.get("created_at") if isinstance(r.get("created_at"), datetime) else None,
                    valid_at=r.get("valid_at") if isinstance(r.get("valid_at"), datetime) else None,
                    source_description=r.get("source_description"),
                    metadata=meta,
                )
            )
        return out

    async def retrieve_episodes_window(
        self,
        reference_time: datetime,
        last_n: int = 3,
        group_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Episodes with ``reference_time`` at or before the given instant, newest first (Graphiti-style window).
        """
        if self._db is None:
            return []
        gid = group_id or self.group_id
        ref = _as_utc(reference_time)
        sql = (
            "SELECT id, name, body, source_description, reference_time, created_at FROM episode "
            "WHERE group_id = $gid AND reference_time <= $ref "
            "ORDER BY reference_time DESC LIMIT $lim"
        )
        try:
            res = await self._db.query(sql, {"gid": gid, "ref": ref, "lim": last_n})
        except Exception as e:
            logger.warning("retrieve_episodes_window: %s", e)
            return []
        return _flatten_query(res)

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
