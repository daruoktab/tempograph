"""SurrealDB-backed vector store for Vanilla RAG (replaces ChromaDB)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from surrealdb import RecordID

from .connection import connect_surreal, apply_schema

logger = logging.getLogger(__name__)

# Surreal table for one embedding vector per session document (dense / "vanilla" retrieval)
SESSION_PASSAGE_TABLE = "session_passage"

CHROMA_PERSIST_DIR = "./data/chroma"  # legacy constant; Surreal ignores path
COLLECTION_VANILLA_GEMINI = "vanilla_gemini"
COLLECTION_VANILLA_GEMMA = "vanilla_gemma"


@dataclass
class VanillaDocument:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VanillaSearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SurrealVanillaVectorDB:
    """Same responsibilities as ChromaVectorDB but stored in SurrealDB."""

    def __init__(self, collection_name: str, persist_directory: str = CHROMA_PERSIST_DIR):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._db: Any | None = None
        self._embedder = None
        self._doc_count: int = 0

    def _safe_rid_suffix(self, doc_id: str) -> str:
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in doc_id)[:180]

    async def _refresh_doc_count(self) -> None:
        if self._db is None:
            self._doc_count = 0
            return
        res = await self._db.query(
            f"SELECT count() AS c FROM {SESSION_PASSAGE_TABLE} WHERE collection = $coll GROUP ALL",
            {"coll": self.collection_name},
        )
        rows = _flatten_query(res)
        if rows and rows[0].get("c") is not None:
            self._doc_count = int(rows[0]["c"])
        else:
            self._doc_count = 0

    async def initialize(self, embedder=None):
        self._embedder = embedder
        self._db = await connect_surreal()
        await apply_schema(self._db)
        await self._refresh_doc_count()
        logger.info(
            "SurrealDB vanilla collection '%s' ready (%s documents)",
            self.collection_name,
            self._doc_count,
        )

    async def add_documents(
        self, documents: List[VanillaDocument], show_progress: bool = True
    ) -> int:
        if not documents or self._db is None:
            return 0
        from tqdm import tqdm

        iterator = tqdm(documents, desc="Adding documents") if show_progress else documents
        for doc in iterator:
            emb = doc.embedding
            if emb is None:
                if self._embedder is None:
                    raise ValueError("No embedding and no embedder")
                emb_res = await self._embedder.embed([doc.text])
                emb = emb_res.embeddings[0] if emb_res.embeddings else []
            rid = RecordID(SESSION_PASSAGE_TABLE, self._safe_rid_suffix(doc.id))
            await self._db.upsert(
                rid,
                {
                    "collection": self.collection_name,
                    "doc_id": doc.id,
                    "text": doc.text,
                    "embedding": list(float(x) for x in emb),
                    "metadata": doc.metadata or {},
                },
            )
        await self._refresh_doc_count()
        logger.info("Added %s documents to '%s'", len(documents), self.collection_name)
        return len(documents)

    async def add_document(self, document: VanillaDocument) -> bool:
        n = await self.add_documents([document], show_progress=False)
        return n > 0

    async def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[VanillaSearchResult]:
        if self._db is None:
            raise RuntimeError("Not initialized")
        if query_embedding:
            q_emb = query_embedding
        elif self._embedder:
            q_emb = await self._embedder.embed(query)
        else:
            raise ValueError("No query embedding and no embedder")

        qv = list(float(x) for x in q_emb)
        sql = (
            "SELECT doc_id, text, metadata, "
            "vector::similarity::cosine(embedding, $qv) AS score "
            f"FROM {SESSION_PASSAGE_TABLE} WHERE collection = $coll "
            "ORDER BY score DESC LIMIT $lim"
        )
        res = await self._db.query(
            sql, {"qv": qv, "coll": self.collection_name, "lim": n_results}
        )
        rows = _flatten_query(res)
        out: List[VanillaSearchResult] = []
        for r in rows:
            out.append(
                VanillaSearchResult(
                    id=str(r.get("doc_id", "")),
                    text=str(r.get("text", "")),
                    score=float(r.get("score", 0.0)),
                    metadata=dict(r.get("metadata") or {}),
                )
            )
        return out

    async def get_document(self, doc_id: str) -> Optional[VanillaDocument]:
        if self._db is None:
            return None
        sql = (
            f"SELECT * FROM {SESSION_PASSAGE_TABLE} WHERE collection = $coll AND doc_id = $id LIMIT 1"
        )
        res = await self._db.query(sql, {"coll": self.collection_name, "id": doc_id})
        rows = _flatten_query(res)
        if not rows:
            return None
        r = rows[0]
        return VanillaDocument(
            id=str(r.get("doc_id", doc_id)),
            text=str(r.get("text", "")),
            embedding=r.get("embedding"),
            metadata=dict(r.get("metadata") or {}),
        )

    def count(self) -> int:
        return self._doc_count

    async def clear(self):
        if self._db is None:
            return
        await self._db.query(
            f"DELETE FROM {SESSION_PASSAGE_TABLE} WHERE collection = $coll",
            {"coll": self.collection_name},
        )
        self._doc_count = 0
        logger.warning(
            "Cleared SurrealDB %s for collection '%s'",
            SESSION_PASSAGE_TABLE,
            self.collection_name,
        )

    async def close(self) -> None:
        if self._db is not None:
            try:
                await self._db.close()
            except Exception as e:
                logger.debug("vanilla store close: %s", e)
            self._db = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "backend": "surrealdb",
            "document_count": self._doc_count,
        }


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


_instances: Dict[str, SurrealVanillaVectorDB] = {}


def get_surreal_vanilla_client(
    collection_name: str, persist_directory: str = CHROMA_PERSIST_DIR
) -> SurrealVanillaVectorDB:
    key = f"{persist_directory}:{collection_name}"
    if key not in _instances:
        _instances[key] = SurrealVanillaVectorDB(
            collection_name=collection_name, persist_directory=persist_directory
        )
    return _instances[key]


def get_vanilla_gemini_db() -> SurrealVanillaVectorDB:
    return get_surreal_vanilla_client(COLLECTION_VANILLA_GEMINI)


def get_vanilla_gemma_db() -> SurrealVanillaVectorDB:
    return get_surreal_vanilla_client(COLLECTION_VANILLA_GEMMA)
