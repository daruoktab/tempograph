# src/rag/vectordb/__init__.py
"""
Vector store for Vanilla RAG (SurrealDB MTREE cosine; legacy name chroma kept as alias).
"""

from ..surreal.vanilla_store import (
    SESSION_PASSAGE_TABLE,
    SurrealVanillaVectorDB,
    VanillaDocument,
    VanillaSearchResult,
    get_surreal_vanilla_client,
)

ChromaVectorDB = SurrealVanillaVectorDB
get_chroma_client = get_surreal_vanilla_client

__all__ = [
    "ChromaVectorDB",
    "SESSION_PASSAGE_TABLE",
    "SurrealVanillaVectorDB",
    "VanillaDocument",
    "VanillaSearchResult",
    "get_chroma_client",
    "get_surreal_vanilla_client",
]
