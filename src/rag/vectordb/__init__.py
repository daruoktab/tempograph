# src/rag/vectordb/__init__.py
"""Public entry for dense session retrieval (implementation: ``surreal.vanilla_store``)."""

from ..surreal.vanilla_store import (
    SESSION_PASSAGE_TABLE,
    SurrealVanillaVectorDB,
    VanillaDocument,
    VanillaSearchResult,
    get_surreal_vanilla_client,
)

__all__ = [
    "SESSION_PASSAGE_TABLE",
    "SurrealVanillaVectorDB",
    "VanillaDocument",
    "VanillaSearchResult",
    "get_surreal_vanilla_client",
]
