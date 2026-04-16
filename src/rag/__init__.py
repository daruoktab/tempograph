# rag/__init__.py
"""RAG system core module.

Heavy imports (fact graph client, embedders) are lazy so ``src.rag.surreal.*`` can be
used from lightweight tooling (e.g. ``scripts/test_surreal_connection.py``) without pulling Gemini.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .surreal.fact_graph import TemporalGraphClient, SearchResult

__all__ = ["TemporalGraphClient", "SearchResult"]


def __getattr__(name: str) -> Any:
    if name == "TemporalGraphClient":
        from .surreal.fact_graph import TemporalGraphClient

        return TemporalGraphClient
    if name == "SearchResult":
        from .surreal.fact_graph import SearchResult

        return SearchResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
