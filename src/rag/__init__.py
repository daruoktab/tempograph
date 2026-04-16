# rag/__init__.py
"""RAG system core module.

Heavy imports (graph client, embedders) are lazy so ``src.rag.surreal.*`` can be
used from lightweight tooling (e.g. ``scripts/test_surreal_connection.py``) without pulling Gemini.
"""

from __future__ import annotations

import typing as _t

__all__ = ["TemporalGraphClient", "SearchResult"]


def __getattr__(name: str) -> _t.Any:
    if name == "TemporalGraphClient":
        from .graph_client import TemporalGraphClient

        return TemporalGraphClient
    if name == "SearchResult":
        from .graph_client import SearchResult

        return SearchResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
