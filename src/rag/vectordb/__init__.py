# src/vectordb/__init__.py
"""
Vector Database Module
======================
ChromaDB untuk Vanilla RAG (pure vector similarity search).
"""

from .chroma_client import (
    ChromaVectorDB,
    VanillaDocument,
    VanillaSearchResult,
    get_chroma_client,
)

__all__ = [
    "ChromaVectorDB",
    "VanillaDocument",
    "VanillaSearchResult",
    "get_chroma_client",
]

