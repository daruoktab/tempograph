"""SurrealDB persistence (vanilla vectors + agentic graph facts)."""

from .connection import connect_surreal, apply_schema
from .vanilla_store import SurrealVanillaVectorDB, get_surreal_vanilla_client

__all__ = [
    "connect_surreal",
    "apply_schema",
    "SurrealVanillaVectorDB",
    "get_surreal_vanilla_client",
]
