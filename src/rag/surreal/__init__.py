"""SurrealDB persistence (connection + vanilla session vectors).

Agentic fact graph lives in ``fact_graph``; import it explicitly to avoid loading
embedders in lightweight tooling (e.g. ``scripts/test_surreal_connection.py``).
"""

from .connection import apply_schema, connect_surreal
from .vanilla_store import SurrealVanillaVectorDB, get_surreal_vanilla_client

__all__ = [
    "apply_schema",
    "connect_surreal",
    "get_surreal_vanilla_client",
    "SurrealVanillaVectorDB",
]
