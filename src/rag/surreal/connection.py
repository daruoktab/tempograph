"""SurrealDB async connection and schema bootstrap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from surrealdb import AsyncSurreal

from ...config.settings import SurrealDBConfig, get_config

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.surql")


async def connect_surreal(cfg: SurrealDBConfig | None = None) -> Any:
    """Connect, authenticate, and USE namespace + database.

    Return type is ``Any`` because ``AsyncSurreal`` in package stubs is a factory
    callable, not a generic class, so ``AsyncSurreal[...]`` is not valid for checkers.
    """
    c = cfg or get_config().surreal
    db = cast(Any, AsyncSurreal(c.url))
    await db.connect()
    await db.signin({"username": c.username, "password": c.password})
    await db.use(c.namespace, c.database)
    logger.info("SurrealDB connected %s / %s", c.namespace, c.database)
    return db


async def apply_schema(db: Any) -> None:
    """Idempotent schema (DEFINE TABLE/FIELD/INDEX)."""
    if not _SCHEMA_PATH.is_file():
        logger.warning("schema.surql missing at %s", _SCHEMA_PATH)
        return
    raw = _SCHEMA_PATH.read_text(encoding="utf-8")
    for stmt in raw.split(";"):
        stmt = stmt.strip()
        if not stmt or stmt.startswith("--"):
            continue
        try:
            await db.query(stmt)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "found" in msg and "index" in msg:
                continue
            logger.debug("schema stmt note: %s: %s", stmt[:60], e)
