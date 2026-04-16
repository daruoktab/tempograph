#!/usr/bin/env python3
"""Smoke test: SurrealDB sign-in, schema bootstrap, and a trivial read/write.

Run from repo root (with .env or env vars set):

    conda activate porto-skripsi
    python scripts/test_surreal_connection.py

SURREAL_URL should be like ws://127.0.0.1:8000 (no /rpc; the SDK appends it).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from src.config.settings import SurrealDBConfig
from src.rag.surreal.connection import apply_schema, connect_surreal


async def main(url: str | None) -> int:
    cfg = SurrealDBConfig(url=url) if url else SurrealDBConfig()
    print("Config:", cfg.url, cfg.namespace, cfg.database, "(user:", cfg.username + ")")

    db = await connect_surreal(cfg)
    try:
        await apply_schema(db)
        print("Schema apply finished (idempotent; warnings for 'already exists' are OK).")

        info = await db.query("INFO FOR DB;")
        print("INFO FOR DB:", info)

        await db.query(
            "CREATE _smoke_test CONTENT { note: 'scripts/test_surreal_connection.py', at: time::now() };"
        )
        rows = await db.query("SELECT * FROM _smoke_test LIMIT 1;")
        print("Smoke CREATE/SELECT:", rows)
        await db.query("REMOVE TABLE IF EXISTS _smoke_test;")
        print("OK: connection, schema, and read/write succeeded.")
        return 0
    finally:
        await db.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default=None,
        help="Override SURREAL_URL (default from env, e.g. ws://127.0.0.1:8000)",
    )
    args = p.parse_args()
    url = args.url or os.getenv("SURREAL_URL")
    raise SystemExit(asyncio.run(main(url)))
