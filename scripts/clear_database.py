#!/usr/bin/env python3
"""Hapus data SurrealDB untuk RAG (graph + dense passages).

- **Default / ``--group``:** hapus hanya baris dengan ``group_id`` itu (episode, facts, entities, edge terkait).
- **``--all``:** hapus **semua** baris di tabel eksperimen (semua group + semua koleksi ``session_passage`` + legacy ``vanilla_chunk``).

Butuh: ``.env``, SurrealDB jalan (mis. ``run_with_local_surreal.py --serve-only``).

Contoh:

    python scripts/run_with_local_surreal.py --no-start -- python scripts/clear_database.py
    python scripts/run_with_local_surreal.py --no-start -- python scripts/clear_database.py --group agentic_gemma
    python scripts/run_with_local_surreal.py --no-start -- python scripts/clear_database.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def _load_dotenv() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(_REPO / ".env")


async def _run(group_id: str) -> int:
    from src.rag.surreal.fact_graph import TemporalGraphClient

    client = TemporalGraphClient(group_id=group_id)
    try:
        await client.initialize()
        await client.clear_group()
        print(f"OK: cleared Surreal tables for group_id={group_id!r}")
        return 0
    finally:
        await client.close()


async def _purge_all_tables() -> int:
    """Truncate all RAG-related tables in the current NS/DB (any group_id / collection)."""
    from src.rag.surreal.connection import connect_surreal
    from src.rag.surreal.vanilla_store import SESSION_PASSAGE_TABLE

    db = await connect_surreal()
    try:
        # Edges first, then nodes / passages
        for table in (
            "has_fact",
            "fact_involves",
            "extracted_fact",
            "episode",
            "entity",
            SESSION_PASSAGE_TABLE,
            "vanilla_chunk",
        ):
            try:
                await db.query(f"DELETE FROM {table}")
                print(f"OK: DELETE FROM {table}")
            except Exception as e:
                print(f"SKIP {table}: {e}")
        print("OK: purge complete (all rows in RAG tables above).")
        return 0
    finally:
        await db.close()


def main() -> int:
    from src.config.experiment_setups import NEO4J_GROUP_IDS, SetupType

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--all",
        action="store_true",
        help="Delete every row in RAG tables (full wipe for this NS/DB)",
    )
    p.add_argument(
        "--group",
        default=NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMINI],
        metavar="GROUP_ID",
        help=f"group_id when not using --all (default: {NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMINI]})",
    )
    args = p.parse_args()

    os.chdir(_REPO)
    _load_dotenv()
    if args.all:
        return asyncio.run(_purge_all_tables())
    return asyncio.run(_run(args.group))


if __name__ == "__main__":
    raise SystemExit(main())
