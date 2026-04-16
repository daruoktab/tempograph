#!/usr/bin/env python3
"""Uji Agentic dan/atau Hybrid (Gemini) dengan 5 pertanyaan yang sama seperti skrip vanilla top-3.

- **Agentic:** ``RetrievalAgent.retrieve()`` → cetak **3 fakta** pertama (sama pola eval).
- **Hybrid:** ``HybridRetriever.retrieve()`` → cetak **3 baris** pertama (graph ``[FACT]`` / vanilla ``[DETAIL]``).

**Urutan wajib: ingestion dulu, baru tes.** (Tanpa ``extracted_fact``, agentic/hybrid graph kosong.)

1. SurrealDB jalan (satu terminal):

       python scripts/run_with_local_surreal.py --serve-only

2. **Ingest unified** (``session_passage`` + graph ``agentic_gemini`` — cukup **sekali**):

       python scripts/run_with_local_surreal.py --no-start -- python scripts/ingest_agentic.py --setup gemini --limit 10

   (Hanya vektor sesi tanpa graph: ``ingest_vanilla.py``; hybrid butuh graph + ``session_passage``.)

3. **Baru jalankan tes** (``--no-start`` = Surreal sudah jalan di langkah 1):

       python scripts/run_with_local_surreal.py --no-start -- python scripts/test_agentic_hybrid_top3_questions.py
       python scripts/run_with_local_surreal.py --no-start -- python scripts/test_agentic_hybrid_top3_questions.py --mode agentic
       python scripts/run_with_local_surreal.py --no-start -- python scripts/test_agentic_hybrid_top3_questions.py --mode hybrid

Satu proses alternatif (start Surreal + ingest + tes tanpa ``--serve-only``): bungkus tiap perintah dengan
``python scripts/run_with_local_surreal.py -- …`` (Surreal mati setelah child selesai — untuk tes GUI pakai langkah 1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Sama dengan scripts/test_vanilla_top3_session_questions.py (sesi 1,2,4,5,9)
SESSION_QUERIES: list[tuple[str, str, str]] = [
    (
        "S1-skincare",
        "1",
        "Apa yang paling menantang bagi Aisha saat menyiapkan campaign produk skincare untuk klien baru terkait visual dan tone of voice?",
    ),
    (
        "S2-fashion",
        "2",
        "Proyek kampanye klien brand fashion lokal milik Aisha ingin tampilan seperti apa, dan apa yang membuat visual dan copy sulit ditentukan?",
    ),
    (
        "S4-kafe",
        "4",
        "Siapa nama kafe di Dago Atas yang dikunjungi Aisha dan bagaimana keunikan signature kopinya?",
    ),
    (
        "S5-eisenhower",
        "5",
        "Metode manajemen prioritas apa yang Aisha pelajari dari buku dan bagaimana ia menerapkannya pada tugas proyek Kopi Senja Pagi?",
    ),
    (
        "S9-rizky",
        "9",
        "Saran Rizky kepada Aisha dan Dewi agar mengatasi creative block pada proyek fashion itu apa?",
    ),
]


def _load_dotenv() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(_REPO / ".env")


def _print_dataset_header() -> None:
    p = _REPO / "output" / "example_dataset" / "conversation_dataset.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])
    head = sessions[:10]
    print(f"Dataset: {p}")
    print(f"  first-10 session_ids: {[s['session_id'] for s in head]}\n")


def _gemini_sufficiency_llm_stub(model_name: str, api_key: str):
    return type(
        "SuffGeminiCfg",
        (),
        {
            "config": type(
                "C",
                (),
                {"model": model_name, "api_key": api_key, "base_url": None},
            )()
        },
    )()


async def _count_facts(tc) -> int:
    from src.rag.graph_client import _flatten_query  # noqa: PLC0415

    if tc._db is None:
        return 0
    res = await tc._db.query(
        "SELECT count() AS c FROM extracted_fact WHERE group_id = $gid GROUP ALL",
        {"gid": tc.group_id},
    )
    rows = _flatten_query(res)
    if not rows or rows[0].get("c") is None:
        return 0
    return int(rows[0]["c"])


async def _count_all_facts(tc) -> int:
    """Total extracted_fact rows (semua group) — untuk debug."""
    from src.rag.graph_client import _flatten_query  # noqa: PLC0415

    if tc._db is None:
        return 0
    res = await tc._db.query("SELECT count() AS c FROM extracted_fact GROUP ALL")
    rows = _flatten_query(res)
    if not rows or rows[0].get("c") is None:
        return 0
    return int(rows[0]["c"])


def _make_graph_adapter(tc, group_id: str):
    from src.rag.graph_client import SearchResult

    class GraphitiClientAdapter:
        def __init__(self, graphiti, gid: str):
            self.graphiti = graphiti
            self.group_id = gid

        async def search(self, query: str, num_results: int = 10):
            results = await self.graphiti.search(
                query=query, group_ids=[self.group_id], num_results=num_results
            )
            if not results:
                return []
            return [
                SearchResult(
                    fact=r.fact,
                    score=getattr(r, "score", 0.8),
                    entity_name=getattr(r, "entity_name", None),
                    created_at=getattr(r, "created_at", None),
                    valid_at=getattr(r, "valid_at", None),
                )
                for r in results
            ]

        async def search_with_temporal_filter(
            self, query: str, before=None, after=None, num_results: int = 10
        ):
            return await self.search(query, num_results)

        async def get_entity_facts(self, entity_name: str):
            return await self.search(entity_name, num_results=5)

    return GraphitiClientAdapter(tc, group_id)


class _AgentGraphWrapper:
    """Sama konsep dengan evaluate_agentic: graph side = RetrievalAgent."""

    def __init__(self, agent):
        self.agent = agent

    async def search(self, query, num_results=10):
        result = await self.agent.retrieve(query)
        return result.facts[:num_results]


async def _build_retrieval_agent(tc, group_id: str):
    from src.config.settings import RetrievalConfig, get_config
    from src.rag.retrieval.agent import RetrievalAgent

    config = get_config()
    model_name = "gemini-2.5-flash"
    llm_client = _gemini_sufficiency_llm_stub(
        model_name, config.gemini.api_key
    )
    adapter = _make_graph_adapter(tc, group_id)
    retrieval_config = RetrievalConfig(
        max_iterations=5, num_results=5, similarity_threshold=0.3
    )
    return RetrievalAgent(adapter, retrieval_config, llm_client=llm_client)


async def _run_agentic(agent) -> None:
    print("\n" + "=" * 78)
    print("MODE: AGENTIC (RetrievalAgent → fakta ter-ranked)")
    print("=" * 78)
    for tag, hint_sess, query in SESSION_QUERIES:
        print("\n" + "-" * 78)
        print(f"[{tag}] ~sesi {hint_sess}")
        print(f"Q: {query}\n")
        res = await agent.retrieve(query)
        top = res.facts[:3]
        print(
            f"  iterations={res.iterations}  query_type={res.query_type.value}  "
            f"facts_total={len(res.facts)}"
        )
        if not top:
            print("  (tidak ada fakta)\n")
            continue
        for i, f in enumerate(top, 1):
            fact = (f.fact or "").replace("\n", " ").strip()
            if len(fact) > 420:
                fact = fact[:420] + "…"
            ent = f.entity_name or "—"
            print(f"  #{i}  score={f.score:.4f}  entity={ent}")
            print(f"      {fact}\n")
        await asyncio.sleep(0.1)


async def _run_hybrid(hybrid) -> None:
    print("\n" + "=" * 78)
    print("MODE: HYBRID (graph via agent + vanilla; 3 baris pertama merged)")
    print("=" * 78)
    for tag, hint_sess, query in SESSION_QUERIES:
        print("\n" + "-" * 78)
        print(f"[{tag}] ~sesi {hint_sess}")
        print(f"Q: {query}\n")
        rows = await hybrid.retrieve(query, limit=10)
        top = rows[:3]
        if not top:
            print("  (kosong)\n")
            continue
        for i, r in enumerate(top, 1):
            body = (r.content or "").replace("\n", " ").strip()
            if len(body) > 480:
                body = body[:480] + "…"
            print(f"  #{i}  [{r.source_type}] score={r.score:.4f}")
            print(f"      {body}\n")
        await asyncio.sleep(0.1)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=("agentic", "hybrid", "both"),
        default="both",
        help="agentic saja, hybrid saja, atau keduanya (default: both)",
    )
    args = parser.parse_args()

    os.chdir(_REPO)
    _load_dotenv()
    _print_dataset_header()

    from src.config.experiment_setups import (
        SETUP_1A_AGENTIC_GEMINI,
        SETUP_1H_HYBRID_GEMINI,
        SETUP_1V_VANILLA_GEMINI,
    )
    from src.rag.graph_client import TemporalGraphClient
    from src.rag.retrieval.hybrid_retriever import HybridRetriever
    from src.rag.retrieval.vanilla_retriever import create_vanilla_retriever

    # Pakai setup agentic untuk koneksi DB (group + embedder sama dengan hybrid graph)
    tc = TemporalGraphClient(setup=SETUP_1A_AGENTIC_GEMINI)
    await tc.initialize()
    gid = tc.group_id
    n_facts = await _count_facts(tc)
    n_all = await _count_all_facts(tc)
    print(f"Surreal: group_id={gid!r} | extracted_fact (group ini) = {n_facts}")
    if n_all != n_facts:
        print(f"  (total extracted_fact di DB, semua group) = {n_all}")
    if n_facts == 0:
        print(
            "\n⚠️  Belum ada fakta untuk group agentic Gemini ini.\n"
            "   ``ingest_agentic.py`` mengisi ``extracted_fact`` + ``episode`` + ``session_passage`` (dense).\n"
            "   Jalankan (satu proses dengan Surreal, atau ``--serve-only`` + ``--no-start``):\n"
            "     python scripts/run_with_local_surreal.py -- python scripts/ingest_agentic.py --setup gemini --limit 10\n"
            "   Pakai SurrealKV ``data/surreal_local`` yang sama supaya data tidak hilang.\n"
        )
        await tc.close()
        return 1

    agent = await _build_retrieval_agent(tc, gid)

    if args.mode in ("agentic", "both"):
        await _run_agentic(agent)

    if args.mode in ("hybrid", "both"):
        vanilla = await create_vanilla_retriever(SETUP_1V_VANILLA_GEMINI)
        if vanilla.db.count() == 0:
            print("\n⚠️  Hybrid butuh vanilla_gemini terisi. Lewati hybrid.")
        else:
            wrapper = _AgentGraphWrapper(agent)
            hybrid = HybridRetriever(
                graph_client=wrapper,
                vanilla_retriever=vanilla,
                setup=SETUP_1H_HYBRID_GEMINI,
            )
            await hybrid.initialize()
            await _run_hybrid(hybrid)

    await tc.close()
    print("\nSelesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
