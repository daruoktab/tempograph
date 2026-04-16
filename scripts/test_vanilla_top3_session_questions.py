#!/usr/bin/env python3
"""Baca 10 sesi awal example dataset (verifikasi), lalu uji Vanilla RAG: cetak top-3 per pertanyaan.

Pertanyaan ditulis manual dari isi ``output/example_dataset/conversation_dataset.json``
(sesi 1–10: skincare, fashion, kopi, kafe Dago, Eisenhower, meeting Kopi Senja, brainstorm, creative block, Rizky, Dago refresh).

**Urutan wajib (jangan balik): ingestion dulu, baru tes.**

1. SurrealDB jalan (satu terminal dibiarkan terbuka), dari root repo + ``.env``:

       python scripts/run_with_local_surreal.py --serve-only

2. **Ingest** 10 sesi ke koleksi ``vanilla_gemini`` (tabel ``session_passage``), terminal lain:

       python scripts/run_with_local_surreal.py --no-start -- python scripts/ingest_vanilla.py --setup gemini --limit 10

   Atau unified (graph + ``session_passage``): ``ingest_agentic.py --setup gemini --limit 10``.

   (Opsional reset: ``--clear`` pada ``ingest_agentic`` / ``ingest_vanilla``.)

3. **Baru tes** top-3:

       python scripts/run_with_local_surreal.py --no-start -- python scripts/test_vanilla_top3_session_questions.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def _load_dotenv() -> None:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(_REPO / ".env")


def _print_dataset_header() -> None:
    p = _REPO / "output" / "example_dataset" / "conversation_dataset.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    sessions = data.get("sessions", [])
    head = sessions[:10]
    print(f"Dataset: {p}")
    print(f"  Total sessions: {len(sessions)} | first-10 session_ids: {[s['session_id'] for s in head]}")
    turns_10 = sum(len(s["turns"]) for s in head)
    print(f"  Turns in first 10 sessions: {turns_10}\n")


# (tag, rough target session id for your sanity, question)
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


async def main() -> int:
    os.chdir(_REPO)
    _load_dotenv()
    _print_dataset_header()

    from src.config.experiment_setups import SETUP_1V_VANILLA_GEMINI
    from src.rag.retrieval.vanilla_retriever import create_vanilla_retriever

    retriever = await create_vanilla_retriever(SETUP_1V_VANILLA_GEMINI)
    n = retriever.db.count()
    print(f"Retriever: `{retriever.db.collection_name}` | documents: {n}\n")
    if n == 0:
        print("Koleksi kosong — jalankan ingest dulu.")
        return 1

    for tag, hint_sess, query in SESSION_QUERIES:
        print("=" * 78)
        print(f"[{tag}] ~sesi {hint_sess}")
        print(f"Q: {query}\n")
        out = await retriever.retrieve(query)
        top = out.results[:3]
        if not top:
            print("  (tidak ada hasil)\n")
            continue
        for i, r in enumerate(top, 1):
            meta = r.metadata or {}
            sid = meta.get("session_id", "?")
            preview = (r.text or "").replace("\n", " ").strip()
            if len(preview) > 360:
                preview = preview[:360] + "…"
            print(f"  #{i}  score={r.score:.4f}  session_id={sid}")
            print(f"      doc_id: {r.id}")
            print(f"      {preview}\n")
        await asyncio.sleep(0.05)

    print("Selesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
