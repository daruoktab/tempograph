#!/usr/bin/env python3
"""Export first N sessions (+ user block) for ``web/sim`` static demo.

Sumber: ``output/example_dataset/conversation_dataset.json`` (isi setara .toon di repo).
Output: ``web/sim/public/dataset/conversation_sim_10.json`` + ``user_events.json`` (salinan penuh).

Usage (dari root repo):
    python scripts/export_sim_ui_dataset.py
    python scripts/export_sim_ui_dataset.py --sessions 10 --source path/to/conversation_dataset.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=int, default=10, help="Number of sessions to keep (default 10)")
    p.add_argument(
        "--source",
        type=Path,
        default=root / "output" / "example_dataset" / "conversation_dataset.json",
    )
    p.add_argument(
        "--events",
        type=Path,
        default=root / "output" / "example_dataset" / "user_events.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "web" / "sim" / "public" / "dataset",
    )
    args = p.parse_args()
    src: Path = args.source
    if not src.is_file():
        print(f"Missing source JSON: {src}", flush=True)
        return 1
    with src.open(encoding="utf-8") as f:
        data = json.load(f)
    sessions = data.get("sessions") or []
    if len(sessions) < args.sessions:
        print(f"Warning: only {len(sessions)} sessions in source", flush=True)
    slim = {
        "user": data.get("user"),
        "secondary_personas": data.get("secondary_personas") or [],
        "sessions": sessions[: args.sessions],
    }
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "conversation_sim_10.json"
    out_json.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} ({len(slim['sessions'])} sessions)", flush=True)

    ev = args.events
    if ev.is_file():
        dest_ev = out_dir / "user_events.json"
        shutil.copyfile(ev, dest_ev)
        print(f"Copied events -> {dest_ev}", flush=True)
    else:
        print(f"(skip) no user_events at {ev}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
