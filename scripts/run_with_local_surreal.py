#!/usr/bin/env python3
"""Start SurrealDB from repo root using ``.env``, then run another command.

Menghindari keharusan mengetik ``surreal start ...`` manual (dan salah cwd).
Selalu resolve repo root dari lokasi skrip ini, lalu ``load_dotenv(repo/.env)``.

Usage (dari root repo):

    python scripts/run_with_local_surreal.py -- python scripts/test_surreal_connection.py
    python scripts/run_with_local_surreal.py -- python scripts/ingest_vanilla.py

    # Terminal terpisah: biarkan Surreal hidup untuk Surrealist / skrip lain (Ctrl+C stop):
    python scripts/run_with_local_surreal.py --serve-only

Tanpa argumen setelah ``--``, default: smoke test koneksi Surreal.

Storage default: SurrealKV on-disk di ``data/surreal_local`` (relatif repo root).
Override dengan env ``SURREAL_START_PATH`` (contoh: ``memory`` atau ``surrealkv://data/mydb``).
Uji cepat tanpa disk: ``python scripts/run_with_local_surreal.py --memory -- ...``.

Requires: binary ``surreal`` di PATH (install SurrealDB CLI), atau set ``SURREAL_CLI``.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _terminate_surreal(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError as e:
        print("Install python-dotenv: uv pip install python-dotenv", file=sys.stderr)
        raise SystemExit(1) from e
    load_dotenv(_REPO_ROOT / ".env")


def _parse_bind_from_surreal_url(url: str) -> tuple[str, int]:
    u = urlparse(url)
    if u.scheme not in ("ws", "wss", "http", "https"):
        raise SystemExit(
            f"SURREAL_URL must start with ws:// or wss:// (got scheme {u.scheme!r}): {url!r}"
        )
    host = u.hostname or "127.0.0.1"
    if u.port is not None:
        port = u.port
    elif u.scheme in ("wss", "https"):
        port = 443
    elif u.scheme in ("ws", "http"):
        port = 8000
    else:
        port = 8000
    return host, port


def _wait_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)
    raise SystemExit(f"Timeout: nothing listening on {host}:{port} after {timeout_s}s")


def main() -> int:
    os.chdir(_REPO_ROOT)
    _load_env()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--no-start",
        action="store_true",
        help="Skip starting Surreal (assume server already up on SURREAL_URL port).",
    )
    p.add_argument(
        "--bind-override",
        default=None,
        metavar="HOST:PORT",
        help="Override --bind for surreal start (default from SURREAL_URL).",
    )
    p.add_argument(
        "--memory",
        action="store_true",
        help="Use in-memory storage (default is SurrealKV under data/surreal_local).",
    )
    p.add_argument(
        "--serve-only",
        action="store_true",
        help="Start Surreal and keep the process running (e.g. Surrealist). Ctrl+C stops the server. No child command.",
    )
    p.add_argument(
        "child",
        nargs=argparse.REMAINDER,
        help="Command after optional '--' (default: smoke test).",
    )
    args = p.parse_args()
    if args.child and args.child[0] == "--":
        args.child = args.child[1:]

    if args.serve_only and args.no_start:
        print("Cannot use --serve-only together with --no-start.", file=sys.stderr)
        return 2
    if args.serve_only and args.child:
        print("Note: --serve-only ignores the child command; open another terminal for ingest/eval.", file=sys.stderr)

    surreal_url = os.getenv("SURREAL_URL", "ws://127.0.0.1:8000")
    user = os.getenv("SURREAL_USER", "root")
    password = os.getenv("SURREAL_PASS", "root")
    surreal_bin = os.getenv("SURREAL_CLI") or shutil.which("surreal")

    child: list[str] | None = None
    if not args.serve_only:
        child = args.child if args.child else [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "test_surreal_connection.py"),
        ]

    proc: subprocess.Popen[bytes] | None = None
    if not args.no_start:
        if not surreal_bin:
            print(
                "Binary 'surreal' tidak ditemukan di PATH.\n"
                "Install SurrealDB CLI: https://surrealdb.com/install\n"
                "Atau set SURREAL_CLI=/path/to/surreal di .env",
                file=sys.stderr,
            )
            return 1

        host, port = _parse_bind_from_surreal_url(surreal_url)
        bind = args.bind_override or f"{host}:{port}"

        if args.memory:
            storage_path = "memory"
        else:
            storage_path = os.getenv("SURREAL_START_PATH", "").strip()
            if not storage_path:
                # SurrealDB 3: persist with SurrealKV; path relative to cwd (repo root).
                kv_dir = _REPO_ROOT / "data" / "surreal_local"
                kv_dir.mkdir(parents=True, exist_ok=True)
                storage_path = "surrealkv://data/surreal_local"

        cmd = [
            surreal_bin,
            "start",
            "--log",
            "info",
            "--user",
            user,
            "--pass",
            password,
            "--bind",
            bind,
            storage_path,
        ]
        print("Repo root:", _REPO_ROOT)
        print("Starting SurrealDB:", " ".join(cmd[:6]), "...", f"({storage_path})")
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            stdout=None,
            stderr=None,
        )

        atexit.register(lambda p=proc: _terminate_surreal(p))

        try:
            _wait_port(host, port)
        except SystemExit:
            _terminate_surreal(proc)
            raise
        print(f"OK: Surreal listening on {host}:{port} (matches SURREAL_URL host/port).")

    if args.serve_only:
        if proc is None:
            print("--serve-only requires Surreal to be started (omit --no-start).", file=sys.stderr)
            return 2
        ns = os.getenv("SURREAL_NS", "tempograph")
        db = os.getenv("SURREAL_DB", "main")
        host, port = _parse_bind_from_surreal_url(surreal_url)
        print()
        print("SurrealDB is running. Leave this window open.")
        print("  Surrealist: WS", f"{host}:{port}", "| Root auth:", user, "| then in query: USE NS", ns, "DB", db + ";")
        print("  Other terminals: python scripts/run_with_local_surreal.py --no-start -- <command>")
        print("Press Ctrl+C to stop the server.")
        print()
        try:
            return int(proc.wait())
        except KeyboardInterrupt:
            print("\nStopping SurrealDB...", flush=True)
            _terminate_surreal(proc)
            return 0

    assert child is not None
    print("Running:", " ".join(child))
    result = subprocess.run(child, cwd=str(_REPO_ROOT), env=os.environ.copy())
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
