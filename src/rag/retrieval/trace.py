"""Structured retrieval latency / phase logging → JSONL (RETRIEVAL_TRACE_JSONL)."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def _trace_path(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.getenv("RETRIEVAL_TRACE_JSONL") or None


def append_retrieval_trace(record: Dict[str, Any], path: Optional[str] = None) -> None:
    p = _trace_path(path)
    if not p:
        return
    line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
    try:
        parent = os.path.dirname(os.path.abspath(p))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        logger.debug("retrieval trace write failed: %s", e)


@contextmanager
def retrieval_span(
    phase: str,
    *,
    path: Optional[str] = None,
    **attrs: Any,
) -> Iterator[None]:
    """Record wall time for a named phase when RETRIEVAL_TRACE_JSONL is set."""
    t0 = time.perf_counter()
    base: Dict[str, Any] = {"phase": phase, **attrs}
    try:
        yield
    finally:
        base["ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        append_retrieval_trace(base, path=path)
