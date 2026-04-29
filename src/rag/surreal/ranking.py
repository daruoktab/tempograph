"""
Ranking helpers (RRF, MMR) — algorithms reimplemented for Tempograph.

RRF / MMR are standard IR techniques; implementation follows common definitions
(e.g. reciprocal rank fusion with configurable rank constant).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rank_const: int = 60,
    min_score: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Fuse ordered lists of document ids into a single ranking.

    ``rank_const`` follows common defaults (e.g. 60 in some vector DB docs);
    Graphiti's reference uses 1 — tune via ``RetrievalConfig.rrf_rank_const``.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        if not ranking:
            continue
        for i, doc_id in enumerate(ranking):
            if not doc_id:
                continue
            scores[str(doc_id)] += 1.0 / (rank_const + i)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if min_score > 0:
        ranked = [(d, s) for d, s in ranked if s >= min_score]
    return ranked


def _normalize_l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0:
        return v
    return v / n


def maximal_marginal_relevance(
    query_vector: Sequence[float],
    candidates: Mapping[str, Sequence[float]],
    *,
    k: int,
    lambda_mult: float = 0.5,
) -> List[str]:
    """
    Greedy MMR selection of up to ``k`` candidate ids.

    Each step picks argmax ( λ * sim(q, d) - (1-λ) * max_{s in selected} sim(d, s) ).
    """
    if not candidates or k <= 0:
        return []

    q = _normalize_l2(np.array(list(query_vector), dtype=np.float64))
    ids = [cid for cid in candidates if candidates[cid] is not None]
    if not ids:
        return []

    vecs = {cid: _normalize_l2(np.array(list(candidates[cid]), dtype=np.float64)) for cid in ids}
    sim_q = {cid: float(np.dot(q, vecs[cid])) for cid in ids}

    selected: List[str] = []
    remaining = set(ids)

    while remaining and len(selected) < k:
        best_id: Optional[str] = None
        best_score = -1e9
        for cid in remaining:
            max_sim_selected = 0.0
            for sid in selected:
                max_sim_selected = max(max_sim_selected, float(np.dot(vecs[cid], vecs[sid])))
            mmr = lambda_mult * sim_q[cid] - (1.0 - lambda_mult) * max_sim_selected
            if mmr > best_score:
                best_score = mmr
                best_id = cid
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)

    return selected


def estimate_chars_per_token() -> int:
    """Heuristic chars/token (same order as Graphiti content_chunking defaults)."""
    return 4


def should_chunk_by_density(
    text: str,
    *,
    min_tokens: int = 1000,
    density_threshold: float = 0.12,
) -> bool:
    """
    Return True if text is long and appears entity-dense (capitalized tokens heuristic).
    """
    est_tokens = max(len(text) // estimate_chars_per_token(), 1)
    if est_tokens < min_tokens:
        return False
    caps = sum(1 for w in text.split() if w[:1].isupper() and len(w) > 1)
    density = caps / float(est_tokens)
    return density >= density_threshold
