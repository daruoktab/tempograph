# src/config/dataset_generation_env.py
"""
Gemini model IDs for ``src/dataset/generator.py`` (three tiers).

Loaded from environment; defaults match the previous hard-coded stack.
``python-dotenv`` is loaded elsewhere (e.g. ``gemini_utils``) before typical use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _pick(name: str, default: str) -> str:
    v = (os.getenv(name) or "").strip()
    return v if v else default


@dataclass(frozen=True)
class DatasetGeminiModels:
    """Maps to roles in ``generator.py``."""

    # Persona JSON, life events (structured JSON / schema)
    structured: str
    # Multi-turn sessions + context-cache target model
    dialog: str
    # Session summaries, per-turn ground truth, conflict merge
    light: str


@lru_cache(maxsize=1)
def get_dataset_gemini_models() -> DatasetGeminiModels:
    return DatasetGeminiModels(
        structured=_pick("DATASET_GEMINI_MODEL_STRUCTURED", "gemini-2.5-flash"),
        dialog=_pick("DATASET_GEMINI_MODEL_DIALOG", "gemini-2.5-pro"),
        light=_pick("DATASET_GEMINI_MODEL_LIGHT", "gemini-2.5-flash-lite"),
    )


def clear_dataset_gemini_models_cache() -> None:
    """For tests or REPL: allow re-reading env."""
    get_dataset_gemini_models.cache_clear()
