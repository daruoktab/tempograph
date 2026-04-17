# src/config/runtime_setup.py
"""
Build :class:`ExperimentSetup` from environment variables (single stack).

Env (see ``.env.example``):
  ``LLM_PROVIDER`` / ``LLM_MODEL`` — fact extraction & related GenAI paths
  ``EMBED_PROVIDER`` / ``EMBED_MODEL`` — dense vectors (session + facts)
  ``RAG_GROUP_ID`` — Surreal ``group_id`` for agentic graph
  ``RAG_SESSION_COLLECTION`` — ``session_passage.collection`` for vanilla / hybrid dense leg
  ``RAG_MODE`` — ``agentic`` | ``vanilla`` | ``hybrid`` (used with ``--setup env`` in eval)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .experiment_setups import (
    CHROMA_PERSIST_DIR,
    DIRECT_RETRIEVAL_SETTINGS,
    ExperimentSetup,
    GEMINI_LLM_SMALL,
    ModelConfig,
    ModelStack,
    RAGType,
    SetupType,
    StorageConfig,
    StorageType,
)


def _norm(s: Optional[str], default: str) -> str:
    return (s or default).strip().lower()


def get_rag_group_id() -> str:
    return os.getenv("RAG_GROUP_ID", "agentic_default").strip()


def get_session_passage_collection() -> str:
    return os.getenv("RAG_SESSION_COLLECTION", "vanilla_default").strip()


def llm_model_config_from_env() -> ModelConfig:
    p = _norm(os.getenv("LLM_PROVIDER"), "gemini")
    model = (os.getenv("LLM_MODEL") or "").strip()
    if p == "gemini":
        name = model or "gemini-2.5-flash"
        return ModelConfig(
            name=name,
            provider="gemini",
            description="Fact extraction (Gemini)",
            rpm=1000,
            tpm=1_000_000,
            rpd=None,
        )
    if p == "novita":
        name = model or "google/gemma-3-27b-it"
        return ModelConfig(
            name=name,
            provider="novita",
            description="Fact extraction (OpenAI-compatible, e.g. Novita)",
            rpm=100,
            tpm=500_000,
            rpd=None,
        )
    if p == "huggingface":
        raise ValueError(
            "LLM_PROVIDER=huggingface is not supported for graph fact extraction in this repo; "
            "use gemini or novita."
        )
    raise ValueError(f"Unsupported LLM_PROVIDER={p!r} (use gemini or novita)")


def embedder_model_config_from_env() -> ModelConfig:
    p = _norm(os.getenv("EMBED_PROVIDER"), "gemini")
    model = (os.getenv("EMBED_MODEL") or "").strip()
    if p == "gemini":
        name = model or "gemini-embedding-001"
        return ModelConfig(
            name=name,
            provider="gemini",
            description="Embeddings (Gemini)",
            rpm=1500,
            tpm=10_000_000,
            rpd=None,
        )
    if p == "huggingface":
        name = model or "google/embeddinggemma-300m"
        return ModelConfig(
            name=name,
            provider="huggingface",
            description="Embeddings (local HuggingFace)",
            rpm=None,
            tpm=None,
            rpd=None,
        )
    raise ValueError(f"Unsupported EMBED_PROVIDER={p!r} (use gemini or huggingface)")


def model_stack_from_llm_env() -> ModelStack:
    p = _norm(os.getenv("LLM_PROVIDER"), "gemini")
    return ModelStack.GEMINI if p == "gemini" else ModelStack.GEMMA


def get_agentic_experiment_setup_from_env() -> ExperimentSetup:
    llm = llm_model_config_from_env()
    emb = embedder_model_config_from_env()
    gid = get_rag_group_id()
    stack = model_stack_from_llm_env()
    return ExperimentSetup(
        name=f"Agentic (env: {llm.provider} {llm.name})",
        setup_type=SetupType.AGENTIC_GEMINI,
        rag_type=RAGType.AGENTIC,
        model_stack=stack,
        description="Agentic setup from LLM_* / EMBED_* / RAG_GROUP_ID",
        storage=StorageConfig(
            storage_type=StorageType.SURREAL,
            group_id=gid,
            persist_directory=CHROMA_PERSIST_DIR,
        ),
        embedder=emb,
        llm_extraction=llm,
        llm_small=GEMINI_LLM_SMALL if llm.provider == "gemini" else None,
        reranker=emb if emb.provider == "gemini" else None,
        reranker_type="embedding",
        retrieval=DIRECT_RETRIEVAL_SETTINGS,
    )


def get_vanilla_experiment_setup_from_env() -> ExperimentSetup:
    emb = embedder_model_config_from_env()
    coll = get_session_passage_collection()
    stack = model_stack_from_llm_env()
    return ExperimentSetup(
        name=f"Vanilla (env: {emb.provider} {emb.name})",
        setup_type=SetupType.VANILLA_GEMINI,
        rag_type=RAGType.VANILLA,
        model_stack=stack,
        description="Vanilla setup from EMBED_* / RAG_SESSION_COLLECTION",
        storage=StorageConfig(
            storage_type=StorageType.SURREAL,
            collection_name=coll,
            persist_directory=CHROMA_PERSIST_DIR,
        ),
        embedder=emb,
        llm_extraction=None,
        llm_small=None,
        reranker=emb if emb.provider == "gemini" else None,
        reranker_type="embedding",
        retrieval=DIRECT_RETRIEVAL_SETTINGS,
    )


@dataclass
class EvalEnvContext:
    """Resolved from ``--setup env`` + ``RAG_MODE`` for ``evaluate_agentic``."""

    primary_setup: ExperimentSetup
    vanilla_setup: Optional[ExperimentSetup]
    branch_key: str


def load_eval_env() -> EvalEnvContext:
    mode = _norm(os.getenv("RAG_MODE"), "agentic")
    ag = get_agentic_experiment_setup_from_env()
    vn = get_vanilla_experiment_setup_from_env()

    if mode == "vanilla":
        vk = "vanilla_gemini" if vn.embedder.provider == "gemini" else "vanilla_gemma"
        return EvalEnvContext(primary_setup=vn, vanilla_setup=None, branch_key=vk)

    if mode == "hybrid":
        hk = "gemini_hybrid" if ag.model_stack == ModelStack.GEMINI else "gemma_hybrid"
        return EvalEnvContext(primary_setup=ag, vanilla_setup=vn, branch_key=hk)

    if mode != "agentic":
        raise ValueError(f"Unsupported RAG_MODE={mode!r} (use agentic, vanilla, or hybrid)")

    bk = "gemini" if ag.model_stack == ModelStack.GEMINI else "gemma"
    return EvalEnvContext(primary_setup=ag, vanilla_setup=None, branch_key=bk)
