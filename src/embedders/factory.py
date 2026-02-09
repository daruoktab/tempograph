# src/embedders/factory.py
"""
Embedder Factory
Centralized creation of embedders for easy experimentation
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from .base import BaseEmbedder, EmbedderType
from .gemini_embedder import GeminiEmbedderWrapper
from .hf_embedder import HuggingFaceEmbedder

# Re-export for convenience
__all__ = [
    "create_embedder",
    "create_embedder_by_name",
    "get_available_embedders",
    "EXPERIMENT_CONFIGS",
    "benchmark_embedder",
]


@dataclass
class EmbedderConfig:
    """Configuration for creating an embedder"""

    embedder_type: EmbedderType
    model_name: str
    description: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)


# Pre-defined configurations for experiments
EXPERIMENT_CONFIGS: Dict[str, EmbedderConfig] = {
    # === PRIMARY: Gemini Embedding (used as embedding + reranker) ===
    "gemini-001": EmbedderConfig(
        embedder_type=EmbedderType.GEMINI,
        model_name="models/gemini-embedding-001",
        description="Google Gemini Embedding-001 (768d, embedding + reranker)",
    ),
    # === PRIMARY: EmbeddingGemma (local, paired with Gemini Flash Lite as reranker) ===
    "embeddinggemma-300m": EmbedderConfig(
        embedder_type=EmbedderType.HUGGINGFACE,
        model_name="google/embeddinggemma-300m",
        description="Google EmbeddingGemma 300M (768d, local embedding)",
    ),
}


def create_embedder(
    config: Union[EmbedderConfig, None] = None,
    embedder_type: Optional[EmbedderType] = None,
    model_name: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    device: Optional[str] = None,
) -> BaseEmbedder:
    """
    Create an embedder from configuration or type+model_name.

    Args:
        config: Embedder configuration (optional if type and model provided)
        embedder_type: Embedder type (used if config not provided)
        model_name: Model name (used if config not provided)
        gemini_api_key: API key for Gemini (required if using Gemini)
        device: Device for HuggingFace models

    Returns:
        Uninitialized embedder instance
    """
    # If no config, create from type and model_name
    if config is None:
        if embedder_type is None:
            raise ValueError("Either config or embedder_type must be provided")

        config = EmbedderConfig(
            embedder_type=embedder_type,
            model_name=model_name or "default",
            description="Dynamic config",
        )

    if config.embedder_type == EmbedderType.GEMINI:
        if not gemini_api_key:
            raise ValueError("gemini_api_key required for Gemini embedder")
        return GeminiEmbedderWrapper(
            api_key=gemini_api_key, model_name=config.model_name
        )

    elif config.embedder_type == EmbedderType.HUGGINGFACE:
        # IMPORTANT: use_fp16=False to avoid NaN on long text (>700 chars)
        # FP16 causes precision overflow with embeddinggemma on long sessions
        return HuggingFaceEmbedder(
            model_name=config.model_name,
            device=device,
            use_fp16=False,  # Disable FP16 to prevent NaN on long text
            **config.extra_params,
        )

    else:
        raise ValueError(f"Unsupported embedder type: {config.embedder_type}")


def create_embedder_by_name(
    name: str, gemini_api_key: Optional[str] = None, device: Optional[str] = None
) -> BaseEmbedder:
    """
    Create embedder by preset name.

    Args:
        name: Preset name from EXPERIMENT_CONFIGS
        gemini_api_key: API key for Gemini
        device: Device for HuggingFace models

    Returns:
        Uninitialized embedder instance
    """
    if name not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown embedder preset: {name}. "
            f"Available: {list(EXPERIMENT_CONFIGS.keys())}"
        )

    return create_embedder(
        EXPERIMENT_CONFIGS[name], gemini_api_key=gemini_api_key, device=device
    )


def get_available_embedders() -> Dict[str, str]:
    """Get list of available embedder presets with descriptions"""
    return {name: config.description for name, config in EXPERIMENT_CONFIGS.items()}


async def benchmark_embedder(
    embedder: BaseEmbedder, test_texts: List[str], num_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark an embedder's performance.

    Args:
        embedder: Initialized embedder
        test_texts: Texts to use for benchmarking
        num_runs: Number of runs to average

    Returns:
        Benchmark results
    """
    import time

    latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()
        _ = await embedder.embed(test_texts)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "model_name": embedder.model_name,
        "model_type": embedder.model_type.value,
        "dimension": embedder.dimension,
        "num_texts": len(test_texts),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "throughput_texts_per_sec": len(test_texts)
        / (sum(latencies) / len(latencies) / 1000),
    }
