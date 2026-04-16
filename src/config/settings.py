# src/config.py
"""
Configuration Management untuk Temporal RAG System
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# =============================================================================
# RATE LIMIT CONFIGURATION
# =============================================================================


@dataclass
class ModelRateLimit:
    """Rate limit configuration for a specific model"""

    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute
    rpd: Optional[int] = None  # Requests per day (None = unlimited)

    @property
    def min_delay_seconds(self) -> float:
        """Minimum delay between requests to stay within RPM limit"""
        return 60.0 / self.rpm


# Rate limits based on user's Gemini API quotas
# Format dari user: current_usage / max_limit
# Kita gunakan MAX LIMIT untuk rate limiting
#
# User's quota:
# | Model                  | RPM      | TPM        | RPD          |
# |------------------------|----------|------------|--------------|
# | gemini-2.5-pro         | (TBD)    | (TBD)      | (TBD)        |
# | gemini-2.5-flash       | 1K       | 1M         | 10K          |
# | gemini-2.5-flash-lite  | 4K       | 4M         | Unlimited    |

GEMINI_RATE_LIMITS: Dict[str, ModelRateLimit] = {
    # gemini-2.5-pro: Max 15 RPM, 1M TPM, 300 RPD
    # (User current: 11 RPM, 118.14K TPM, 256 RPD)
    "gemini-2.5-pro": ModelRateLimit(
        rpm=15,  # 15 requests per minute
        tpm=1_000_000,  # 1M tokens per minute
        rpd=300,  # 300 requests per day (tight limit!)
    ),
    # gemini-2.5-flash: Max 1K RPM, 1M TPM, 10K RPD
    "gemini-2.5-flash": ModelRateLimit(
        rpm=1000,  # 1K requests per minute
        tpm=1_000_000,  # 1M tokens per minute
        rpd=10_000,  # 10K requests per day
    ),
    # gemini-2.5-flash-lite: Max 4K RPM, 4M TPM, Unlimited RPD
    "gemini-2.5-flash-lite": ModelRateLimit(
        rpm=4000,  # 4K requests per minute
        tpm=4_000_000,  # 4M tokens per minute
        rpd=None,  # Unlimited requests per day
    ),
    # gemini-embedding-001: Estimated (no official data from user)
    # Conservative estimate based on typical embedding API limits
    "models/gemini-embedding-001": ModelRateLimit(
        rpm=1500,  # Embedding APIs typically have high RPM
        tpm=10_000_000,  # Very high TPM for embeddings
        rpd=None,  # Usually unlimited
    ),
    # Alias for embedding model
    "gemini-embedding-001": ModelRateLimit(rpm=1500, tpm=10_000_000, rpd=None),
}


def get_rate_limit(model_name: str) -> ModelRateLimit:
    """Get rate limit for a model, with fallback to flash limits"""
    if model_name in GEMINI_RATE_LIMITS:
        return GEMINI_RATE_LIMITS[model_name]

    # Fallback: use flash limits as default
    return GEMINI_RATE_LIMITS["gemini-2.5-flash"]


@dataclass
class RateLimitConfig:
    """Global rate limiting configuration"""

    enabled: bool = True
    max_retry_delay: float = 60.0  # Maximum delay between retries
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    initial_delay: float = 1.0  # Initial delay between requests


@dataclass
class SurrealDBConfig:
    """SurrealDB connection (graph + vectors)."""

    url: str = field(
        default_factory=lambda: os.getenv("SURREAL_URL", "ws://127.0.0.1:8000")
    )
    username: str = field(default_factory=lambda: os.getenv("SURREAL_USER", "root"))
    password: str = field(default_factory=lambda: os.getenv("SURREAL_PASS", "root"))
    namespace: str = field(default_factory=lambda: os.getenv("SURREAL_NS", "skripsi"))
    database: str = field(default_factory=lambda: os.getenv("SURREAL_DB", "pending"))


@dataclass
class GeminiConfig:
    """Google Gemini API configuration"""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Model assignments based on task complexity
    model_hard: str = "gemini-2.5-flash"  # Complex tasks: conversation, reasoning
    model_medium: str = "gemini-2.5-flash"  # Medium tasks: extraction, classification
    model_easy: str = "gemini-2.5-flash-lite"  # Simple tasks: reranking, parsing

    # Legacy field name: Gemma **chat** in this repo uses Novita (see NovitaConfig), not Gemini API.
    gemma_model: str = "gemma-3-27b-it"  # Gemma 3 27B Instruct (logical id)

    # Embedding model
    embedding_model: str = "models/gemini-embedding-001"

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")


@dataclass
class EmbedderExperimentConfig:
    """Configuration for embedding model experiments"""

    # Which embedders to evaluate (from factory.EXPERIMENT_CONFIGS)
    embedders_to_test: list = field(
        default_factory=lambda: [
            "gemini-001",  # Google Gemini Embedding-001 (embed + rerank)
            "embeddinggemma-300m",  # Google EmbeddingGemma 300M (local)
        ]
    )

    # Device for HuggingFace models
    hf_device: Optional[str] = None  # None = auto-detect (cuda if available)

    # Whether to use half precision
    use_fp16: bool = True

    # Batch size for embedding
    batch_size: int = 32

    # Save intermediate results
    save_embeddings: bool = False
    embeddings_dir: str = "output/embeddings_cache"


@dataclass
class OpenRouterConfig:
    """OpenRouter API configuration"""

    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    site_url: Optional[str] = None
    app_name: str = "TemporalRAG"

    def is_configured(self) -> bool:
        """Check if OpenRouter is configured"""
        return bool(self.api_key)


@dataclass
class NovitaConfig:
    """Novita AI API configuration (OpenAI-compatible)

    Used for running Gemma 3 27B IT via Novita AI API.
    Docs: https://novita.ai/docs
    """

    api_key: str = field(default_factory=lambda: os.getenv("NOVITAAI_API_KEY", ""))
    base_url: str = "https://api.novita.ai/openai"

    # Default model for Gemma
    gemma_model: str = "google/gemma-3-27b-it"

    # Rate limits (Novita AI typically has generous limits)
    rpm: int = 100  # Requests per minute (conservative estimate)
    tpm: int = 500_000  # Tokens per minute

    def is_configured(self) -> bool:
        """Check if Novita AI is configured"""
        return bool(self.api_key)


@dataclass
class LLMExperimentConfig:
    """Configuration for LLM model experiments"""

    # Which LLM providers to evaluate (from llm_providers.factory)
    llm_providers_to_test: list = field(
        default_factory=lambda: [
            "gemini-flash",  # Gemini 2.5 Flash (main)
            "gemini-flash-lite",  # Gemini 2.5 Flash Lite (reranker)
        ]
    )

    # Device for local models
    hf_device: Optional[str] = None

    # Use quantization for local models
    use_4bit: bool = True
    use_8bit: bool = False


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline"""

    batch_size: int = 10  # Number of turns to process in batch
    max_retries: int = 3  # Retries for failed API calls
    retry_delay: float = 2.0  # Delay between retries (seconds)
    enable_fact_extraction: bool = True  # Whether to extract facts from turns
    enable_entity_linking: bool = True  # Whether to link entities


@dataclass
class RetrievalConfig:
    """Configuration for retrieval agent - FAIR settings for all setups"""

    # Top-K settings (SAME for all setups for fair comparison)
    embedding_top_k: int = 30  # Candidates from vector search
    rerank_top_k: int = 10  # Final results after rerank
    num_results: int = 10  # Alias for rerank_top_k (backward compat)

    # Thresholds
    similarity_threshold: float = 0.5  # Minimum similarity score

    # Agentic-only settings (ignored in Vanilla)
    max_iterations: int = 3  # Max iterations for agentic loop
    enable_reranking: bool = True  # Whether to rerank results
    enable_temporal_filter: bool = True  # Whether to apply temporal filtering

    def __post_init__(self):
        # Ensure num_results matches rerank_top_k
        self.num_results = self.rerank_top_k


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    use_llm_judge: bool = True  # Use LLM for context sufficiency evaluation
    judge_model: str = "gemini-2.5-pro"  # Model for LLM judge (largest 2.5 model)
    save_detailed_results: bool = True  # Save per-turn results

    # Metrics to calculate (normalized for fair comparison)
    metrics: list = field(
        default_factory=lambda: [
            "context_recall",  # Required facts found / total required
            "context_precision",  # Relevant facts / total retrieved
            "hit_rate",  # Binary: answer found in context?
            "mrr",  # Mean Reciprocal Rank
            "temporal_recall",  # Temporal facts found
            "context_sufficiency",  # LLM Judge: is context sufficient?
        ]
    )


@dataclass
class Config:
    """Main configuration class combining all configs"""

    surreal: SurrealDBConfig = field(default_factory=SurrealDBConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    novita: NovitaConfig = field(default_factory=NovitaConfig)
    embedder_experiment: EmbedderExperimentConfig = field(
        default_factory=EmbedderExperimentConfig
    )
    llm_experiment: LLMExperimentConfig = field(default_factory=LLMExperimentConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Paths
    dataset_path: str = "output/final_dataset_v1/conversation_dataset.json"
    output_dir: str = "output/evaluation_results"

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls()


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config():
    """Reset global config (useful for testing)"""
    global _config
    _config = None


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("Configuration loaded successfully!")
        print(f"  SurrealDB URL: {config.surreal.url}")
        print(f"  SurrealDB NS/DB: {config.surreal.namespace}/{config.surreal.database}")
        print(f"  Gemini Model (Hard): {config.gemini.model_hard}")
        print(f"  Embedding Model: {config.gemini.embedding_model}")
        print(f"  Dataset Path: {config.dataset_path}")
    except ValueError as e:
        print(f"Configuration error: {e}")
