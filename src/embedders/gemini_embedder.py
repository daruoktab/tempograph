# src/embedders/gemini_embedder.py
"""
Gemini Embedding Model Wrapper (Simple - no graphiti dependency)
Uses google.generativeai directly for Vanilla RAG.
"""

from typing import Any, List
import logging

from .base import BaseEmbedder, EmbedderType

logger = logging.getLogger(__name__)


class GeminiEmbedderWrapper(BaseEmbedder):
    """
    Simple wrapper untuk Google Gemini Embedding API.
    Menggunakan google.generativeai langsung (tanpa graphiti).
    """

    # Model options dengan dimensi
    MODELS = {"models/gemini-embedding-001": 768, "gemini-embedding-001": 768}

    # Pricing per 1M tokens (Paid Tier, Dec 2024)
    # Standard: $0.15, Batch: $0.075
    EMBEDDING_PRICE_PER_1M = 0.15

    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        super().__init__(model_name=model_name, model_type=EmbedderType.GEMINI)
        self.api_key = api_key
        self._genai: Any = None
        self._cost_tracker = None

        # Set dimension from known models
        if model_name in self.MODELS:
            self._dimension = self.MODELS[model_name]

    async def initialize(self):
        """Initialize Gemini embedder"""
        import google.generativeai as genai

        _configure = getattr(genai, "configure")
        _configure(api_key=self.api_key)
        self._genai = genai

        # Try to get cost tracker (optional)
        try:
            from ..utils.cost_tracker import CostTracker

            self._cost_tracker = CostTracker.get_instance()
        except Exception:
            pass  # Cost tracking not available

        # Verify dimension with test embedding if not known
        if self._dimension is None:
            _embed_content = getattr(genai, "embed_content")
            result = _embed_content(model=self.model_name, content="test")
            self._dimension = len(result["embedding"])

        logger.info(
            f"Gemini embedder initialized: {self.model_name} (dim={self._dimension})"
        )

    async def _embed_impl(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini API"""
        if self._genai is None:
            raise RuntimeError("Embedder not initialized")

        embeddings = []
        total_tokens = 0

        for text in texts:
            result = self._genai.embed_content(model=self.model_name, content=text)
            embeddings.append(result["embedding"])

            # Estimate token count (rough: 4 chars per token)
            total_tokens += len(text) // 4

        # Track cost if tracker available
        if self._cost_tracker and total_tokens > 0:
            await self._cost_tracker.track(
                input_tokens=total_tokens, output_tokens=0, model_name=self.model_name
            )

        return embeddings

    async def close(self):
        """Cleanup (Gemini doesn't need explicit cleanup)"""
        self._genai = None
