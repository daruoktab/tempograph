# src/embedders/gemini_embedder.py
"""
Gemini Embedding Model Wrapper (google-genai SDK).
"""

from typing import List
import logging

from google import genai

from .base import BaseEmbedder, EmbedderType

logger = logging.getLogger(__name__)


class GeminiEmbedderWrapper(BaseEmbedder):
    """
    Wrapper untuk Google Gemini Embedding API (``google.genai``).
    """

    MODELS = {"models/gemini-embedding-001": 768, "gemini-embedding-001": 768}

    EMBEDDING_PRICE_PER_1M = 0.15

    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        super().__init__(model_name=model_name, model_type=EmbedderType.GEMINI)
        self.api_key = api_key
        self._client: genai.Client | None = None
        self._cost_tracker = None

        if model_name in self.MODELS:
            self._dimension = self.MODELS[model_name]

    async def initialize(self):
        """Initialize Gemini embedder"""
        self._client = genai.Client(api_key=self.api_key)

        try:
            from ..utils.cost_tracker import CostTracker

            self._cost_tracker = CostTracker.get_instance()
        except Exception:
            pass

        if self._dimension is None:
            resp = self._client.models.embed_content(
                model=self.model_name,
                contents="test",
            )
            em = (resp.embeddings or [None])[0]
            self._dimension = len(em.values) if em and em.values else 0

        logger.info(
            "Gemini embedder initialized: %s (dim=%s)",
            self.model_name,
            self._dimension,
        )

    async def _embed_impl(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini API"""
        if self._client is None:
            raise RuntimeError("Embedder not initialized")

        embeddings: List[List[float]] = []
        total_tokens = 0

        for text in texts:
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            emb = (result.embeddings or [None])[0]
            if not emb or not emb.values:
                raise RuntimeError("Empty embedding from Gemini API")
            embeddings.append(list(emb.values))
            total_tokens += len(text) // 4

        if self._cost_tracker and total_tokens > 0:
            await self._cost_tracker.track(
                input_tokens=total_tokens, output_tokens=0, model_name=self.model_name
            )

        return embeddings

    async def close(self):
        """Cleanup"""
        self._client = None
