# src/embedders/base.py
"""
Base Embedder Interface
Abstraksi untuk semua embedding models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import time


class EmbedderType(Enum):
    """Supported embedder types"""
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"  # Future support
    COHERE = "cohere"  # Future support


@dataclass
class EmbeddingResult:
    """Result from embedding operation"""
    embeddings: List[List[float]]
    model_name: str
    dimension: int
    latency_ms: float
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class EmbedderMetrics:
    """Metrics for embedder performance tracking"""
    model_name: str
    model_type: EmbedderType
    dimension: int
    total_requests: int = 0
    total_texts: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests
    
    @property
    def avg_texts_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_texts / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "dimension": self.dimension,
            "total_requests": self.total_requests,
            "total_texts": self.total_texts,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_texts_per_request": self.avg_texts_per_request,
            "errors": self.errors
        }


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.
    Menyediakan interface yang konsisten untuk berbagai embedding models.
    """
    
    def __init__(self, model_name: str, model_type: EmbedderType):
        self.model_name = model_name
        self.model_type = model_type
        self._dimension: Optional[int] = None
        self._metrics = EmbedderMetrics(
            model_name=model_name,
            model_type=model_type,
            dimension=0
        )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        return self._dimension
    
    @property
    def metrics(self) -> EmbedderMetrics:
        """Get current metrics"""
        self._metrics.dimension = self._dimension or 0
        return self._metrics
    
    @abstractmethod
    async def initialize(self):
        """Initialize the embedder (load model, etc.)"""
        pass
    
    @abstractmethod
    async def _embed_impl(self, texts: List[str]) -> List[List[float]]:
        """Internal embedding implementation"""
        pass
    
    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed a list of texts.
        Tracks metrics automatically.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model_name=self.model_name,
                dimension=self.dimension,
                latency_ms=0.0
            )
        
        start_time = time.perf_counter()
        
        try:
            embeddings = await self._embed_impl(texts)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self._metrics.total_requests += 1
            self._metrics.total_texts += len(texts)
            self._metrics.total_latency_ms += latency_ms
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                dimension=self.dimension,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            self._metrics.errors += 1
            raise
    
    async def embed_single(self, text: str) -> List[float]:
        """Convenience method for single text embedding"""
        result = await self.embed([text])
        return result.embeddings[0] if result.embeddings else []
    
    @abstractmethod
    async def close(self):
        """Cleanup resources"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self._dimension})"
