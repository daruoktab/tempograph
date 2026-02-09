# src/llm_providers/base.py
"""
Base LLM Provider Interface
Abstraksi untuk semua LLM providers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, AsyncIterator
import time


class LLMProviderType(Enum):
    """Supported LLM provider types"""

    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"  # Future: local Ollama support
    OPENAI_COMPATIBLE = "openai_compatible"  # Generic OpenAI-compatible APIs
    NOVITA = "novita"  # Novita AI (OpenAI-compatible)


@dataclass
class LLMResponse:
    """Standardized LLM response"""

    content: str
    model: str
    provider: LLMProviderType

    # Usage stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Performance
    latency_ms: float = 0.0

    # Cost tracking (in USD)
    cost_usd: float = 0.0

    # Metadata
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMMetrics:
    """Metrics for LLM provider performance tracking"""

    provider: LLMProviderType
    model: str

    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def avg_tokens_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (
            self.total_prompt_tokens + self.total_completion_tokens
        ) / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_tokens_per_request": self.avg_tokens_per_request,
            "errors": self.errors,
        }


@dataclass
class Message:
    """Chat message"""

    role: str  # "system", "user", "assistant"
    content: str


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Menyediakan interface yang konsisten untuk berbagai LLM.
    """

    def __init__(
        self,
        model: str,
        provider_type: LLMProviderType,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.provider_type = provider_type
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._metrics = LLMMetrics(provider=provider_type, model=model)

    @property
    def metrics(self) -> LLMMetrics:
        """Get current metrics"""
        return self._metrics

    @abstractmethod
    async def initialize(self):
        """Initialize the provider (load model, verify API key, etc.)"""
        pass

    @abstractmethod
    async def _generate_impl(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Internal generation implementation"""
        pass

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        Tracks metrics automatically.

        Args:
            messages: List of chat messages
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            LLMResponse with content and metadata
        """
        start_time = time.perf_counter()

        try:
            response = await self._generate_impl(
                messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs,
            )

            # Update latency
            response.latency_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._metrics.total_requests += 1
            self._metrics.total_prompt_tokens += response.prompt_tokens
            self._metrics.total_completion_tokens += response.completion_tokens
            self._metrics.total_latency_ms += response.latency_ms
            self._metrics.total_cost_usd += response.cost_usd

            return response

        except Exception as e:
            self._metrics.errors += 1
            raise

    async def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Convenience method for simple chat.

        Args:
            user_message: User's message
            system_prompt: Optional system prompt

        Returns:
            Assistant's response text
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))

        response = await self.generate(messages)
        return response.content

    @abstractmethod
    async def close(self):
        """Cleanup resources"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
