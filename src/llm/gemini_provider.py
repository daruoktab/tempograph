# src/llm_providers/gemini_provider.py
"""
Google Gemini LLM Provider (google-genai SDK).
"""

import asyncio
import logging
from typing import Any, List, Optional, cast

from google import genai
from google.genai import types

from .base import BaseLLMProvider, LLMProviderType, LLMResponse, Message

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Provider untuk Google Gemini API via ``google.genai``.

    Models:
    - gemini-2.5-flash: Fast, good for most tasks
    - gemini-2.5-pro: Best quality, slower
    - gemini-2.5-flash-lite: Fastest, lightweight
    """

    # Approximate USD per 1M tokens (for rough cost logging only)
    PRICING: dict[str, dict[str, float]] = {
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.5-flash-lite": {"input": 0.05, "output": 0.20},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        super().__init__(
            model=model,
            provider_type=LLMProviderType.GEMINI,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.api_key = api_key
        self._client: genai.Client | None = None

    async def initialize(self):
        """Initialize Gemini client"""
        self._client = genai.Client(api_key=self.api_key)
        logger.info("Gemini provider initialized: %s", self.model)

    async def _generate_impl(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate using Gemini API"""
        if self._client is None:
            raise RuntimeError("Provider not initialized")
        client = self._client

        contents: List[types.Content] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        t = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens

        config = types.GenerateContentConfig(
            temperature=t,
            max_output_tokens=mt,
            system_instruction=system_instruction,
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.model,
                contents=cast(Any, contents),
                config=config,
            ),
        )

        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = (
            getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        )

        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (
            prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
        ) / 1_000_000

        finish = None
        if response.candidates:
            fr = response.candidates[0].finish_reason
            finish = str(fr) if fr is not None else None

        return LLMResponse(
            content=response.text or "",
            model=self.model,
            provider=self.provider_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            finish_reason=finish,
        )

    async def close(self):
        """Cleanup"""
        self._client = None
