# src/utils/gemma_client.py
"""
Gemma (instruction) models via Novita AI — OpenAI-compatible API.

Project default: Gemma LLM calls go through Novita (NOVITAAI_API_KEY), not Google GenAI.

Usage:
    from src.utils.gemma_client import GemmaClient

    client = GemmaClient()
    response = await client.generate("Hello, world!")
    print(response.text)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _novita_model_id(model: str) -> str:
    """Novita expects ids like ``google/gemma-3-27b-it``."""
    if "/" in model:
        return model
    if model.startswith("gemma-"):
        return f"google/{model}"
    return model


@dataclass
class GemmaResponse:
    """Response from a Gemma chat completion (Novita)."""

    text: str
    model: str
    usage: Optional[Dict[str, int]] = None


class GemmaClient:
    """Client for Gemma models via Novita AI (OpenAI-compatible)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemma-3-27b-it"):
        from ..config.settings import get_config

        cfg = get_config().novita
        self.api_key = api_key if api_key is not None else cfg.api_key
        if not self.api_key:
            raise ValueError(
                "NOVITAAI_API_KEY is required for GemmaClient (Novita AI)."
            )
        self.base_url = cfg.base_url
        self.model = _novita_model_id(model)

    def generate_sync(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> GemmaResponse:
        from openai import OpenAI

        from .rate_limiter import get_rate_limiter

        rate_limiter = get_rate_limiter()

        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.run_until_complete(
                rate_limiter.wait_if_needed(
                    self.model, estimated_tokens=len(prompt) // 4
                )
            )

        max_tokens = max_output_tokens or kwargs.pop("max_tokens", None) or 4096
        kwargs.pop("max_output_tokens", None)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            logger.error("Gemma (Novita) sync generation failed: %s", e)
            raise

        text = (resp.choices[0].message.content or "").strip()
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
                "total_tokens": resp.usage.total_tokens or 0,
            }
            rate_limiter.record_success(self.model, usage.get("total_tokens", 0))

        return GemmaResponse(text=text, model=self.model, usage=usage)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> GemmaResponse:
        from openai import AsyncOpenAI

        from .rate_limiter import get_rate_limiter

        rate_limiter = get_rate_limiter()
        await rate_limiter.wait_if_needed(self.model, estimated_tokens=len(prompt) // 4)

        max_tokens = max_output_tokens or kwargs.pop("max_tokens", None) or 4096
        kwargs.pop("max_output_tokens", None)

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            logger.error("Gemma (Novita) async generation failed: %s", e)
            raise

        text = (resp.choices[0].message.content or "").strip()
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens or 0,
                "completion_tokens": resp.usage.completion_tokens or 0,
                "total_tokens": resp.usage.total_tokens or 0,
            }
            rate_limiter.record_success(self.model, usage.get("total_tokens", 0))

        return GemmaResponse(text=text, model=self.model, usage=usage)


_gemma_client: Optional[GemmaClient] = None


def get_gemma_client(model: str = "gemma-3-27b-it") -> GemmaClient:
    """Get or create Gemma client singleton (Novita)."""
    global _gemma_client
    norm = _novita_model_id(model)
    if _gemma_client is None or _gemma_client.model != norm:
        _gemma_client = GemmaClient(model=model)
    return _gemma_client


if __name__ == "__main__":

    async def _test() -> None:
        client = GemmaClient()
        response = await client.generate("Roses are red...")
        print(f"Model: {response.model}")
        print(f"Response: {response.text}")
        if response.usage:
            print(f"Usage: {response.usage}")

    asyncio.run(_test())
