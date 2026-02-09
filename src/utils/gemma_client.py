# src/utils/gemma_client.py
"""
Gemma 3 client using Google GenAI API

Usage:
    from src.utils.gemma_client import GemmaClient

    client = GemmaClient()
    response = await client.generate("Hello, world!")
    print(response)
"""

import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GemmaResponse:
    """Response from Gemma model"""

    text: str
    model: str
    usage: Optional[Dict[str, int]] = None


class GemmaClient:
    """Client for Gemma 3 models via Google GenAI API"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemma-3-27b-it"):
        """
        Initialize Gemma client

        Args:
            api_key: Gemini API key (uses GEMINI_API_KEY env var if not provided)
            model: Model name (gemma-3-27b-it, gemma-3-12b-it, gemma-3-4b-it)
        """
        from google import genai

        if api_key is None:
            from ..config.settings import get_config

            config = get_config()
            api_key = config.gemini.api_key

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._genai = genai

    def generate_sync(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> GemmaResponse:
        """
        Generate content synchronously

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            **kwargs: Additional generation config

        Returns:
            GemmaResponse with generated text
        """
        from .rate_limiter import get_rate_limiter

        rate_limiter = get_rate_limiter()

        # Wait for rate limit (sync version)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, just proceed
            pass
        else:
            loop.run_until_complete(
                rate_limiter.wait_if_needed(
                    self.model, estimated_tokens=len(prompt) // 4
                )
            )

        # Build generation config
        generation_config = {
            "temperature": temperature,
        }
        if max_output_tokens:
            generation_config["max_output_tokens"] = max_output_tokens
        generation_config.update(kwargs)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=generation_config,  # type: ignore[invalid-argument-type]
            )

            # Extract usage if available
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ),
                    "completion_tokens": getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ),
                    "total_tokens": getattr(
                        response.usage_metadata, "total_token_count", 0
                    ),
                }
                rate_limiter.record_success(self.model, usage.get("total_tokens", 0))

            assert response.text is not None
            return GemmaResponse(text=response.text, model=self.model, usage=usage)

        except Exception as e:
            logger.error(f"Gemma generation failed: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> GemmaResponse:
        """
        Generate content asynchronously

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            **kwargs: Additional generation config

        Returns:
            GemmaResponse with generated text
        """
        from .rate_limiter import get_rate_limiter

        rate_limiter = get_rate_limiter()

        # Wait for rate limit
        await rate_limiter.wait_if_needed(self.model, estimated_tokens=len(prompt) // 4)

        # Build generation config
        generation_config = {
            "temperature": temperature,
        }
        if max_output_tokens:
            generation_config["max_output_tokens"] = max_output_tokens
        generation_config.update(kwargs)

        try:
            # Run in executor since google.genai might not have native async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=generation_config,  # type: ignore[invalid-argument-type]
                ),
            )

            # Extract usage if available
            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ),
                    "completion_tokens": getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ),
                    "total_tokens": getattr(
                        response.usage_metadata, "total_token_count", 0
                    ),
                }
                rate_limiter.record_success(self.model, usage.get("total_tokens", 0))

            return GemmaResponse(text=response.text, model=self.model, usage=usage)

        except Exception as e:
            logger.error(f"Gemma generation failed: {e}")
            raise


# Singleton instance
_gemma_client: Optional[GemmaClient] = None


def get_gemma_client(model: str = "gemma-3-27b-it") -> GemmaClient:
    """Get or create Gemma client singleton"""
    global _gemma_client
    if _gemma_client is None or _gemma_client.model != model:
        _gemma_client = GemmaClient(model=model)
    return _gemma_client


# Quick test
if __name__ == "__main__":
    import asyncio

    async def test():
        client = GemmaClient()
        response = await client.generate("Roses are red...")
        print(f"Model: {response.model}")
        print(f"Response: {response.text}")
        if response.usage:
            print(f"Usage: {response.usage}")

    asyncio.run(test())
