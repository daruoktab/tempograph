# src/llm_providers/novita_provider.py
"""
Novita AI LLM Provider (OpenAI-compatible)

Used for running Gemma 3 27B IT via Novita AI's OpenAI-compatible API.
Supports structured output via JSON schema.
"""

import asyncio
from typing import List, Optional, Dict, Any
import logging

from .base import BaseLLMProvider, LLMProviderType, LLMResponse, Message

logger = logging.getLogger(__name__)


class NovitaProvider(BaseLLMProvider):
    """
    Provider untuk Novita AI (OpenAI-compatible API).
    Used for Gemma 3 27B IT.
    
    Models via Novita AI:
    - google/gemma-3-27b-it: Gemma 3 27B Instruct
    """
    
    # Pricing for Novita AI models (per 1M tokens)
    # https://novita.ai/pricing
    NOVITA_PRICING = {
        "google/gemma-3-27b-it": {"input": 0.0952, "output": 0.16},
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemma-3-27b-it",
        base_url: str = "https://api.novita.ai/openai",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        super().__init__(
            model=model,
            provider_type=LLMProviderType.OPENAI_COMPATIBLE,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
    
    async def initialize(self):
        """Initialize OpenAI-compatible client for Novita AI"""
        from openai import OpenAI
        
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"Novita AI provider initialized: {self.model}")
    
    async def _generate_impl(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Novita AI (OpenAI-compatible) API"""
        if self._client is None:
            raise RuntimeError("Provider not initialized")
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Build request params
        request_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        # Add response format for structured output if provided
        if response_format:
            request_params["response_format"] = response_format
        
        # Generate
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(**request_params)
        )
        
        # Extract usage
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Calculate cost
        pricing = self.NOVITA_PRICING.get(self.model, {"input": 0.20, "output": 0.20})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider=self.provider_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            finish_reason=response.choices[0].finish_reason if response.choices else None
        )
    
    async def close(self):
        """Cleanup"""
        self._client = None
    
    def get_graphiti_client(self):
        """
        Return a Graphiti-compatible LLM client for Novita AI.
        Uses OpenAIGenericClient since Novita AI is OpenAI-compatible.
        """
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.llm_client.config import LLMConfig
        
        return OpenAIGenericClient(
            config=LLMConfig(
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url
            )
        )
