# src/llm_providers/gemini_provider.py
"""
Google Gemini LLM Provider
"""

import asyncio
from typing import List, Optional, Dict, Any
import logging

from .base import BaseLLMProvider, LLMProviderType, LLMResponse, Message

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Provider untuk Google Gemini API.
    Kompatibel dengan graphiti_core.
    
    Models:
    - gemini-2.5-flash: Fast, good for most tasks
    - gemini-2.5-pro: Best quality, slower
    - gemini-2.5-flash-lite: Fastest, lightweight
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        super().__init__(
            model=model,
            provider_type=LLMProviderType.GEMINI,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.api_key = api_key
        self._client = None
    
    async def initialize(self):
        """Initialize Gemini client"""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model)
        
        logger.info(f"Gemini provider initialized: {self.model}")
    
    async def _generate_impl(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate using Gemini API"""
        if self._client is None:
            raise RuntimeError("Provider not initialized")
        
        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                gemini_messages.append({
                    "role": role,
                    "parts": [msg.content]
                })
        
        # Create generation config
        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Generate
        loop = asyncio.get_event_loop()
        
        if system_instruction:
            # Recreate model with system instruction
            import google.generativeai as genai
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction
            )
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    gemini_messages,
                    generation_config=gen_config
                )
            )
        else:
            response = await loop.run_in_executor(
                None,
                lambda: self._client.generate_content(
                    gemini_messages,
                    generation_config=gen_config
                )
            )
        
        # Extract usage
        prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        
        # Calculate cost
        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
        
        return LLMResponse(
            content=response.text,
            model=self.model,
            provider=self.provider_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None
        )
    
    async def close(self):
        """Cleanup"""
        self._client = None
    
    def get_graphiti_client(self):
        """
        Return a Graphiti-compatible LLM client.
        For use with Graphiti's knowledge graph operations.
        """
        from graphiti_core.llm_client.gemini_client import GeminiClient
        from graphiti_core.llm_client.config import LLMConfig
        
        return GeminiClient(
            config=LLMConfig(
                api_key=self.api_key,
                model=self.model
            )
        )
