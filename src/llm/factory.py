# src/llm_providers/factory.py
"""
LLM Provider Factory
Centralized creation of LLM providers for experiments
"""

from typing import Dict, Optional, Any, Union
from dataclasses import dataclass, field
import os

from .base import BaseLLMProvider, LLMProviderType
from .gemini_provider import GeminiProvider
from .novita_provider import NovitaProvider


@dataclass
class LLMProviderConfig:
    """Configuration for creating an LLM provider"""

    provider_type: LLMProviderType
    model: str
    description: str = ""
    cost_tier: str = "medium"  # "free", "low", "medium", "high"
    quality_tier: str = "medium"  # "low", "medium", "high"
    extra_params: Dict[str, Any] = field(default_factory=dict)


# Pre-defined configurations for experiments
LLM_PROVIDER_CONFIGS: Dict[str, LLMProviderConfig] = {
    # === PRIMARY: Gemini Flash (main LLM for extraction) ===
    "gemini-flash": LLMProviderConfig(
        provider_type=LLMProviderType.GEMINI,
        model="gemini-2.5-flash",
        description="Google Gemini 2.5 Flash - Main LLM",
        cost_tier="low",
        quality_tier="high",
    ),
    # === GEMMA via Novita AI ===
    "gemma-novita": LLMProviderConfig(
        provider_type=LLMProviderType.NOVITA,
        model="google/gemma-3-27b-it",
        description="Gemma 3 27B IT via Novita AI",
        cost_tier="low",
        quality_tier="high",
    ),
}


def create_llm_provider(
    config: Optional[LLMProviderConfig] = None,
    provider_type: Optional[LLMProviderType] = None,
    model_name: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    Create an LLM provider from configuration or type+model.

    Args:
        config: Provider configuration (optional if type+model provided)
        provider_type: Provider type (used if config not provided)
        model_name: Model name (used if config not provided)
        gemini_api_key: API key for Gemini
        openrouter_api_key: API key for OpenRouter
        device: Device for local models
        **kwargs: Additional parameters

    Returns:
        Uninitialized provider instance
    """
    # If no config, create from type and model_name
    if config is None:
        if provider_type is None:
            raise ValueError("Either config or provider_type must be provided")

        config = LLMProviderConfig(
            provider_type=provider_type,
            model=model_name or "default",
            description="Dynamic config",
        )

    merged_params = {**config.extra_params, **kwargs}

    if config.provider_type == LLMProviderType.GEMINI:
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY required for Gemini provider")
        return GeminiProvider(api_key=api_key, model=config.model, **merged_params)

    elif config.provider_type == LLMProviderType.NOVITA:
        api_key = kwargs.get("novita_api_key") or os.getenv("NOVITAAI_API_KEY")
        if not api_key:
            raise ValueError("NOVITAAI_API_KEY required for Novita provider")
        return NovitaProvider(
            api_key=api_key,
            model=config.model,
            base_url="https://api.novita.ai/openai",
            **merged_params,
        )

    else:
        raise ValueError(f"Unsupported provider type: {config.provider_type}")


def create_llm_provider_by_name(
    name: str,
    gemini_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    Create provider by preset name.

    Args:
        name: Preset name from LLM_PROVIDER_CONFIGS
        gemini_api_key: API key for Gemini
        openrouter_api_key: API key for OpenRouter
        device: Device for local models

    Returns:
        Uninitialized provider instance
    """
    if name not in LLM_PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown LLM provider preset: {name}. "
            f"Available: {list(LLM_PROVIDER_CONFIGS.keys())}"
        )

    return create_llm_provider(
        LLM_PROVIDER_CONFIGS[name],
        gemini_api_key=gemini_api_key,
        openrouter_api_key=openrouter_api_key,
        device=device,
        **kwargs,
    )


def get_available_providers() -> Dict[str, Dict[str, str]]:
    """Get list of available provider presets with info"""
    return {
        name: {
            "description": config.description,
            "type": config.provider_type.value,
            "model": config.model,
            "cost": config.cost_tier,
            "quality": config.quality_tier,
        }
        for name, config in LLM_PROVIDER_CONFIGS.items()
    }


def get_providers_by_type(
    provider_type: LLMProviderType,
) -> Dict[str, LLMProviderConfig]:
    """Get providers filtered by type"""
    return {
        name: config
        for name, config in LLM_PROVIDER_CONFIGS.items()
        if config.provider_type == provider_type
    }


def get_free_providers() -> Dict[str, LLMProviderConfig]:
    """Get providers with no API cost (local models)"""
    return {
        name: config
        for name, config in LLM_PROVIDER_CONFIGS.items()
        if config.cost_tier == "free"
    }
