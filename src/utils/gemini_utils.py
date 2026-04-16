"""
Global utility methods for API calls to Google Gemini (google-genai SDK).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    """Shared Gemini API client (Developer API, API key)."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def set_gemini_key():
    """Reset client so the next call picks up the current GEMINI_API_KEY from the environment."""
    global _client
    _client = None
    get_gemini_client()


# Global variable for log path
TOKEN_LOG_PATH = os.path.join("output", "token_usage.jsonl")


def set_token_log_path(directory):
    """Set the directory where token usage logs will be saved."""
    global TOKEN_LOG_PATH
    TOKEN_LOG_PATH = os.path.join(directory, "token_usage.jsonl")


def log_token_usage(usage, model_name, cached_content=None):
    """Log token usage and calculate price."""
    try:
        if usage:
            input_tokens = getattr(usage, "prompt_token_count", None) or 0
            output_tokens = getattr(usage, "candidates_token_count", None) or 0

            cached_tokens = 0
            if hasattr(usage, "cached_content_token_count"):
                cached_tokens = usage.cached_content_token_count or 0

            if cached_tokens == 0 and cached_content:
                try:
                    um = getattr(cached_content, "usage_metadata", None)
                    if um is not None:
                        cached_tokens = getattr(um, "total_token_count", 0) or 0
                except Exception as e:
                    print(f"Warning: Could not get cached token count from object: {e}")

            price_data = None
            try:
                from genai_prices import Usage, calc_price

                usage_obj = Usage(
                    input_tokens=input_tokens,
                    cache_read_tokens=cached_tokens if cached_tokens > 0 else None,
                    output_tokens=output_tokens,
                )
                price_data = calc_price(
                    usage_obj, model_ref=model_name, provider_id="google"
                )
            except Exception as e:
                print(f"Warning: Price calculation failed: {e}")

            log_entry = {
                "timestamp": time.time(),
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
            }

            if price_data:
                log_entry["input_price"] = float(price_data.input_price)
                log_entry["output_price"] = float(price_data.output_price)
                log_entry["total_price"] = float(price_data.total_price)

            log_dir = os.path.dirname(TOKEN_LOG_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            with open(TOKEN_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log token usage: {e}")


def _rate_limited(exc: Exception) -> bool:
    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc).upper():
        return True
    return getattr(exc, "code", None) == 429


def _prompt_blocked(exc: Exception) -> bool:
    name = type(exc).__name__
    if "Blocked" in name:
        return True
    s = str(exc).lower()
    return "blocked" in s and ("prompt" in s or "safety" in s)


def run_gemini(
    prompt,
    max_output_tokens=2048,
    temperature=0.8,
    model_name="gemini-2.5-flash",
    cached_content=None,
):
    """
    Run query through Google Gemini API with retry logic.

    cached_content: optional ``google.genai.types.CachedContent`` (or any object
    with a ``name`` attribute holding the cache resource name).
    """
    set_gemini_key()
    client = get_gemini_client()

    cfg_kwargs: dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    if cached_content is not None:
        cname = getattr(cached_content, "name", None) or str(cached_content)
        cfg_kwargs["cached_content"] = cname

    wait_time = 2
    max_retries = 5
    attempt = 0

    time.sleep(1)

    while attempt < max_retries:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            text = (response.text or "").strip() if response.text else None
            if text:
                log_token_usage(
                    response.usage_metadata, model_name, cached_content
                )
                return text

            print("Warning: Received an empty response from Gemini API.")
            return None

        except Exception as e:
            if _rate_limited(e):
                print(f"Rate limit error detected. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                attempt += 1
            elif _prompt_blocked(e):
                print(f"Prompt was blocked by the API: {e}")
                return None
            else:
                print(f"An unexpected error occurred: {type(e).__name__}: {e}")
                return None

    print("Max retries reached. Could not get a response from Gemini.")
    return None


def get_gemini_embedding(
    texts,
    model="models/gemini-embedding-001",
    task_type="RETRIEVAL_DOCUMENT",
    output_dimensionality=None,
):
    """Generate embeddings using the Gemini API (google-genai)."""
    set_gemini_key()
    client = get_gemini_client()

    ecfg: dict[str, Any] = {}
    if task_type:
        ecfg["task_type"] = task_type
    if output_dimensionality:
        ecfg["output_dimensionality"] = output_dimensionality
    embed_config = types.EmbedContentConfig(**ecfg) if ecfg else None

    wait_time = 2
    max_retries = 5
    attempt = 0

    while attempt < max_retries:
        try:
            resp = client.models.embed_content(
                model=model,
                contents=texts,
                config=embed_config,
            )
            embs = resp.embeddings or []
            if not embs:
                return None
            if isinstance(texts, str):
                return embs[0].values
            return [e.values for e in embs]
        except Exception as e:
            if _rate_limited(e):
                print(
                    f"Rate limit error detected for embeddings. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                wait_time *= 2
                attempt += 1
            else:
                print(
                    f"An unexpected error occurred during embedding generation: {type(e).__name__}: {e}"
                )
                return None

    print("Max retries reached. Could not get embeddings from Gemini.")
    return None
