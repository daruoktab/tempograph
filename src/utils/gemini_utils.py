"""
Global utility methods for API calls to Google Gemini.
"""
import os
import time

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()


def set_gemini_key():
    """Configure Google Gemini API with key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    # The type ignore is kept because the genai library's dynamic
    # nature can sometimes confuse static type checkers.
    genai.configure(api_key=api_key)  # type: ignore[attr-defined]


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
            input_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            
            # Calculate Price
            cached_tokens = 0
            
            # Method 1: Try to get from response usage metadata (Standard way)
            if hasattr(usage, 'cached_content_token_count'):
                cached_tokens = usage.cached_content_token_count
            
            # Method 2: Fallback to cached_content object if provided and Method 1 failed
            if cached_tokens == 0 and cached_content:
                try:
                    if hasattr(cached_content, 'usage_metadata'):
                            cached_tokens = cached_content.usage_metadata.total_token_count
                except Exception as e:
                    print(f"Warning: Could not get cached token count from object: {e}")
            
            price_data = None
            try:
                from genai_prices import Usage, calc_price
                # Note: genai_prices expects 'input_tokens' to be the TOTAL prompt tokens.
                # It subtracts 'cache_read_tokens' internally to find the uncached count.
                usage_obj = Usage(
                    input_tokens=input_tokens,
                    cache_read_tokens=cached_tokens if cached_tokens > 0 else None,
                    output_tokens=output_tokens
                )
                # Use google provider for Gemini models
                price_data = calc_price(usage_obj, model_ref=model_name, provider_id='google')
            except Exception as e:
                print(f"Warning: Price calculation failed: {e}")

            log_entry = {
                "timestamp": time.time(),
                "model": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens
            }
            
            if price_data:
                log_entry["input_price"] = float(price_data.input_price)
                log_entry["output_price"] = float(price_data.output_price)
                log_entry["total_price"] = float(price_data.total_price)

            # Append to log file
            log_dir = os.path.dirname(TOKEN_LOG_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            with open(TOKEN_LOG_PATH, "a", encoding="utf-8") as f:
                import json
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log token usage: {e}")

def run_gemini(prompt, max_output_tokens=2048, temperature=0.8,
               model_name="gemini-2.5-flash", cached_content=None):
    """
    Run query through Google Gemini API with retry logic.

    Args:
        prompt (str): The text prompt to send to the model.
        max_output_tokens (int): The maximum number of tokens to generate.
        temperature (float): The sampling temperature.
    model_name (str): The name of the Gemini model to use.
        cached_content (any, optional): The cached content object to use for the model. Defaults to None.

    Returns:
        str or None: The generated text from the model, or None if an error occurs.
    """
    # Ensure the API key is set before making a call
    # This is a safeguard in case the main script doesn't call it.
    set_gemini_key()

    if cached_content:
        # Initialize model from cache
        model = genai.GenerativeModel.from_cached_content(cached_content=cached_content)
    else:
        model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]

    # Exponential backoff for handling rate limits
    wait_time = 2
    max_retries = 5
    attempt = 0

    # Constant 1-second delay between all calls to be safe
    time.sleep(1)

    while attempt < max_retries:
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(  # type: ignore[attr-defined]
                    max_output_tokens=max_output_tokens,
                    temperature=temperature
                )
            )
            # Check for valid response text
            if response.text:
                # --- LOG TOKEN USAGE ---
                log_token_usage(response.usage_metadata, model_name, cached_content)
                # -----------------------

                return response.text.strip()

            # Handle cases where the response is empty but not an error
            print("Warning: Received an empty response from Gemini API.")
            return None

        # pylint: disable=broad-exception-caught
        except Exception as e:
            # Check for specific, recoverable errors like rate limiting
            if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
                print(f"Rate limit error detected. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Increase wait time for the next potential retry
                attempt += 1
            # Handle cases where the prompt was blocked
            elif "BlockedPromptException" in type(e).__name__:
                print(f"Prompt was blocked by the API: {e}")
                return None
            # Handle other non-recoverable API errors
            else:
                print(f"An unexpected error occurred: {type(e).__name__}: {e}")
                return None

    print("Max retries reached. Could not get a response from Gemini.")
    return None


def get_gemini_embedding(texts, model="models/embedding-001", task_type="RETRIEVAL_DOCUMENT", output_dimensionality=None):
    """
    Generate embeddings for a list of texts using the Gemini API.

    Args:
        texts (list[str] or str): The text or list of texts to embed.
        model (str): The name of the embedding model to use.
        task_type (str): The task type for the embedding.
        output_dimensionality (int, optional): The desired dimension of the output embedding. Defaults to None.

    Returns:
        list[list[float]] or list[float] or None: A list of embeddings, a single embedding, or None if an error occurs.
    """
    set_gemini_key()

    wait_time = 2
    max_retries = 5
    attempt = 0

    while attempt < max_retries:
        try:
            result = genai.embed_content(  # type: ignore[attr-defined]
                model=model,
                content=texts,
                task_type=task_type,
                output_dimensionality=output_dimensionality if output_dimensionality else None
            )
            return result['embedding']
        # pylint: disable=broad-exception-caught
        except Exception as e:
            if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
                print(f"Rate limit error detected for embeddings. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                attempt += 1
            else:
                print(f"An unexpected error occurred during embedding generation: {type(e).__name__}: {e}")
                return None

    print("Max retries reached. Could not get embeddings from Gemini.")
    return None
