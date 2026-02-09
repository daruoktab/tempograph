import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict
from genai_prices import Usage, calc_price

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Centralized cost tracker using genai-prices library.
    Thread-safe and async-friendly.
    Tracks per-model breakdown for detailed reporting.
    Now supports file-based logging for persistence.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CostTracker, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize all tracking variables"""
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._lock = asyncio.Lock()

        # Per-model breakdown
        self.model_stats = defaultdict(
            lambda: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "call_count": 0,
            }
        )

        # File logging (optional)
        self.log_file_path: Optional[Path] = None
        self.setup_name: Optional[str] = None  # "gemini" or "gemma"

    @classmethod
    def get_instance(cls) -> "CostTracker":
        if cls._instance is None:
            cls()
        assert cls._instance is not None
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton for fresh tracking"""
        if cls._instance is not None:
            cls._instance._initialize()

    def set_log_file(self, path: str, setup_name: str | None = None):
        """Set file path for logging each API call to JSONL"""
        self.log_file_path = Path(path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if setup_name:
            self.setup_name = setup_name
        logger.info(
            f"💾 Cost logging enabled: {self.log_file_path} (setup: {self.setup_name})"
        )

    def _save_entry(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        call_type: str = "llm",
    ):
        """Save a single entry to the log file"""
        if not self.log_file_path:
            return

        entry = {
            "timestamp": time.time(),
            "setup": self.setup_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "call_type": call_type,
        }

        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write cost log: {e}")

    async def track(self, input_tokens: int, output_tokens: int, model_name: str):
        """Track usage using genai-prices with per-model breakdown"""
        cost = 0.0

        # Clean model name
        model_ref = model_name.replace("models/", "")

        try:
            # Use genai_prices
            p = calc_price(
                Usage(input_tokens=int(input_tokens), output_tokens=int(output_tokens)),
                model_ref=model_ref,
                provider_id="google",
            )
            cost = float(p.total_price)
        except Exception as e:
            # Fallback for Embedding if genai_prices fails
            if "embedding" in model_name.lower():
                # Gemini Embedding API Pricing: $0.15 per 1M tokens
                price_per_token = 0.15 / 1_000_000
                cost = input_tokens * price_per_token
            elif "gemma" in model_name.lower():
                # Novita Gemma 3 27B IT: $0.0952/1M input, $0.16/1M output
                cost = (input_tokens * 0.0952 + output_tokens * 0.16) / 1_000_000
            else:
                # Manual fallback for Flash/Pro
                if "flash-lite" in model_name.lower():
                    # $0.10/1M input, $0.40/1M output
                    cost = (input_tokens * 0.10 + output_tokens * 0.40) / 1_000_000
                elif "flash" in model_name.lower():
                    # $0.30/1M input, $2.50/1M output
                    cost = (input_tokens * 0.30 + output_tokens * 2.50) / 1_000_000
                elif "pro" in model_name.lower():
                    # $1.25/1M input, $10.00/1M output
                    cost = (input_tokens * 1.25 + output_tokens * 10.00) / 1_000_000

        async with self._lock:
            # Global totals
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost

            # Per-model breakdown
            self.model_stats[model_ref]["input_tokens"] += input_tokens
            self.model_stats[model_ref]["output_tokens"] += output_tokens
            self.model_stats[model_ref]["cost_usd"] += cost
            self.model_stats[model_ref]["call_count"] += 1

        # Save to file (outside lock for performance)
        self._save_entry(model_ref, input_tokens, output_tokens, cost, "llm")

    async def track_chars(self, char_count: int, model_name: str):
        """Track based on characters (useful for embeddings)"""
        tokens = int(char_count / 4.0)
        price_per_token = 0.15 / 1_000_000
        cost = tokens * price_per_token

        model_ref = model_name.replace("models/", "")

        async with self._lock:
            self.total_input_tokens += tokens
            self.total_cost_usd += cost

            self.model_stats[model_ref]["input_tokens"] += tokens
            self.model_stats[model_ref]["cost_usd"] += cost
            self.model_stats[model_ref]["call_count"] += 1

        # Save to file
        self._save_entry(model_ref, tokens, 0, cost, "embedding")

    def get_stats(self) -> Dict[str, float]:
        return {
            "cost_usd": self.total_cost_usd,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    def get_model_breakdown(self) -> Dict[str, Dict]:
        """Get per-model breakdown"""
        return dict(self.model_stats)

    def get_summary(self) -> str:
        return f"💰 Cost: ${self.total_cost_usd:.4f}"

    def get_detailed_summary(self) -> str:
        """Get detailed breakdown by model"""
        lines = [f"💰 Total Cost: ${self.total_cost_usd:.4f}"]
        lines.append(
            f"   Input: {int(self.total_input_tokens):,} tokens | Output: {int(self.total_output_tokens):,} tokens"
        )
        lines.append("")
        lines.append("Per-Model Breakdown:")

        for model, stats in sorted(self.model_stats.items()):
            lines.append(f"  {model}:")
            lines.append(
                f"    Calls: {stats['call_count']} | Cost: ${stats['cost_usd']:.4f}"
            )
            lines.append(
                f"    In: {int(stats['input_tokens']):,} | Out: {int(stats['output_tokens']):,}"
            )

        if self.log_file_path:
            lines.append("")
            lines.append(f"📁 Log file: {self.log_file_path}")

        return "\n".join(lines)

    def save_summary(self, path: str):
        """Save summary to JSON file"""
        data = {
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "model_breakdown": dict(self.model_stats),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Cost summary saved to {path}")


# Global instance accessor
def get_cost_tracker() -> CostTracker:
    return CostTracker.get_instance()
