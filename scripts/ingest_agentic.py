#!/usr/bin/env python
"""Agentic RAG Ingestion Script (``scripts/ingest_agentic.py``).
============================

Ingest conversation sessions ke Neo4j untuk Agentic RAG.
Menggunakan Graphiti untuk fact extraction dan graph building.

PENTING: Ingestion dilakukan PER-SESSION, bukan per-turn!
- Sesuai real-world scenario: dalam 1 sesi, semua turn masih di context window
- RAG hanya dibutuhkan untuk retrieve info dari sesi LAIN
- LLM extraction lebih akurat karena bisa lihat full session context
- Lebih efisien: 100 LLM calls vs 1143 calls

Features:
- Sequential processing untuk fairness
- Rate limit handling dengan auto-retry untuk Gemini API
- Checkpoint/resume support
- Progress tracking
- Separate group_id untuk Gemini vs Gemma

Usage:
    python scripts/ingest_agentic.py --setup gemini
    python scripts/ingest_agentic.py --setup gemma
    python scripts/ingest_agentic.py --setup all
    python scripts/ingest_agentic.py --setup gemini --resume  # Resume dari checkpoint
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
import time
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# ============================================================
# SUPPRESS NOISY WARNINGS (before any imports that trigger them)
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"  # Suppress gRPC
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress TensorFlow oneDNN info
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Suppress MPS warnings
os.environ["TORCHAO_DISABLE_WARNINGS"] = "1"  # Suppress torchao warnings
os.environ["TRITON_CACHE_DIR"] = ""  # Suppress triton warnings
os.environ["TORCH_LOGS"] = "-all"  # Suppress all torch logs

# Suppress all warnings before imports
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", message=".*torchao.*")
warnings.filterwarnings("ignore", message=".*Redirects.*")
warnings.filterwarnings("ignore", message=".*cpp extensions.*")
warnings.filterwarnings("ignore", message=".*tf_keras.*")
warnings.filterwarnings("ignore", message=".*sparse_softmax.*")
warnings.filterwarnings("ignore", message=".*Skipping import.*")
warnings.filterwarnings("ignore", message=".*Detected no triton.*")

# Suppress Neo4j driver notifications (very noisy schema index logs)
logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.io").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.pool").setLevel(logging.CRITICAL)
logging.getLogger("neo4j.notifications").setLevel(logging.CRITICAL)
logging.getLogger("graphiti_core").setLevel(logging.WARNING)
logging.getLogger("graphiti_core.driver").setLevel(logging.CRITICAL)

# Suppress ML framework loggers
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.distributed").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Custom filter to suppress harmless Neo4j warnings
class Neo4jNoiseFilter(logging.Filter):
    def filter(self, record):
        msg = str(record.getMessage()).lower()
        # Suppress harmless warnings
        suppressed_patterns = [
            "equivalentschemarulealreadyexists",
            "equivalent index already exists",
            "equivalent schema rule already exists",
            "transaction failed and will be retried",
            "failed to read from defunct connection",
            "semaphore timeout period has expired",
            "connection error",
            "<connection> error",
            "error executing neo4j query",
            "neo4j_code:",
            "gql_status:",
            "create index",
            "create fulltext index",
        ]
        for pattern in suppressed_patterns:
            if pattern in msg:
                return False
        return True


# Apply filter to all relevant loggers
for logger_name in [
    "",
    "neo4j",
    "neo4j.io",
    "neo4j.pool",
    "graphiti_core",
    "graphiti_core.driver",
]:
    logging.getLogger(logger_name).addFilter(Neo4jNoiseFilter())

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.experiment_setups import ExperimentSetup  # noqa: E402
from src.rag.graph_client import TemporalGraphClient  # noqa: E402
from tqdm import tqdm  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = Path("output/final_dataset_v1/conversation_dataset.json")
CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IngestionCheckpoint:
    """Checkpoint for resuming ingestion"""

    setup_name: str
    total_sessions: int
    processed_sessions: int
    last_session_id: int
    started_at: str
    updated_at: str
    status: str  # "in_progress", "completed", "failed"
    error_message: Optional[str] = None
    entities_created: int = 0
    facts_created: int = 0
    episodes_created: int = 0

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IngestionCheckpoint":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# TOKEN BUDGET MANAGER - Fine-grained per-LLM-call budgeting
# =============================================================================


class TokenBudgetManager:
    """
    Proactive token budget management with sliding window.

    Key insight: Instead of waiting 65s for EVERY LLM call,
    we track actual token usage and only wait when approaching limit.

    OpenRouter Gemma limit: 15,000 tokens/minute
    We use 14,000 as target (1000 token buffer for safety)
    Estimated tokens per LLM call: ~5,000
    """

    TOKENS_PER_MINUTE = 14000  # Target limit (1000 buffer from 15k)
    WINDOW_SECONDS = 60

    # Estimated tokens per call type
    TOKENS_PER_CALL = 4000  # Conservative estimate (~3 calls per minute)

    def __init__(self):
        self.window_start = time.time()
        self.tokens_used = 0
        self.call_count = 0
        self.wait_count = 0
        self._lock = asyncio.Lock()

    def _reset_if_needed(self):
        """Reset window if expired"""
        now = time.time()
        if now - self.window_start >= self.WINDOW_SECONDS:
            self.window_start = now
            self.tokens_used = 0
            self.call_count = 0

    async def acquire(self, estimated_tokens: int | None = None):
        """
        Acquire tokens before making an LLM call.
        Waits if necessary to stay within budget.

        IMPORTANT: This now acquires a GLOBAL SEMAPHORE that serializes
        all LLM calls to prevent race conditions.
        """
        # First, acquire the global semaphore to serialize API calls
        # This prevents multiple concurrent calls from bypassing the budget
        await _llm_semaphore.acquire()

        try:
            async with self._lock:
                if estimated_tokens is None:
                    estimated_tokens = self.TOKENS_PER_CALL

                self._reset_if_needed()

                # Check if we would exceed budget
                if self.tokens_used + estimated_tokens > self.TOKENS_PER_MINUTE:
                    # Calculate wait time until window resets
                    elapsed = time.time() - self.window_start
                    wait_time = max(0, self.WINDOW_SECONDS - elapsed + 2)  # +2s buffer

                    if wait_time > 0:
                        self.wait_count += 1
                        logger.info(
                            f"⏸️  Token budget: {self.tokens_used}/{self.TOKENS_PER_MINUTE} used, waiting {wait_time:.0f}s for window reset"
                        )
                        await asyncio.sleep(wait_time)
                        # Reset after waiting
                        self.window_start = time.time()
                        self.tokens_used = 0
                        self.call_count = 0

                # Reserve tokens
                self.tokens_used += estimated_tokens
                self.call_count += 1
        except Exception:
            # Release semaphore if anything fails during acquire
            _llm_semaphore.release()
            raise

    def release_semaphore(self):
        """Release the global semaphore after API call completes"""
        try:
            _llm_semaphore.release()
        except ValueError:
            pass  # Already released

    def release(self, actual_tokens: int, estimated_tokens: int):
        """Adjust for actual vs estimated (optional refinement)"""
        diff = estimated_tokens - actual_tokens
        self.tokens_used = max(0, self.tokens_used - diff)

    def get_stats(self) -> dict:
        return {
            "tokens_used": self.tokens_used,
            "call_count": self.call_count,
            "wait_count": self.wait_count,
            "budget_remaining": self.TOKENS_PER_MINUTE - self.tokens_used,
        }


# Global semaphore to serialize LLM API calls (prevents race condition)
# Only 1 LLM call can be in-flight at a time to respect rate limits
_llm_semaphore = asyncio.Semaphore(1)

# Global token budget manager (shared across all LLM calls)
_token_budget = None


def get_token_budget() -> TokenBudgetManager:
    global _token_budget
    if _token_budget is None:
        _token_budget = TokenBudgetManager()
    return _token_budget


class RateLimitHandler:
    """
    Smart rate limit handling with PROACTIVE throttling.

    OpenRouter Gemma limit: 15,000 tokens/minute

    Strategy:
    1. Track estimated token usage per window (60s)
    2. Proactively slow down when approaching limit
    3. Only do long waits when actually hit limit
    4. Reset counter every 60s window
    """

    # OpenRouter Gemma limit
    TOKENS_PER_MIN = 15000
    WINDOW_SECONDS = 60

    # Estimated tokens per Graphiti operation
    # (input + output tokens)
    TOKENS_PER_EXTRACT_NODES = 2000
    TOKENS_PER_EXTRACT_EDGES = 2000
    TOKENS_PER_DEDUPE = 1500
    TOKENS_PER_SUMMARIZE = 500
    TOKENS_PER_OTHER = 800

    # Total per session: ~8-10 LLM calls
    ESTIMATED_TOKENS_PER_SESSION = 12000

    def __init__(
        self,
        base_delay: float = 2.0,
        max_delay: float = 120.0,  # Reduced from 300s
        tokens_per_min: int = TOKENS_PER_MIN,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.tokens_per_min = tokens_per_min

        # Token tracking
        self.tokens_used_in_window = 0
        self.window_start = time.time()

        # Stats
        self.consecutive_errors = 0
        self.total_rate_limit_waits = 0
        self.total_proactive_waits = 0

    def _reset_window_if_needed(self):
        """Reset token counter if window expired"""
        now = time.time()
        if now - self.window_start >= self.WINDOW_SECONDS:
            self.tokens_used_in_window = 0
            self.window_start = now

    def estimate_remaining_tokens(self) -> int:
        """How many tokens left in current window"""
        self._reset_window_if_needed()
        return max(0, self.tokens_per_min - self.tokens_used_in_window)

    def record_token_usage(self, tokens: int):
        """Record token usage"""
        self._reset_window_if_needed()
        self.tokens_used_in_window += tokens

    async def proactive_throttle(self):
        """
        Proactively wait if we're close to the limit.
        Called BEFORE each session to pace ourselves.
        """
        self._reset_window_if_needed()

        _remaining = self.estimate_remaining_tokens()
        usage_pct = (self.tokens_used_in_window / self.tokens_per_min) * 100

        # If we've used more than 80% of our budget, wait for window reset
        if usage_pct > 80:
            time_in_window = time.time() - self.window_start
            time_to_wait = max(0, self.WINDOW_SECONDS - time_in_window + 2)

            if time_to_wait > 0:
                self.total_proactive_waits += 1
                logger.info(
                    f"⏸️  Proactive throttle: {usage_pct:.0f}% used, waiting {time_to_wait:.0f}s for window reset"
                )
                await asyncio.sleep(time_to_wait)
                self._reset_window_if_needed()

        # Otherwise, just record that we're about to use tokens
        self.record_token_usage(self.ESTIMATED_TOKENS_PER_SESSION)

    def reset(self):
        """Reset after successful request"""
        self.consecutive_errors = 0

    async def wait_on_rate_limit(self, error_msg: str = ""):
        """Wait when rate limit is hit (reactive)"""
        self.consecutive_errors += 1
        self.total_rate_limit_waits += 1

        # When we hit rate limit, wait for window to reset
        time_in_window = time.time() - self.window_start
        wait_time = max(self.WINDOW_SECONDS - time_in_window + 5, 30)  # At least 30s
        wait_time = min(wait_time, self.max_delay)

        logger.warning(
            f"\n⏳ Rate limit hit! Waiting {wait_time:.0f}s for window reset..."
        )
        logger.warning(
            f"   Tokens used: {self.tokens_used_in_window}, Error: {error_msg[:100]}..."
        )

        await asyncio.sleep(wait_time)

        # Reset window after waiting
        self.tokens_used_in_window = 0
        self.window_start = time.time()


# =============================================================================
# COST TRACKING (ESTIMATED)
# =============================================================================

# Gemini 1.5 Flash Pricing (Approximate)
# Cost Tracker (Centralized)
from src.utils.cost_tracker import get_cost_tracker  # noqa: E402

# Enable file logging for cost tracking
_cost_tracker = get_cost_tracker()
_cost_tracker.set_log_file("output/ingestion_cost.jsonl")


class AgenticIngester:
    """
    Ingest sessions ke Neo4j untuk Agentic RAG.
    Menggunakan Graphiti untuk fact extraction.

    UNIT: Per-Session (bukan per-turn)
    - Setiap session = 1 episode di Graphiti
    - Graphiti akan extract facts dari seluruh session
    """

    def __init__(self, setup_name: str):
        """
        Args:
            setup_name: "gemini" or "gemma"
        """
        self.setup_name = setup_name
        self.rate_limiter = RateLimitHandler()
        self._graph: TemporalGraphClient | None = None
        self._setup: ExperimentSetup | None = None
        self._group_id: str | None = None

    def _agentic_ctx(self) -> tuple[ExperimentSetup, TemporalGraphClient]:
        if self._setup is None or self._graph is None:
            raise RuntimeError("AgenticIngester.initialize() must be called before use.")
        return self._setup, self._graph

    async def initialize(self):
        """Initialize Graphiti client"""
        from src.config.experiment_setups import (
            SETUP_1A_AGENTIC_GEMINI,
            SETUP_2A_AGENTIC_GEMMA,  # Gemma via Novita AI
        )

        # Set setup name in cost tracker for proper logging
        _cost_tracker.setup_name = self.setup_name

        # Get setup
        if self.setup_name == "gemini":
            # Use high-detail fact extraction setup
            self._setup = SETUP_1A_AGENTIC_GEMINI
        elif self.setup_name == "gemma":
            # Gemma via Novita AI (OpenAI-compatible)
            self._setup = SETUP_2A_AGENTIC_GEMMA
        else:
            raise ValueError(f"Unknown setup: {self.setup_name}")

        self._group_id = self._setup.storage.group_id

        llm_x = self._setup.llm_extraction
        assert llm_x is not None, "Agentic setup requires llm_extraction"

        logger.info(f"Initializing {self._setup.name}...")
        logger.info(f"  Group ID: {self._group_id}")
        logger.info(f"  LLM Extraction: {llm_x.name}")
        logger.info(f"  Embedder: {self._setup.embedder.name}")

        self._graph = TemporalGraphClient(setup=self._setup)
        await self._graph.initialize()
        logger.info(f"✅ Initialized {self._setup.name} (SurrealDB)")

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load conversation dataset"""
        logger.info(f"Loading dataset from {DATASET_PATH}...")

        with open(DATASET_PATH) as f:
            data = json.load(f)

        sessions = data["sessions"]
        total_turns = sum(len(s["turns"]) for s in sessions)

        logger.info(f"Loaded {len(sessions)} sessions ({total_turns} total turns)")
        return sessions

    def _session_to_text(self, session: Dict[str, Any]) -> str:
        """
        Convert session to single text for Graphiti episode.

        Format:
        [Session 1 - 2024-01-15]
        [User]: Halo, apa kabar?
        [Assistant]: Baik, ada yang bisa saya bantu?
        ...
        """
        lines = []

        # Header with session info
        session_id = session["session_id"]
        first_turn = session["turns"][0] if session["turns"] else {}
        timestamp = first_turn.get("timestamp", "")[:10]  # Just date
        lines.append(f"[Session {session_id} - {timestamp}]")

        # All turns
        for turn in session["turns"]:
            speaker = turn["speaker"].capitalize()
            text = turn["text"]
            lines.append(f"[{speaker}]: {text}")

        return "\n".join(lines)

    def _get_checkpoint_path(self) -> Path:
        """Get checkpoint file path"""
        return CHECKPOINT_DIR / f"agentic_{self.setup_name}_checkpoint.json"

    def _load_checkpoint(self) -> Optional[IngestionCheckpoint]:
        """Load checkpoint if exists"""
        path = self._get_checkpoint_path()
        if path.exists():
            try:
                return IngestionCheckpoint.load(path)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None

    def _save_checkpoint(self, checkpoint: IngestionCheckpoint):
        """Save checkpoint"""
        checkpoint.updated_at = datetime.now().isoformat()
        checkpoint.save(self._get_checkpoint_path())

    async def _add_episode_with_retry(
        self,
        content: str,
        source: str,
        reference_time: datetime,
        max_retries: int = 10,
        timeout_minutes: int = 10,
    ) -> bool:
        """
        Add episode dengan retry untuk rate limit.

        Untuk Gemini API: retry dengan backoff
        Untuk Gemma (local embedding): tetap perlu retry untuk LLM extraction

        Includes:
        - Timeout detection (default 10 minutes)
        - Heartbeat logging every 60 seconds
        """
        import asyncio

        async def _heartbeat_logger(session_name: str, interval: int = 60):
            """Log heartbeat every interval seconds"""
            elapsed = 0
            while True:
                await asyncio.sleep(interval)
                elapsed += interval
                mins = elapsed // 60
                logger.warning(
                    f"⏳ Still processing {session_name}... ({mins} min elapsed)"
                )
                if elapsed >= timeout_minutes * 60 - interval:
                    logger.error(
                        f"🚨 {session_name} taking too long! (>{timeout_minutes} min). May be stuck."
                    )

        for attempt in range(max_retries):
            try:
                _, graph = self._agentic_ctx()
                # Create heartbeat task
                heartbeat_task = asyncio.create_task(_heartbeat_logger(source))

                try:
                    # Run with timeout
                    await asyncio.wait_for(
                        graph.add_episode(
                            content=content,
                            name=source,
                            source_description=source,
                            reference_time=reference_time,
                        ),
                        timeout=timeout_minutes * 60,  # Convert to seconds
                    )
                finally:
                    # Cancel heartbeat
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass

                self.rate_limiter.reset()  # Success, reset backoff
                return True

            except asyncio.TimeoutError:
                logger.error(
                    f"⏰ TIMEOUT: {source} exceeded {timeout_minutes} minutes!"
                )
                logger.error(
                    "   This usually means SurrealDB connection died or LLM API is not responding."
                )
                logger.error(f"   Attempt {attempt + 1}/{max_retries} - will retry...")
                # Continue to retry
                continue

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limit error
                if any(
                    x in error_str
                    for x in ["rate", "limit", "429", "quota", "resource_exhausted"]
                ):
                    await self.rate_limiter.wait_on_rate_limit(str(e))
                    continue  # Retry
                else:
                    # Non-rate-limit error
                    logger.error(f"Non-rate-limit error: {e}")
                    raise

        raise Exception(
            f"Failed to add episode after {max_retries} retries due to rate limits/timeouts"
        )

    async def ingest(
        self, resume: bool = False, batch_size: int = 5, limit: Optional[int] = None
    ):
        """
        Ingest dataset into Graphiti.

        Args:
            resume: Resume from checkpoint
            batch_size: Process N episodes before pausing (for memory/safety)
            limit: Limit total episodes to process
        """
        # Load dataset
        sessions = self._load_dataset()

        # Limit sessions if requested
        if limit:
            sessions = sessions[:limit]
            logger.info(f"Limiting to {limit} sessions for testing.")

        total_sessions = len(sessions)

        setup, _ = self._agentic_ctx()
        llm_x = setup.llm_extraction
        assert llm_x is not None, "Agentic setup requires llm_extraction"

        # Check for existing checkpoint
        checkpoint = None
        start_index = 0

        if resume:
            checkpoint = self._load_checkpoint()
            # Resume from both 'in_progress', 'failed', AND 'batch_done' status
            if checkpoint and checkpoint.status in [
                "in_progress",
                "failed",
                "batch_done",
            ]:
                # Find the index to resume from
                for i, session in enumerate(sessions):
                    if session["session_id"] == checkpoint.last_session_id:
                        start_index = i + 1
                        break

                logger.info("📥 Resuming from checkpoint:")
                logger.info(
                    f"   Processed: {checkpoint.processed_sessions}/{checkpoint.total_sessions}"
                )
                logger.info(f"   Last session: {checkpoint.last_session_id}")
                logger.info(f"   Starting from index: {start_index}")
            elif checkpoint and checkpoint.status == "completed":
                logger.info(f"✅ Ingestion already completed for {self.setup_name}")
                return

        # Create new checkpoint if not resuming
        if checkpoint is None:
            checkpoint = IngestionCheckpoint(
                setup_name=self.setup_name,
                total_sessions=total_sessions,
                processed_sessions=0,
                last_session_id=0,
                started_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                status="in_progress",
            )

        # Calculate end index based on batch_size
        if batch_size:
            end_index = min(start_index + batch_size, total_sessions)
            batch_sessions = end_index - start_index
        else:
            end_index = total_sessions
            batch_sessions = total_sessions - start_index

        # Process sessions
        logger.info(f"\n{'=' * 60}")
        logger.info(f"INGESTING TO {setup.name.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total sessions: {total_sessions}")
        logger.info(f"Starting from: {start_index}")
        if batch_size:
            logger.info(
                f"📦 BATCH: Processing {batch_sessions} sessions (until index {end_index - 1})"
            )
        logger.info(f"Group ID: {self._group_id}")
        logger.info(f"LLM Extraction: {llm_x.name}")
        logger.info(f"Embedder: {setup.embedder.name}")
        logger.info("Unit: PER-SESSION (1 episode = 1 session)")
        logger.info(f"{'=' * 60}\n")

        pbar = tqdm(
            total=end_index, initial=start_index, desc=f"Ingesting ({self.setup_name})"
        )

        try:
            for i in range(start_index, end_index):
                session = sessions[i]
                session_id = session["session_id"]

                # Convert session to text
                session_text = self._session_to_text(session)

                # Parse timestamp from first turn
                first_turn = session["turns"][0] if session["turns"] else {}
                timestamp_str = first_turn.get("timestamp", "")
                try:
                    reference_time = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                except Exception:
                    reference_time = datetime.now()

                # Source identifier
                source = f"session_{session_id}"

                # PROACTIVE THROTTLE: Pace ourselves to stay within rate limits
                await self.rate_limiter.proactive_throttle()

                # Add episode with retry
                try:
                    await self._add_episode_with_retry(
                        content=session_text,
                        source=source,
                        reference_time=reference_time,
                    )
                    # Reset error counter on success
                    self.rate_limiter.reset()
                except Exception as e:
                    logger.error(f"❌ Failed to add session {session_id}: {e}")
                    checkpoint.status = "failed"
                    checkpoint.error_message = str(e)
                    self._save_checkpoint(checkpoint)
                    raise

                # Update checkpoint
                checkpoint.processed_sessions = i + 1
                checkpoint.last_session_id = session_id

                # Save checkpoint after EVERY session (not just batch)
                # This ensures proper resume on crash/interrupt
                self._save_checkpoint(checkpoint)

                # Update cost estimate in progress bar
                pbar.set_description(
                    f"Ingesting ({self.setup_name}) {get_cost_tracker().get_summary()}"
                )
                pbar.update(1)

            # Check if batch or full completion
            if end_index >= total_sessions:
                # Full completion
                checkpoint.status = "completed"
                self._save_checkpoint(checkpoint)

                pbar.close()

                # Get final stats
                stats = await self._get_stats()

                logger.info(f"\n{'=' * 60}")
                logger.info("✅ INGESTION COMPLETED!")
                logger.info(f"{'=' * 60}")
                logger.info(f"Setup: {setup.name}")
                logger.info(f"Group ID: {self._group_id}")
                logger.info(f"Entities: {stats.get('entities', 0)}")
                logger.info(f"Facts: {stats.get('facts', 0)}")
                logger.info(f"Episodes: {stats.get('episodes', 0)}")
                logger.info(
                    f"Proactive throttles: {self.rate_limiter.total_proactive_waits}"
                )
                logger.info(
                    f"Rate limit hits: {self.rate_limiter.total_rate_limit_waits}"
                )
                logger.info(
                    f"FINAL COST ESTIMATE:\n{get_cost_tracker().get_detailed_summary()}"
                )
                logger.info(f"{'=' * 60}\n")
            else:
                # Batch done, more to process
                checkpoint.status = "batch_done"
                self._save_checkpoint(checkpoint)

                pbar.close()

                remaining = total_sessions - end_index

                # Get stats so far
                stats = await self._get_stats()

                logger.info(f"\n{'=' * 60}")
                logger.info("📦 BATCH COMPLETE!")
                logger.info(f"{'=' * 60}")
                logger.info(
                    f"Processed: {checkpoint.processed_sessions}/{total_sessions} sessions"
                )
                logger.info(f"Remaining: {remaining} sessions")
                logger.info(f"Entities so far: {stats.get('entities', 0)}")
                logger.info(f"Facts so far: {stats.get('facts', 0)}")
                logger.info(
                    f"Proactive throttles: {self.rate_limiter.total_proactive_waits}"
                )
                logger.info(
                    f"Rate limit hits: {self.rate_limiter.total_rate_limit_waits}"
                )
                logger.info(f"COST SO FAR: {get_cost_tracker().get_summary()}")
                logger.info(f"{'=' * 60}")
                logger.info("\n🔄 To continue, run:")
                logger.info(
                    f"   python scripts/ingest_agentic.py --setup {self.setup_name} --resume --batch {batch_size or 5}"
                )
                logger.info(f"{'=' * 60}\n")

        except KeyboardInterrupt:
            logger.warning("\n⚠️ Ingestion interrupted by user")
            self._save_checkpoint(checkpoint)
            logger.info("Checkpoint saved. Resume with --resume flag.")
            raise

        except Exception as e:
            logger.error(f"\n❌ Ingestion failed: {e}")
            self._save_checkpoint(checkpoint)
            raise

    async def _get_stats(self) -> Dict[str, int]:
        """Get stats for the group"""
        try:
            if self._graph is None:
                return {"entities": 0, "facts": 0, "episodes": 0}
            return await self._graph.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get stats: {e}")
            return {"entities": 0, "facts": 0, "episodes": 0}

    async def close(self):
        """Close connections"""
        if self._graph:
            await self._graph.close()


async def ingest_setup(
    setup_name: str,
    resume: bool = False,
    batch_size: int = 5,
    limit: Optional[int] = None,
):
    """Run ingestion for a specific setup."""
    ingester = AgenticIngester(setup_name)
    try:
        await ingester.initialize()
        await ingester.ingest(resume=resume, batch_size=batch_size, limit=limit)
    finally:
        await ingester.close()


async def clear_group(group_id: str):
    """Clear all SurrealDB records for a logical group_id."""
    from src.rag.graph_client import TemporalGraphClient

    client = TemporalGraphClient(group_id=group_id)
    try:
        await client.initialize()
        await client.clear_group()
        logger.info(f"Cleared SurrealDB group: {group_id}")
    finally:
        await client.close()


async def main():
    parser = argparse.ArgumentParser(description="Ingest data for Agentic RAG")
    parser.add_argument(
        "--setup",
        choices=["gemini", "gemma", "all"],
        required=True,
        help="Which setup to ingest",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing data before ingesting"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=5,  # Changed default from None to 5 as per user's implied change
        help="Process only N sessions per run (for safer incremental ingestion)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of episodes (for testing)"
    )

    args = parser.parse_args()

    # Show batch mode info
    if args.batch:
        logger.info(f"\n📦 BATCH MODE: Processing {args.batch} sessions per run")
        logger.info("   Use --resume to continue from last checkpoint\n")

    # Clear if requested
    if args.clear:
        from src.config.experiment_setups import NEO4J_GROUP_IDS, SetupType

        if args.setup in ["gemini", "all"]:
            await clear_group(NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMINI])

        if args.setup in ["gemma", "all"]:
            await clear_group(NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMMA])

    # Ingest
    if args.setup == "all":
        # Sequential untuk fairness
        logger.info("\n" + "=" * 60)
        logger.info("SEQUENTIAL INGESTION: GEMINI FIRST, THEN GEMMA")
        logger.info("=" * 60 + "\n")

        await ingest_setup(
            "gemini", resume=args.resume, batch_size=args.batch, limit=args.limit
        )
        await ingest_setup(
            "gemma", resume=args.resume, batch_size=args.batch, limit=args.limit
        )
    else:
        await ingest_setup(
            args.setup, resume=args.resume, batch_size=args.batch, limit=args.limit
        )

    # Show final status
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATABASE STATUS")
    logger.info("=" * 60)

    # Show final status (manage_database.py was deleted, no longer needed)
    logger.info(
        "Use 'python scripts/clear_database.py' to view/manage database status."
    )


if __name__ == "__main__":
    asyncio.run(main())
