#!/usr/bin/env python
# scripts/ingest_vanilla.py
"""
Vanilla-only ingestion (dense ``session_passage`` rows).

**Disarankan:** ``scripts/ingest_agentic.py`` — satu alur mengisi graph **dan**
``session_passage`` (hybrid / eval tanpa jalur ingest ganda).

Skrip ini tetap berguna untuk baseline **hanya** vektor sesi (tanpa ekstraksi fakta / graph).

PENTING: Ingestion dilakukan PER-SESSION, bukan per-turn!
- Sesuai real-world scenario: dalam 1 sesi, semua turn masih di context window
- RAG hanya dibutuhkan untuk retrieve info dari sesi LAIN
- Lebih efisien: 100 embedding calls vs 1143 calls

Features:
- Sequential processing untuk fairness
- Rate limit handling dengan auto-retry untuk Gemini API
- Checkpoint/resume support
- Progress tracking

Usage:
    python scripts/ingest_vanilla.py --setup gemini
    python scripts/ingest_vanilla.py --setup gemma
    python scripts/ingest_vanilla.py --setup all
    python scripts/ingest_vanilla.py --setup gemini --resume  # Resume dari checkpoint
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.experiment_setups import ExperimentSetup
from src.embedders.base import BaseEmbedder
from src.rag.vectordb import SurrealVanillaVectorDB

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = Path("output/example_dataset/conversation_dataset.json")
CHECKPOINT_DIR = Path("data/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# from src.utils.cost_tracker import get_cost_tracker
# tracker = get_cost_tracker()


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

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IngestionCheckpoint":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class RateLimitHandler:
    """
    Handle rate limits with exponential backoff.
    Khusus untuk Gemini API embedding.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 300.0,  # 5 minutes max
        backoff_factor: float = 2.0,
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = base_delay
        self.consecutive_errors = 0
        self.total_rate_limit_waits = 0

    def reset(self):
        """Reset after successful request"""
        self.current_delay = self.base_delay
        self.consecutive_errors = 0

    async def wait_on_rate_limit(self, error_msg: str = ""):
        """Wait when rate limit is hit"""
        self.consecutive_errors += 1
        self.total_rate_limit_waits += 1

        wait_time = min(self.current_delay, self.max_delay)

        logger.warning(f"⏳ Rate limit hit! Waiting {wait_time:.1f}s before retry...")
        logger.warning(f"   Error: {error_msg[:100]}...")
        logger.warning(f"   Consecutive errors: {self.consecutive_errors}")

        # Show countdown
        for remaining in range(int(wait_time), 0, -10):
            logger.info(f"   Resuming in {remaining}s...")
            await asyncio.sleep(min(10, remaining))

        # Increase delay for next time
        self.current_delay = min(
            self.current_delay * self.backoff_factor, self.max_delay
        )


class VanillaIngester:
    """
    Ingest sessions ke ChromaDB untuk Vanilla RAG.

    UNIT: Per-Turn (TIDAK per-session!)
    - Setiap turn = 1 document di ChromaDB
    - Ini membuat retrieval lebih challenging (harus menemukan turn spesifik)
    - Fair comparison dengan Agentic RAG yang juga retrieve facts terpisah
    """

    def __init__(self, setup_name: str):
        """
        Args:
            setup_name: "gemini" or "gemma"
        """
        self.setup_name = setup_name
        self.rate_limiter = RateLimitHandler()
        self._embedder: Optional[BaseEmbedder] = None
        self._chroma_db: Optional[SurrealVanillaVectorDB] = None
        self._setup: Optional[ExperimentSetup] = None

    def _ingest_ctx(self) -> Tuple[ExperimentSetup, BaseEmbedder, SurrealVanillaVectorDB]:
        """Resolved components after ``initialize()``; raises if not ready."""
        if self._setup is None or self._embedder is None or self._chroma_db is None:
            raise RuntimeError("VanillaIngester not initialized; call await initialize() first")
        return self._setup, self._embedder, self._chroma_db

    async def initialize(self):
        """Initialize embedder and ChromaDB"""
        from src.config.experiment_setups import (
            SETUP_1V_VANILLA_GEMINI,
            SETUP_2V_VANILLA_GEMMA,
        )
        from src.rag.vectordb import get_chroma_client
        from src.embedders import create_embedder, EmbedderType
        from src.config.settings import get_config

        # Get config for API key
        config = get_config()

        # Get setup
        if self.setup_name == "gemini":
            self._setup = SETUP_1V_VANILLA_GEMINI
        elif self.setup_name == "gemma":
            self._setup = SETUP_2V_VANILLA_GEMMA
        else:
            raise ValueError(f"Unknown setup: {self.setup_name}")

        logger.info(f"Initializing {self._setup.name}...")

        # Create embedder
        if self._setup.embedder.provider == "huggingface":
            logger.info(
                f"Using LOCAL HuggingFace embedder: {self._setup.embedder.name}"
            )
            self._embedder = create_embedder(
                embedder_type=EmbedderType.HUGGINGFACE,
                model_name=self._setup.embedder.name,
            )
        else:
            logger.info(f"Using Gemini API embedder: {self._setup.embedder.name}")
            self._embedder = create_embedder(
                embedder_type=EmbedderType.GEMINI,
                model_name=self._setup.embedder.name,
                gemini_api_key=config.gemini.api_key,
            )

        await self._embedder.initialize()

        # Get ChromaDB client
        assert self._setup.storage.collection_name is not None
        self._chroma_db = get_chroma_client(
            collection_name=self._setup.storage.collection_name,
            persist_directory=self._setup.storage.persist_directory,
        )
        await self._chroma_db.initialize(embedder=self._embedder)

        logger.info(f"✅ Initialized {self._setup.name}")
        logger.info(f"   Collection: {self._setup.storage.collection_name}")
        logger.info(f"   Embedder: {self._setup.embedder.name}")
        logger.info(f"   Current documents: {self._chroma_db.count()}")

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load conversation dataset (sessions)"""
        logger.info(f"Loading dataset from {DATASET_PATH}...")

        with open(DATASET_PATH) as f:
            data = json.load(f)

        sessions = data["sessions"]
        total_turns = sum(len(s["turns"]) for s in sessions)

        logger.info(f"Loaded {len(sessions)} sessions ({total_turns} total turns)")
        return sessions

    def _session_to_text(self, session: Dict[str, Any]) -> str:
        """
        Convert session to single text document.

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
        return CHECKPOINT_DIR / f"vanilla_{self.setup_name}_checkpoint.json"

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

    async def _embed_with_retry(self, text: str, max_retries: int = 10) -> List[float]:
        """
        Embed text dengan retry untuk rate limit.

        Untuk Gemini API: retry dengan backoff
        Untuk HuggingFace (local): langsung embed tanpa retry
        """
        setup, embedder, _ = self._ingest_ctx()
        is_local = setup.embedder.provider == "huggingface"

        if is_local:
            # Local embedder, tidak perlu rate limit handling
            return await embedder.embed_single(text)

        # Gemini API - perlu rate limit handling
        for attempt in range(max_retries):
            try:
                # Track cost (input chars)
                if not is_local:
                    from src.utils.cost_tracker import get_cost_tracker

                    tracker = get_cost_tracker()
                    await tracker.track_chars(len(text), setup.embedder.name)

                embedding = await embedder.embed_single(text)
                self.rate_limiter.reset()  # Success, reset backoff
                return embedding

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
                    # Non-rate-limit error, raise immediately
                    raise

        raise Exception(
            f"Failed to embed after {max_retries} retries due to rate limits"
        )

    async def ingest(
        self,
        resume: bool = False,
        checkpoint_interval: int = 10,
        limit: Optional[int] = None,
    ):
        """
        Ingest all sessions ke ChromaDB.

        Args:
            resume: Resume dari checkpoint jika ada
            checkpoint_interval: Simpan checkpoint setiap N sessions
            limit: Limit number of sessions to process (for smoke testing)
        """
        # Load dataset
        sessions = self._load_dataset()
        if limit:
            sessions = sessions[:limit]
            logger.info(f"Limiting to {limit} sessions for testing.")

        setup, _, chroma_db = self._ingest_ctx()

        total_sessions = len(sessions)

        # Check for existing checkpoint
        checkpoint = None
        start_index = 0

        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.status == "in_progress":
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
                logger.info(f"   Total documents: {chroma_db.count()}")
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

        # Process sessions
        logger.info(f"\n{'=' * 60}")
        logger.info(f"INGESTING TO {setup.name.upper()}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total sessions: {total_sessions}")
        logger.info(f"Starting from: {start_index}")
        logger.info(f"Embedder: {setup.embedder.name}")
        logger.info(f"Collection: {setup.storage.collection_name}")
        logger.info("Unit: PER-TURN (1 doc = 1 turn)")
        logger.info(f"{'=' * 60}\n")

        from src.rag.vectordb import VanillaDocument

        # Count total turns for progress bar
        total_turns = sum(len(s["turns"]) for s in sessions)

        pbar = tqdm(total=total_turns, desc=f"Ingesting ({self.setup_name})")

        docs_added = 0

        try:
            for i in range(start_index, total_sessions):
                session = sessions[i]
                session_id = session["session_id"]

                # Process each turn in the session
                for turn_idx, turn in enumerate(session["turns"]):
                    # Create turn text with context
                    # Dataset uses 'speaker' and 'text', not 'role' and 'content'
                    speaker = turn.get("speaker", "unknown")
                    text = turn.get("text", "")
                    turn_text = f"[{speaker}]: {text}"

                    # Create document ID
                    doc_id = f"session_{session_id}_turn_{turn_idx}"

                    # Embed with retry for rate limits
                    try:
                        embedding = await self._embed_with_retry(turn_text)
                    except Exception as e:
                        logger.error(f"❌ Failed to embed turn {doc_id}: {e}")
                        checkpoint.status = "failed"
                        checkpoint.error_message = str(e)
                        self._save_checkpoint(checkpoint)
                        raise

                    # Create document with turn metadata
                    doc = VanillaDocument(
                        id=doc_id,
                        text=turn_text,
                        embedding=embedding,
                        metadata={
                            "session_id": session_id,
                            "turn_index": turn_idx,
                            "speaker": speaker,
                            "timestamp": turn.get("timestamp", ""),
                            "char_count": len(turn_text),
                        },
                    )

                    # Add to ChromaDB
                    await chroma_db.add_document(doc)
                    docs_added += 1

                    from src.utils.cost_tracker import get_cost_tracker

                    pbar.set_description(
                        f"Ingesting ({self.setup_name}) {get_cost_tracker().get_summary()}"
                    )
                    pbar.update(1)

                    # Small delay for API rate limiting (Gemini only)
                    if setup.embedder.provider != "huggingface":
                        await asyncio.sleep(0.05)  # 50ms between requests

                # Update checkpoint after each session
                checkpoint.processed_sessions = i + 1
                checkpoint.last_session_id = session_id

                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(checkpoint)
                    logger.debug(f"Checkpoint saved at session {i + 1}")

            # Mark as completed
            checkpoint.status = "completed"
            self._save_checkpoint(checkpoint)

            pbar.close()

            logger.info(f"\n{'=' * 60}")
            logger.info("✅ INGESTION COMPLETED!")
            logger.info(f"{'=' * 60}")
            logger.info(f"Setup: {setup.name}")
            logger.info(f"Total documents (turns): {chroma_db.count()}")
            from src.utils.cost_tracker import get_cost_tracker

            logger.info(f"Total Cost: {get_cost_tracker().get_summary()}")
            logger.info(f"Rate limit waits: {self.rate_limiter.total_rate_limit_waits}")
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


async def ingest_setup(
    setup_name: str, resume: bool = False, limit: Optional[int] = None
):
    """Ingest untuk satu setup"""
    ingester = VanillaIngester(setup_name)
    await ingester.initialize()
    await ingester.ingest(resume=resume, limit=limit)


async def main():
    parser = argparse.ArgumentParser(description="Ingest data for Vanilla RAG")
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
        "--limit",
        type=int,
        default=None,
        help="Limit number of sessions (for smoke testing)",
    )

    args = parser.parse_args()

    # Clear if requested
    if args.clear:
        from src.rag.vectordb import get_chroma_client
        from src.config.experiment_setups import CHROMA_PERSIST_DIR

        if args.setup in ["gemini", "all"]:
            db = get_chroma_client("vanilla_gemini", CHROMA_PERSIST_DIR)
            await db.initialize()
            await db.clear()
            logger.info("Cleared vanilla_gemini collection")

        if args.setup in ["gemma", "all"]:
            db = get_chroma_client("vanilla_gemma", CHROMA_PERSIST_DIR)
            await db.initialize()
            await db.clear()
            logger.info("Cleared vanilla_gemma collection")

    # Ingest
    if args.setup == "all":
        # Sequential untuk fairness
        logger.info("\n" + "=" * 60)
        logger.info("SEQUENTIAL INGESTION: GEMINI FIRST, THEN GEMMA")
        logger.info("=" * 60 + "\n")

        await ingest_setup("gemini", resume=args.resume, limit=args.limit)
        await ingest_setup("gemma", resume=args.resume, limit=args.limit)
    else:
        await ingest_setup(args.setup, resume=args.resume, limit=args.limit)

    # Show final status
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATABASE STATUS")
    logger.info("=" * 60)

    from src.rag.vectordb import get_chroma_client
    from src.config.experiment_setups import CHROMA_PERSIST_DIR

    for name in ["vanilla_gemini", "vanilla_gemma"]:
        db = get_chroma_client(name, CHROMA_PERSIST_DIR)
        await db.initialize()
        logger.info(f"  {name}: {db.count()} documents (sessions)")


if __name__ == "__main__":
    asyncio.run(main())
