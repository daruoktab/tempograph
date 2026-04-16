# src/ingestion/episode_ingester.py
"""
Episode Ingester
================
Ingest conversation turns sebagai episodes ke Temporal Knowledge Graph.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from ..graph_client import TemporalGraphClient
from ...config.settings import get_config, IngestionConfig

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Represents a single conversation turn"""

    session_id: int
    turn_id: int
    speaker: str
    text: str
    timestamp: datetime
    ground_truth: Optional[Dict[str, Any]] = None


@dataclass
class Session:
    """Represents a conversation session"""

    session_id: int
    date: str
    datetime_str: Optional[str]
    turns: List[Turn]
    summary: Optional[str] = None
    relevant_events: Optional[List[Dict]] = None


class EpisodeIngester:
    """
    Ingester untuk memasukkan conversation turns ke knowledge graph.

    Features:
    - Batch processing untuk efisiensi
    - Temporal metadata dari session date
    - Error handling dengan retry
    - Progress tracking
    """

    def __init__(
        self,
        graph_client: TemporalGraphClient,
        config: Optional[IngestionConfig] = None,
    ):
        self.client = graph_client
        self.config = config or get_config().ingestion
        self._ingested_count = 0
        self._error_count = 0

    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load conversation dataset from JSON file"""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded dataset from {dataset_path}")
        return data

    def parse_sessions(self, data: Dict[str, Any]) -> List[Session]:
        """Parse sessions from dataset"""
        sessions = []

        for session_data in data.get("sessions", []):
            session_id = session_data["session_id"]
            date_str = session_data.get("date", "")
            datetime_str = session_data.get("datetime", "")

            # Parse date untuk timestamp
            try:
                if datetime_str:
                    # Format: "01 January 2024, 12:37"
                    base_time = datetime.strptime(datetime_str, "%d %B %Y, %H:%M")
                elif date_str:
                    # Format: "2024-01-01"
                    base_time = datetime.strptime(date_str, "%Y-%m-%d")
                else:
                    base_time = datetime.now()
            except ValueError:
                logger.warning(
                    f"Could not parse date for session {session_id}, using current time"
                )
                base_time = datetime.now()

            # Parse turns
            turns = []
            ground_truths = {
                gt["turn_id"]: gt for gt in session_data.get("ground_truths", [])
            }

            for idx, turn_data in enumerate(session_data.get("turns", [])):
                # Increment timestamp per turn (simulate conversation flow)
                turn_time = base_time + timedelta(minutes=idx * 2)

                turn = Turn(
                    session_id=session_id,
                    turn_id=idx,
                    speaker=turn_data["speaker"],
                    text=turn_data["text"],
                    timestamp=turn_time,
                    ground_truth=ground_truths.get(idx),
                )
                turns.append(turn)

            session = Session(
                session_id=session_id,
                date=date_str,
                datetime_str=datetime_str,
                turns=turns,
                summary=session_data.get("summary"),
                relevant_events=session_data.get("relevant_events"),
            )
            sessions.append(session)

        logger.info(
            f"Parsed {len(sessions)} sessions with {sum(len(s.turns) for s in sessions)} total turns"
        )
        return sessions

    async def ingest_turn(self, turn: Turn) -> bool:
        """Ingest a single turn to the graph"""
        try:
            name = f"Session {turn.session_id} Turn {turn.turn_id}"
            source_desc = (
                f"Session {turn.session_id} Turn {turn.turn_id} Speaker {turn.speaker}"
            )

            await self.client.add_episode(
                content=turn.text,
                name=name,
                source_description=source_desc,
                reference_time=turn.timestamp,
            )

            self._ingested_count += 1
            return True

        except Exception as e:
            logger.error(f"Error ingesting turn {turn.session_id}-{turn.turn_id}: {e}")
            self._error_count += 1
            return False

    async def ingest_session(self, session: Session, show_progress: bool = True) -> int:
        """
        Ingest all turns from a session.

        Args:
            session: Session to ingest
            show_progress: Whether to show progress bar

        Returns:
            Number of successfully ingested turns
        """
        success_count = 0

        iterator = (
            tqdm(session.turns, desc=f"Session {session.session_id}")
            if show_progress
            else session.turns
        )

        for turn in iterator:
            success = await self.ingest_turn(turn)
            if success:
                success_count += 1

            # Rate limiting
            await asyncio.sleep(0.1)

        return success_count

    async def ingest_dataset(
        self,
        dataset_path: str,
        limit_sessions: Optional[int] = None,
        limit_turns: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """
        Ingest entire dataset to knowledge graph.

        Args:
            dataset_path: Path to conversation_dataset.json
            limit_sessions: Max sessions to process (for testing)
            limit_turns: Max total turns to process
            show_progress: Whether to show progress

        Returns:
            Statistics dict with ingested/error counts
        """
        # Load and parse dataset
        data = self.load_dataset(dataset_path)
        sessions = self.parse_sessions(data)

        if limit_sessions:
            sessions = sessions[:limit_sessions]

        logger.info(f"Starting ingestion of {len(sessions)} sessions...")

        total_turns = 0
        for session in sessions:
            if limit_turns and total_turns >= limit_turns:
                break

            remaining = limit_turns - total_turns if limit_turns else None
            turns_to_process = session.turns[:remaining] if remaining else session.turns

            # Temporarily replace session turns
            original_turns = session.turns
            session.turns = turns_to_process

            count = await self.ingest_session(session, show_progress)
            total_turns += count

            session.turns = original_turns

        stats = {
            "sessions_processed": min(len(sessions), limit_sessions)
            if limit_sessions
            else len(sessions),
            "turns_ingested": self._ingested_count,
            "errors": self._error_count,
            "group_id": self.client.group_id,
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats  # type: ignore[invalid-return-type]

    def get_stats(self) -> Dict[str, int]:
        """Get current ingestion statistics"""
        return {"ingested": self._ingested_count, "errors": self._error_count}


async def run_ingestion(
    dataset_path: str,
    group_id: Optional[str] = None,
    limit_sessions: Optional[int] = None,
    limit_turns: Optional[int] = None,
) -> Dict[str, int]:
    """
    Convenience function to run ingestion.

    Args:
        dataset_path: Path to dataset
        group_id: Optional group ID (auto-generated if not provided)
        limit_sessions: Limit number of sessions
        limit_turns: Limit total turns

    Returns:
        Ingestion statistics
    """
    client = TemporalGraphClient(group_id=group_id)

    try:
        await client.initialize()

        ingester = EpisodeIngester(client)
        stats = await ingester.ingest_dataset(
            dataset_path, limit_sessions=limit_sessions, limit_turns=limit_turns
        )

        # Get graph stats
        graph_stats = await client.get_stats()
        stats.update({"graph_" + k: v for k, v in graph_stats.items()})

        return stats

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest conversation dataset to knowledge graph"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/example_dataset/conversation_dataset.json",
        help="Path to dataset",
    )
    parser.add_argument(
        "--limit-sessions", type=int, default=None, help="Limit number of sessions"
    )
    parser.add_argument(
        "--limit-turns", type=int, default=None, help="Limit total turns"
    )
    parser.add_argument(
        "--group-id", type=str, default=None, help="Group ID for the ingestion run"
    )

    args = parser.parse_args()

    stats = asyncio.run(
        run_ingestion(
            dataset_path=args.dataset,
            group_id=args.group_id,
            limit_sessions=args.limit_sessions,
            limit_turns=args.limit_turns,
        )
    )

    print("\n" + "=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")
