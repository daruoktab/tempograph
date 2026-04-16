# src/evaluation/evaluator.py
"""
RAG Evaluator
=============
Orchestrator untuk evaluasi end-to-end RAG system.
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tqdm import tqdm

from ..rag.graph_client import TemporalGraphClient
from ..rag.ingestion import EpisodeIngester
from ..rag.retrieval import RetrievalAgent
from .metrics import calculate_all_metrics
from ..config.settings import get_config, EvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class TurnEvaluation:
    """Evaluation result for a single turn"""

    session_id: int
    turn_id: int
    speaker: str

    # Metrics
    context_recall: float = 0.0
    context_precision: float = 0.0
    temporal_recall: float = 0.0
    fact_coverage: float = 0.0

    # Metadata
    retrieved_count: int = 0
    retrieval_iterations: int = 0
    query_type: str = "unknown"

    # Details (optional)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionEvaluation:
    """Aggregated evaluation for a session"""

    session_id: int
    turns: List[TurnEvaluation]

    # Aggregated metrics
    avg_context_recall: float = 0.0
    avg_context_precision: float = 0.0
    avg_temporal_recall: float = 0.0
    avg_fact_coverage: float = 0.0

    def calculate_averages(self):
        """Calculate average metrics from turns"""
        if not self.turns:
            return

        n = len(self.turns)
        self.avg_context_recall = sum(t.context_recall for t in self.turns) / n
        self.avg_context_precision = sum(t.context_precision for t in self.turns) / n
        self.avg_temporal_recall = sum(t.temporal_recall for t in self.turns) / n
        self.avg_fact_coverage = sum(t.fact_coverage for t in self.turns) / n


@dataclass
class EvaluationReport:
    """Complete evaluation report"""

    # Metadata
    run_id: str
    dataset_path: str
    timestamp: str
    group_id: str

    # Configuration
    config: Dict[str, Any]

    # Results
    session_evaluations: List[SessionEvaluation]

    # Aggregated metrics
    overall_context_recall: float = 0.0
    overall_context_precision: float = 0.0
    overall_temporal_recall: float = 0.0
    overall_fact_coverage: float = 0.0

    # Statistics
    total_sessions: int = 0
    total_turns: int = 0
    total_retrieval_iterations: int = 0

    def calculate_overall(self):
        """Calculate overall metrics"""
        if not self.session_evaluations:
            return

        self.total_sessions = len(self.session_evaluations)
        self.total_turns = sum(len(s.turns) for s in self.session_evaluations)

        all_turns = [t for s in self.session_evaluations for t in s.turns]
        if not all_turns:
            return

        n = len(all_turns)
        self.overall_context_recall = sum(t.context_recall for t in all_turns) / n
        self.overall_context_precision = sum(t.context_precision for t in all_turns) / n
        self.overall_temporal_recall = sum(t.temporal_recall for t in all_turns) / n
        self.overall_fact_coverage = sum(t.fact_coverage for t in all_turns) / n
        self.total_retrieval_iterations = sum(t.retrieval_iterations for t in all_turns)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "timestamp": self.timestamp,
            "group_id": self.group_id,
            "config": self.config,
            "statistics": {
                "total_sessions": self.total_sessions,
                "total_turns": self.total_turns,
                "total_retrieval_iterations": self.total_retrieval_iterations,
            },
            "overall_metrics": {
                "context_recall": self.overall_context_recall,
                "context_precision": self.overall_context_precision,
                "temporal_recall": self.overall_temporal_recall,
                "fact_coverage": self.overall_fact_coverage,
            },
            "session_evaluations": [
                {
                    "session_id": s.session_id,
                    "avg_metrics": {
                        "context_recall": s.avg_context_recall,
                        "context_precision": s.avg_context_precision,
                        "temporal_recall": s.avg_temporal_recall,
                        "fact_coverage": s.avg_fact_coverage,
                    },
                    "turns": [asdict(t) for t in s.turns],
                }
                for s in self.session_evaluations
            ],
        }


class RAGEvaluator:
    """
    Evaluator untuk RAG system.

    Workflow:
    1. Load dataset
    2. Ingest conversations to graph
    3. For each turn, retrieve context and evaluate
    4. Calculate metrics
    5. Generate report
    """

    def __init__(
        self,
        graph_client: TemporalGraphClient,
        config: Optional[EvaluationConfig] = None,
    ):
        self.client = graph_client
        self.config = config or get_config().evaluation
        self.ingester = EpisodeIngester(graph_client)
        self.retriever = RetrievalAgent(graph_client)

    async def evaluate_turn(
        self,
        turn_text: str,
        turn_id: int,
        session_id: int,
        speaker: str,
        ground_truth: Dict[str, Any],
    ) -> TurnEvaluation:
        """Evaluate a single turn"""

        # Skip turns without retrieval requirement
        if not ground_truth or not ground_truth.get("retrieval_required"):
            return TurnEvaluation(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                details={"skipped": True, "reason": "no_retrieval_required"},
            )

        # Perform retrieval
        result = await self.retriever.retrieve(turn_text)

        # Extract retrieved facts
        retrieved_facts = [f.fact for f in result.facts]
        retrieved_context = result.context

        # Get expected answer from ground truth (if available)
        expected_answer = ground_truth.get("expected_answer", "")

        # Calculate metrics (including LLM Judge for answer accuracy)
        metrics = await calculate_all_metrics(
            retrieved_facts=retrieved_facts,
            retrieved_context=retrieved_context,
            ground_truth=ground_truth,
            generated_answer=result.context,  # Use retrieved context as "answer"
            expected_answer=expected_answer,
            question_context=turn_text,
            use_llm_judge=self.config.use_llm_judge,
        )

        # Build turn evaluation
        turn_eval = TurnEvaluation(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            context_recall=metrics["context_recall"].score,
            context_precision=metrics["context_precision"].score,
            temporal_recall=metrics["temporal_recall"].score,
            fact_coverage=metrics["fact_coverage"].score,
            retrieved_count=len(result.facts),
            retrieval_iterations=result.iterations,
            query_type=result.query_type.value,
            details={k: v.details for k, v in metrics.items()},
        )

        # Add answer accuracy if available
        if "answer_accuracy" in metrics:
            turn_eval.details["answer_accuracy"] = metrics["answer_accuracy"].details
            turn_eval.details["answer_accuracy_score"] = metrics[
                "answer_accuracy"
            ].score

        return turn_eval

    async def evaluate_session(
        self,
        session: Dict[str, Any],
        ingest_first: bool = True,
        show_progress: bool = True,
    ) -> SessionEvaluation:
        """Evaluate a complete session"""
        session_id = session["session_id"]
        turns = session.get("turns", [])
        ground_truths = {gt["turn_id"]: gt for gt in session.get("ground_truths", [])}

        turn_evaluations = []

        iterator = (
            tqdm(enumerate(turns), total=len(turns), desc=f"Session {session_id}")
            if show_progress
            else enumerate(turns)
        )

        for idx, turn in iterator:
            # Ingest turn first (if enabled)
            if ingest_first:
                await self.ingester.ingest_turn(
                    self.ingester.parse_sessions({"sessions": [session]})[0].turns[idx]
                )

            # Evaluate
            gt = ground_truths.get(idx, {})
            turn_eval = await self.evaluate_turn(
                turn_text=turn["text"],
                turn_id=idx,
                session_id=session_id,
                speaker=turn["speaker"],
                ground_truth=gt,
            )
            turn_evaluations.append(turn_eval)

            # Small delay for rate limiting
            await asyncio.sleep(0.1)

        session_eval = SessionEvaluation(session_id=session_id, turns=turn_evaluations)
        session_eval.calculate_averages()

        return session_eval

    async def evaluate_dataset(
        self,
        dataset_path: str,
        limit_sessions: Optional[int] = None,
        limit_turns_per_session: Optional[int] = None,
        ingest_first: bool = True,
        show_progress: bool = True,
    ) -> EvaluationReport:
        """
        Run full evaluation on dataset.

        Args:
            dataset_path: Path to conversation_dataset.json
            limit_sessions: Limit number of sessions
            limit_turns_per_session: Limit turns per session
            ingest_first: Whether to ingest before evaluating
            show_progress: Show progress bars

        Returns:
            Complete EvaluationReport
        """
        # Load dataset
        data = self.ingester.load_dataset(dataset_path)
        sessions = data.get("sessions", [])

        if limit_sessions:
            sessions = sessions[:limit_sessions]

        logger.info(f"Starting evaluation on {len(sessions)} sessions...")

        session_evaluations = []

        for session in sessions:
            # Limit turns if specified
            if limit_turns_per_session:
                session = dict(session)
                session["turns"] = session["turns"][:limit_turns_per_session]
                session["ground_truths"] = [
                    gt
                    for gt in session.get("ground_truths", [])
                    if gt["turn_id"] < limit_turns_per_session
                ]

            session_eval = await self.evaluate_session(
                session, ingest_first=ingest_first, show_progress=show_progress
            )
            session_evaluations.append(session_eval)

        # Create report
        assert self.client.group_id is not None
        report = EvaluationReport(
            run_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_path=dataset_path,
            timestamp=datetime.now().isoformat(),
            group_id=self.client.group_id,
            config=asdict(self.config),
            session_evaluations=session_evaluations,
        )
        report.calculate_overall()

        return report

    def save_report(self, report: EvaluationReport, output_path: str):
        """Save evaluation report to JSON file"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to {output_path}")


async def run_evaluation(
    dataset_path: str,
    output_path: str = "output/evaluation_results/report.json",
    group_id: Optional[str] = None,
    limit_sessions: Optional[int] = None,
    limit_turns: Optional[int] = None,
) -> EvaluationReport:
    """
    Convenience function to run evaluation.

    Args:
        dataset_path: Path to dataset
        output_path: Path for output report
        group_id: Optional group ID
        limit_sessions: Limit sessions
        limit_turns: Limit turns per session

    Returns:
        EvaluationReport
    """
    client = TemporalGraphClient(group_id=group_id)

    try:
        await client.initialize()

        evaluator = RAGEvaluator(client)
        report = await evaluator.evaluate_dataset(
            dataset_path=dataset_path,
            limit_sessions=limit_sessions,
            limit_turns_per_session=limit_turns,
        )

        evaluator.save_report(report, output_path)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Sessions: {report.total_sessions}")
        print(f"Total Turns: {report.total_turns}")
        print("\nOverall Metrics:")
        print(f"  Context Recall:    {report.overall_context_recall:.4f}")
        print(f"  Context Precision: {report.overall_context_precision:.4f}")
        print(f"  Temporal Recall:   {report.overall_temporal_recall:.4f}")
        print(f"  Fact Coverage:     {report.overall_fact_coverage:.4f}")
        print("=" * 60)

        return report

    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--dataset",
        type=str,
        default="output/example_dataset/conversation_dataset.json",
    )
    parser.add_argument(
        "--output", type=str, default="output/evaluation_results/report.json"
    )
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--limit-turns", type=int, default=None)
    parser.add_argument("--group-id", type=str, default=None)

    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            output_path=args.output,
            group_id=args.group_id,
            limit_sessions=args.limit_sessions,
            limit_turns=args.limit_turns,
        )
    )
