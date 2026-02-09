#!/usr/bin/env python
# scripts/evaluate_vanilla.py
"""
Vanilla RAG Evaluation Script
==============================

Evaluate Vanilla RAG setups using 100 evaluation queries.

Metrics:
- Hit Rate: Apakah context yang relevan ditemukan?
- MRR (Mean Reciprocal Rank): Di posisi berapa hasil relevan pertama?
- Context Sufficiency (LLM Judge): Apakah context cukup untuk menjawab query?

Output: JSON hasil evaluasi untuk comparison dengan Agentic RAG.

Usage:
    python scripts/evaluate_vanilla.py --setup gemini
    python scripts/evaluate_vanilla.py --setup gemma
    python scripts/evaluate_vanilla.py --setup all
    python scripts/evaluate_vanilla.py --setup gemini --no-llm-judge  # Skip LLM judge
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.evaluation.metrics import context_sufficiency_llm_judge, MetricResult

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
QUERIES_PATH = Path("output/final_dataset_v1/evaluation_queries_100.json")
OUTPUT_DIR = Path("output/evaluation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class QueryResult:
    """Result for a single query evaluation"""

    query_id: str
    query: str
    query_type: str
    difficulty: str
    expected_answer: str
    relevant_sessions: List[int]

    # Retrieved info
    retrieved_sessions: List[int]
    retrieval_time_ms: float

    # Metrics
    hit_rate: float  # 1.0 jika ada hit, 0.0 jika tidak ada
    reciprocal_rank: float  # 1/rank dari hit pertama
    context_length: int

    # New fields for detailed tracking (consistent with agentic)
    timestamp: str | None = None  # ISO timestamp when query was evaluated
    iterations: int = 1  # Always 1 for vanilla (no iterative retrieval)
    sub_queries: List[str] | None = None  # None for vanilla (no sub-queries)
    retrieval_metadata: Dict[str, Any] | None = None  # Additional metadata

    context_sufficiency: Optional[float] = None  # LLM judge score (0-1)
    context_sufficiency_details: Dict[str, Any] | None = None

    # Extra info
    top_k_texts: List[str] | None = None  # Top-K retrieved texts (preview)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results"""

    setup_name: str
    total_queries: int
    timestamp: str

    # Aggregate metrics
    mean_hit_rate: float
    mean_mrr: float
    mean_retrieval_time_ms: float
    mean_context_sufficiency: Optional[float] = None  # LLM judge average

    # By type breakdown
    metrics_by_type: Dict[str, Dict[str, float]] | None = None
    metrics_by_difficulty: Dict[str, Dict[str, float]] | None = None


class VanillaEvaluator:
    """Evaluate Vanilla RAG retrieval quality"""

    def __init__(
        self,
        setup_name: str,
        with_llm_judge: bool = True,
        judge_model: str = "gemini-2.5-pro",
        with_rerank: bool = True,
    ):
        self.setup_name = setup_name
        self.with_llm_judge = with_llm_judge
        self.judge_model = judge_model
        self.with_rerank = with_rerank
        self._retriever = None
        self._setup = None

    async def initialize(self):
        """Initialize retriever"""
        from src.rag.retrieval.vanilla_retriever import (
            create_vanilla_retriever,
            VanillaRetriever,
        )
        from src.config.experiment_setups import (
            SETUP_1V_VANILLA_GEMINI,
            SETUP_2V_VANILLA_GEMMA,
            RetrievalSettings,
        )
        from src.rag.vectordb import get_chroma_client
        from src.embedders import create_embedder, EmbedderType
        from src.config.settings import get_config

        if self.setup_name == "gemini":
            self._setup = SETUP_1V_VANILLA_GEMINI
        else:
            self._setup = SETUP_2V_VANILLA_GEMMA

        logger.info(f"Initializing {self._setup.name}...")

        # If no rerank for Gemini, override retrieval settings to direct top-10
        if self.setup_name == "gemini" and not self.with_rerank:
            logger.info("  Reranking DISABLED - using direct top-10 retrieval")
            # Create custom retriever with modified settings
            config = get_config()

            # Get ChromaDB client
            assert self._setup.storage.collection_name is not None
            chroma_db = get_chroma_client(
                collection_name=self._setup.storage.collection_name,
                persist_directory=self._setup.storage.persist_directory,
            )

            # Create embedder
            embedder = create_embedder(
                embedder_type=EmbedderType.GEMINI,
                model_name=self._setup.embedder.name,
                gemini_api_key=config.gemini.api_key,
            )
            await embedder.initialize()
            await chroma_db.initialize(embedder=embedder)

            # Create retriever with direct retrieval settings (no rerank)
            direct_settings = RetrievalSettings(
                embedding_top_k=10, rerank_top_k=10, similarity_threshold=0.0
            )
            self._retriever = VanillaRetriever(
                chroma_db, settings=direct_settings, setup=self._setup
            )
            await self._retriever.initialize(embedder=embedder)
        else:
            # Normal initialization with setup's retrieval settings
            self._retriever = await create_vanilla_retriever(self._setup)

        rerank_status = (
            "enabled"
            if (self.setup_name != "gemini" or self.with_rerank)
            else "disabled"
        )
        logger.info(f"✅ Retriever initialized (rerank: {rerank_status})")

    def _load_queries(self) -> List[Dict[str, Any]]:
        """Load evaluation queries"""
        with open(QUERIES_PATH) as f:
            data = json.load(f)
        return data["queries"]

    def _calculate_hit_rate(
        self, retrieved_sessions: List[int], relevant_sessions: List[int]
    ) -> float:
        """Check if any relevant session is retrieved"""
        for sess in relevant_sessions:
            if sess in retrieved_sessions:
                return 1.0
        return 0.0

    def _calculate_reciprocal_rank(
        self, retrieved_sessions: List[int], relevant_sessions: List[int]
    ) -> float:
        """Calculate reciprocal rank of first relevant hit"""
        for i, sess in enumerate(retrieved_sessions):
            if sess in relevant_sessions:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on all queries"""
        queries = self._load_queries()
        logger.info(f"Loaded {len(queries)} queries")

        results: List[QueryResult] = []

        for q in tqdm(queries, desc=f"Evaluating ({self.setup_name})"):
            import time

            start = time.perf_counter()

            # Retrieve
            result = await self._retriever.retrieve(q["query"])  # type: ignore[possibly-missing-attribute]

            retrieval_time_ms = (time.perf_counter() - start) * 1000

            # Extract session IDs from results
            retrieved_sessions = []
            for r in result.results:
                # Session ID is in metadata
                if r.metadata and "session_id" in r.metadata:
                    retrieved_sessions.append(r.metadata["session_id"])

            # Calculate metrics
            relevant_sessions = q["relevant_sessions"]
            hit_rate = self._calculate_hit_rate(retrieved_sessions, relevant_sessions)
            rr = self._calculate_reciprocal_rank(retrieved_sessions, relevant_sessions)

            # Store result with timestamp
            query_timestamp = datetime.now().isoformat()
            qr = QueryResult(
                query_id=q["id"],
                query=q["query"],
                query_type=q["type"],
                difficulty=q["difficulty"],
                expected_answer=q["expected_answer"],
                relevant_sessions=relevant_sessions,
                retrieved_sessions=retrieved_sessions[:10],
                retrieval_time_ms=retrieval_time_ms,
                hit_rate=hit_rate,
                reciprocal_rank=rr,
                context_length=len(result.context),
                # New fields
                timestamp=query_timestamp,
                iterations=1,  # Always 1 for vanilla
                sub_queries=None,  # No sub-queries for vanilla
                retrieval_metadata={
                    "source": "vanilla",
                    "result_count": len(result.results),
                    "setup": self.setup_name,
                },
                top_k_texts=[
                    r.text[:200] for r in result.results[:5]
                ],  # Store more text for judge
            )
            results.append(qr)

            # Small delay for rate limiting
            await asyncio.sleep(0.05)

        # LLM Judge phase (if enabled)
        if self.with_llm_judge:
            logger.info(f"Running LLM Judge ({self.judge_model})...")
            for qr in tqdm(results, desc=f"LLM Judge ({self.setup_name})"):
                try:
                    # Build context from top_k_texts
                    context = "\n\n".join(qr.top_k_texts) if qr.top_k_texts else ""

                    judge_result: MetricResult = await context_sufficiency_llm_judge(
                        query=qr.query,
                        retrieved_context=context,
                        expected_answer=qr.expected_answer,
                        judge_model=self.judge_model,
                    )

                    qr.context_sufficiency = judge_result.score
                    qr.context_sufficiency_details = judge_result.details

                except Exception as e:
                    logger.warning(f"Judge failed for {qr.query_id}: {e}")
                    qr.context_sufficiency = 0.0
                    qr.context_sufficiency_details = {"error": str(e)[:100]}

                # Rate limit for LLM judge
                await asyncio.sleep(0.5)

        # Calculate summary
        summary = self._calculate_summary(results)

        return {"summary": asdict(summary), "results": [asdict(r) for r in results]}

    def _calculate_summary(self, results: List[QueryResult]) -> EvaluationSummary:
        """Calculate aggregate metrics"""
        total = len(results)

        # Overall metrics
        mean_hr = sum(r.hit_rate for r in results) / total
        mean_mrr = sum(r.reciprocal_rank for r in results) / total
        mean_time = sum(r.retrieval_time_ms for r in results) / total

        # Context sufficiency (if available)
        cs_results = [
            r
            for r in results
            if r.context_sufficiency is not None and r.context_sufficiency > 0
        ]
        mean_cs = (
            sum(r.context_sufficiency for r in cs_results) / len(cs_results)  # type: ignore[no-matching-overload]
            if cs_results
            else None
        )

        # By type
        by_type = {}
        for qtype in [
            "factual_recall",
            "inference",
            "multi_hop",
            "counterfactual",
            "temporal_reasoning",
        ]:
            type_results = [r for r in results if r.query_type == qtype]
            if type_results:
                by_type[qtype] = {
                    "count": len(type_results),
                    "hit_rate": sum(r.hit_rate for r in type_results)
                    / len(type_results),
                    "mrr": sum(r.reciprocal_rank for r in type_results)
                    / len(type_results),
                }

        # By difficulty
        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            diff_results = [r for r in results if r.difficulty == diff]
            if diff_results:
                by_diff[diff] = {
                    "count": len(diff_results),
                    "hit_rate": sum(r.hit_rate for r in diff_results)
                    / len(diff_results),
                    "mrr": sum(r.reciprocal_rank for r in diff_results)
                    / len(diff_results),
                }

        return EvaluationSummary(
            setup_name=self._setup.name,  # type: ignore[possibly-missing-attribute]
            total_queries=total,
            timestamp=datetime.now().isoformat(),
            mean_hit_rate=mean_hr,
            mean_mrr=mean_mrr,
            mean_retrieval_time_ms=mean_time,
            mean_context_sufficiency=mean_cs,
            metrics_by_type=by_type,
            metrics_by_difficulty=by_diff,
        )


async def evaluate_setup(
    setup_name: str,
    with_llm_judge: bool = True,
    judge_model: str = "gemini-2.5-pro",
    with_rerank: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single setup"""
    evaluator = VanillaEvaluator(
        setup_name,
        with_llm_judge=with_llm_judge,
        judge_model=judge_model,
        with_rerank=with_rerank,
    )
    await evaluator.initialize()
    return await evaluator.evaluate()


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Vanilla RAG")
    parser.add_argument(
        "--setup",
        choices=["gemini", "gemma", "all"],
        required=True,
        help="Which setup to evaluate",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (faster, but no context sufficiency score)",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-2.5-pro",
        help="Model to use as judge (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking for Gemini (use direct top-10 like Gemma)",
    )

    args = parser.parse_args()
    with_llm_judge = not args.no_llm_judge
    with_rerank = not args.no_rerank

    print("\n" + "=" * 60)
    print("VANILLA RAG EVALUATION")
    print("=" * 60)
    print(
        f"LLM Judge: {'Enabled (' + args.judge_model + ')' if with_llm_judge else 'Disabled'}"
    )
    print(f"Reranking: {'Enabled' if with_rerank else 'Disabled (Gemini only)'}")
    print()

    results = {}

    if args.setup in ["gemini", "all"]:
        print("\n--- Evaluating Setup 1V: Vanilla Gemini ---")
        results["vanilla_gemini"] = await evaluate_setup(
            "gemini", with_llm_judge, args.judge_model, with_rerank
        )

        # Save individual result - add suffix if no rerank
        suffix = "_norerank" if not with_rerank else ""
        filename = f"vanilla_gemini{suffix}_results.json"
        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(results["vanilla_gemini"], f, indent=2)
        logger.info(f"Saved: {OUTPUT_DIR / filename}")

    if args.setup in ["gemma", "all"]:
        print("\n--- Evaluating Setup 2V: Vanilla Gemma ---")
        results["vanilla_gemma"] = await evaluate_setup(
            "gemma", with_llm_judge, args.judge_model, with_rerank
        )

        # Save individual result
        with open(OUTPUT_DIR / "vanilla_gemma_results.json", "w") as f:
            json.dump(results["vanilla_gemma"], f, indent=2)
        logger.info(f"Saved: {OUTPUT_DIR / 'vanilla_gemma_results.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for name, data in results.items():
        s = data["summary"]
        print(f"\n{name}:")
        print(f"  Hit Rate: {s['mean_hit_rate']:.3f}")
        print(f"  MRR:      {s['mean_mrr']:.3f}")
        if s.get("mean_context_sufficiency") is not None:
            print(f"  Context Sufficiency: {s['mean_context_sufficiency']:.3f}")
        print(f"  Avg Time: {s['mean_retrieval_time_ms']:.1f} ms")

        print("\n  By Difficulty:")
        for diff, metrics in s["metrics_by_difficulty"].items():
            print(
                f"    {diff}: HR={metrics['hit_rate']:.3f}, MRR={metrics['mrr']:.3f} (n={metrics['count']})"
            )

        print("\n  By Type:")
        for qtype, metrics in s["metrics_by_type"].items():
            print(
                f"    {qtype}: HR={metrics['hit_rate']:.3f}, MRR={metrics['mrr']:.3f} (n={metrics['count']})"
            )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
