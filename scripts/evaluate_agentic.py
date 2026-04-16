#!/usr/bin/env python
# scripts/evaluate_agentic.py
"""
Agentic RAG Evaluation Script
==============================

Evaluate Agentic RAG setups (Neo4j Knowledge Graph) using 100 evaluation queries.
Uses same initialization pattern as ingest_agentic.py for compatibility.

Usage:
    python scripts/evaluate_agentic.py --setup gemma --limit 5 --no-llm-judge  # Quick test
    python scripts/evaluate_agentic.py --setup gemma                           # Full evaluation
"""

import asyncio
import sys

print("=== STARTING evaluate_agentic.py ===", flush=True)
import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import List, Dict, Any, Optional  # noqa: E402  # Any used in session extraction
from dataclasses import dataclass, asdict  # noqa: E402

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("neo4j").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm  # noqa: E402
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
    retrieved_facts: List[str]
    retrieved_sessions: List[int]  # Session IDs extracted from retrieved facts
    retrieval_time_ms: float
    hit_rate: float
    reciprocal_rank: float
    context_length: int
    # New fields for detailed tracking
    timestamp: str | None = None  # ISO timestamp when query was evaluated
    iterations: int = 1  # Number of agent iterations (1 for vanilla/simple)
    sub_queries: List[str] | None = None  # Sub-queries used by agent
    retrieval_metadata: Dict[str, Any] | None = (
        None  # Additional metadata from retrieval
    )
    context_sufficiency: Optional[float] = None
    context_sufficiency_details: Optional[Dict[str, float]] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation results"""

    setup_name: str
    total_queries: int
    timestamp: str
    mean_hit_rate: float
    mean_mrr: float
    mean_retrieval_time_ms: float
    mean_context_sufficiency: Optional[float] = None
    total_cost: Optional[str] = None
    metrics_by_type: Dict[str, Dict[str, float]] | None = None
    metrics_by_difficulty: Dict[str, Dict[str, float]] | None = None


class AgenticEvaluator:
    """Evaluate Agentic RAG using same Graphiti init as ingest_agentic.py"""

    def __init__(
        self,
        setup_name: str,
        with_llm_judge: bool = True,
        judge_model: str = "gemini-2.5-pro",
        reuse_agentic_from: str | None = None,
    ):
        self.setup_name = setup_name
        self.with_llm_judge = with_llm_judge
        self.judge_model = judge_model
        self.reuse_agentic_from = reuse_agentic_from

        # Load cached agentic results if provided (for hybrid reuse)
        self._cached_agentic_results = {}
        if reuse_agentic_from and "hybrid" in setup_name:
            try:
                with open(reuse_agentic_from, "r") as f:
                    data = json.load(f)
                for r in data.get("results", []):
                    self._cached_agentic_results[r["query_id"]] = r["retrieved_facts"]
                print(
                    f"✅ Loaded {len(self._cached_agentic_results)} cached agentic results from {reuse_agentic_from}"
                )
            except Exception as e:
                logger.warning(f"Failed to load cached agentic results: {e}")

        self._graphiti = None
        self._tc = None
        self._setup = None
        self._group_id = None
        self._queries = None
        self._neo4j_driver = None  # legacy; Surreal uses _tc

    async def _extract_session_ids_from_facts(self, facts: List[str]) -> List[int]:
        """Map fact strings back to session ids via SurrealDB episode_name."""
        if not getattr(self, "_tc", None) or not facts:
            return []
        db = getattr(self._tc, "_db", None)
        if db is None:
            return []
        session_ids: List[int] = []
        gid = self._group_id
        try:
            for fact in facts[:15]:
                clean_fact = fact
                if clean_fact.startswith("[FACT] "):
                    clean_fact = clean_fact[7:]
                elif clean_fact.startswith("[DETAIL] "):
                    clean_fact = clean_fact[9:]
                needle = clean_fact[:120].replace('"', "")
                res = await db.query(
                    "SELECT episode_name FROM extracted_fact WHERE group_id = $gid "
                    "AND fact_text CONTAINS $needle LIMIT 1",
                    {"gid": gid, "needle": needle},
                )
                rows: List[Any] = []
                if isinstance(res, list) and res and isinstance(res[0], dict):
                    r0 = res[0].get("result")
                    rows = r0 if isinstance(r0, list) else ([r0] if r0 else [])
                if rows:
                    r = rows[0]
                    en = r.get("episode_name") if isinstance(r, dict) else None
                    if en:
                        m = re.search(r"session_(\d+)", str(en))
                        if m:
                            sid = int(m.group(1))
                            if sid not in session_ids:
                                session_ids.append(sid)
        except Exception as e:
            logger.debug("Error extracting session IDs: %s", e)
        return session_ids

    def _calculate_hit_rate(
        self, retrieved_sessions: List[int], relevant_sessions: List[int]
    ) -> float:
        """Check if any relevant session is in retrieved sessions"""
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

    async def initialize(self):
        """Initialize Graphiti client - copied from ingest_agentic.py"""
        from src.config.experiment_setups import (
            SETUP_1A_AGENTIC_GEMINI,
            # SETUP_2A_AGENTIC_GEMMA,  # Commented - Gemma configs disabled
        )
        from src.config.settings import get_config

        config = get_config()

        # Get setup
        if self.setup_name == "gemini":
            # High-detail Agentic Gemini setup
            self._setup = SETUP_1A_AGENTIC_GEMINI
        elif self.setup_name == "gemma":
            from src.config.experiment_setups import SETUP_2A_AGENTIC_GEMMA

            self._setup = SETUP_2A_AGENTIC_GEMMA
        elif self.setup_name == "gemini_hybrid":
            from src.config.experiment_setups import SETUP_1H_HYBRID_GEMINI

            self._setup = SETUP_1H_HYBRID_GEMINI
        elif self.setup_name == "gemma_hybrid":
            from src.config.experiment_setups import SETUP_2H_HYBRID_GEMMA

            self._setup = SETUP_2H_HYBRID_GEMMA
        elif self.setup_name == "vanilla_gemini":
            from src.config.experiment_setups import SETUP_1V_VANILLA_GEMINI

            self._setup = SETUP_1V_VANILLA_GEMINI
        elif self.setup_name == "vanilla_gemma":
            from src.config.experiment_setups import SETUP_2V_VANILLA_GEMMA

            self._setup = SETUP_2V_VANILLA_GEMMA
        else:
            raise ValueError(f"Unknown setup: {self.setup_name}")

        self._group_id = self._setup.storage.group_id

        # Load evaluation queries
        import json

        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._queries = data.get(
                "queries", data
            )  # Handle both dict with 'queries' key and direct list

        print("\nInitializing Agentic Evaluator...")
        print(f"  Setup: {self._setup.name}")
        print(f"  Queries: {len(self._queries)}")

        # Logic for Vanilla (Skip Neo4j/Graphiti)
        if self.setup_name in ("vanilla_gemini", "vanilla_gemma"):
            from src.rag.retrieval.vanilla_retriever import create_vanilla_retriever

            self._vanilla_retriever = await create_vanilla_retriever(self._setup)
            print("✅ Vanilla Retriever Initialized")
            return

        print(f"  Group ID: {self._group_id}")
        print(f"  Embedder: {self._setup.embedder.name}")

        self._neo4j_driver = None

        # NovitaLLMClient for RetrievalAgent sufficiency check (Gemma only)
        self._novita_llm_client = None
        if self.setup_name in ("gemma", "gemma_hybrid"):
            from openai import AsyncOpenAI

            class NovitaLLMClient:
                """Custom LLM client for Novita/Gemma sufficiency check with cost tracking"""

                def __init__(self, api_key: str, base_url: str, model: str):
                    self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                    self.model = model
                    self.config = type(
                        "Config",
                        (),
                        {"api_key": api_key, "base_url": base_url, "model": model},
                    )()

                async def generate_response(self, messages, **kwargs):
                    """Generate response using Novita API directly with cost tracking"""
                    openai_messages = []
                    input_chars = 0
                    for m in messages:
                        if hasattr(m, "content"):
                            openai_messages.append(
                                {"role": m.role, "content": m.content}
                            )
                            input_chars += len(str(m.content))
                        elif isinstance(m, dict):
                            openai_messages.append(m)
                            input_chars += len(str(m.get("content", "")))
                        else:
                            openai_messages.append({"role": "user", "content": str(m)})
                            input_chars += len(str(m))

                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=openai_messages,
                        max_tokens=kwargs.get("max_tokens", 500),
                        temperature=kwargs.get("temperature", 0.3),
                    )

                    # Cost tracking for Novita API
                    try:
                        output_content = response.choices[0].message.content
                        output_chars = len(output_content) if output_content else 0

                        # Estimate tokens (4 chars per token)
                        input_tokens = input_chars // 4
                        output_tokens = output_chars // 4

                        from src.utils.cost_tracker import get_cost_tracker

                        tracker = get_cost_tracker()
                        await tracker.track(input_tokens, output_tokens, self.model)
                    except Exception as e:
                        logger.debug(f"Novita cost tracking failed: {e}")

                    return {"content": response.choices[0].message.content}

            self._novita_llm_client = NovitaLLMClient(
                api_key=config.novita.api_key,
                base_url=config.novita.base_url,
                model="google/gemma-3-27b-it",
            )
            print("  Sufficiency LLM: google/gemma-3-27b-it (Novita direct)")

        model_name = (
            self._setup.llm_extraction.name
            if self._setup and self._setup.llm_extraction
            else "gemini-2.5-flash"
        )
        llm_client = type(
            "SuffGeminiCfg",
            (),
            {
                "config": type(
                    "C",
                    (),
                    {
                        "model": model_name,
                        "api_key": config.gemini.api_key,
                        "base_url": None,
                    },
                )()
            },
        )()
        print(f"  Sufficiency LLM (direct GenAI): {model_name}")

        from src.rag.graph_client import TemporalGraphClient

        self._tc = TemporalGraphClient(setup=self._setup)
        await self._tc.initialize()
        self._graphiti = self._tc

        # Initialize RetrievalAgent for non-hybrid setups (true agentic loop)
        if self.setup_name in ("gemini", "gemma"):
            from src.rag.retrieval.agent import RetrievalAgent
            from src.rag.graph_client import SearchResult
            from src.config.settings import RetrievalConfig

            # Create adapter to wrap Graphiti for RetrievalAgent compatibility
            class GraphitiClientAdapter:
                """Adapter to make Graphiti compatible with RetrievalAgent's expected interface"""

                def __init__(self, graphiti, group_id):
                    self.graphiti = graphiti
                    self.group_id = group_id

                async def search(self, query: str, num_results: int = 10):
                    """Search using Graphiti and return SearchResult objects"""
                    results = await self.graphiti.search(
                        query=query, group_ids=[self.group_id], num_results=num_results
                    )
                    return (
                        [
                            SearchResult(
                                fact=r.fact,
                                score=getattr(
                                    r, "score", 0.8
                                ),  # Default high score if missing
                                entity_name=getattr(r, "entity_name", None),
                                created_at=getattr(r, "created_at", None),
                                valid_at=getattr(r, "valid_at", None),
                            )
                            for r in results
                        ]
                        if results
                        else []
                    )

                async def search_with_temporal_filter(
                    self, query: str, before=None, after=None, num_results: int = 10
                ):
                    """Temporal search (fallback to regular search for now)"""
                    return await self.search(query, num_results)

                async def get_entity_facts(self, entity_name: str):
                    """Get facts related to an entity by searching for it"""
                    return await self.search(entity_name, num_results=5)

            # Create adapter and agent
            adapter = GraphitiClientAdapter(self._graphiti, self._group_id)
            retrieval_config = RetrievalConfig(
                max_iterations=5, num_results=5, similarity_threshold=0.3
            )
            # Use NovitaLLMClient for Gemma, TrackingGeminiClient for Gemini
            sufficiency_client = (
                self._novita_llm_client if self.setup_name == "gemma" else llm_client
            )
            self._retrieval_agent = RetrievalAgent(
                adapter, retrieval_config, llm_client=sufficiency_client
            )
            mode = "Novita" if self.setup_name == "gemma" else "Gemini"
            print(f"✅ RetrievalAgent initialized ({mode} LLM sufficiency, 5-15 facts)")

        print("✅ Initialized")

        if "hybrid" in self.setup_name:
            from src.rag.retrieval.hybrid_retriever import HybridRetriever
            from src.rag.retrieval.vanilla_retriever import create_vanilla_retriever

            # Determine setups
            if "gemini" in self.setup_name:
                from src.config.experiment_setups import (
                    SETUP_1H_HYBRID_GEMINI as HYBRID_SETUP,
                )
                from src.config.experiment_setups import (
                    SETUP_1V_VANILLA_GEMINI as VANILLA_SETUP,
                )
            else:
                # Gemma Hybrid
                from src.config.experiment_setups import (
                    SETUP_2H_HYBRID_GEMMA as HYBRID_SETUP,
                )
                from src.config.experiment_setups import (
                    SETUP_2V_VANILLA_GEMMA as VANILLA_SETUP,
                )

            # 1. Create RetrievalAgent-based Graph Client for multi-hop reasoning
            from src.rag.retrieval.agent import RetrievalAgent
            from src.rag.graph_client import SearchResult
            from src.config.settings import RetrievalConfig

            class _GraphitiClientAdapterHybrid:
                """Adapter to make Graphiti compatible with RetrievalAgent's expected interface"""

                def __init__(self, graphiti, group_id):
                    self.graphiti = graphiti
                    self.group_id = group_id

                async def search(self, query: str, num_results: int = 10):
                    """Search using Graphiti and return SearchResult objects"""
                    results = await self.graphiti.search(
                        query=query, group_ids=[self.group_id], num_results=num_results
                    )
                    return (
                        [
                            SearchResult(
                                fact=r.fact,
                                score=getattr(r, "score", 0.8),
                                entity_name=getattr(r, "entity_name", None),
                                created_at=getattr(r, "created_at", None),
                                valid_at=getattr(r, "valid_at", None),
                            )
                            for r in results
                        ]
                        if results
                        else []
                    )

                async def search_with_temporal_filter(
                    self, query: str, before=None, after=None, num_results: int = 10
                ):
                    return await self.search(query, num_results)

                async def get_entity_facts(self, entity_name: str):
                    return await self.search(entity_name, num_results=5)

            # Create agent with adapter and LLM for sufficiency check
            adapter = _GraphitiClientAdapterHybrid(self._graphiti, self._group_id)
            retrieval_config = RetrievalConfig(
                max_iterations=5, num_results=5, similarity_threshold=0.3
            )
            # Use NovitaLLMClient for Gemma Hybrid, TrackingGeminiClient for Gemini Hybrid
            sufficiency_client = (
                self._novita_llm_client if "gemma" in self.setup_name else llm_client
            )
            agent = RetrievalAgent(
                adapter, retrieval_config, llm_client=sufficiency_client
            )

            # Wrapper that uses RetrievalAgent for HybridRetriever compatibility
            class AgentGraphWrapper:
                """Wrapper that uses RetrievalAgent for multi-hop graph search"""

                def __init__(self, agent):
                    self.agent = agent

                async def search(self, query, num_results=10):
                    # Use agent's iterative retrieval
                    result = await self.agent.retrieve(query)
                    # Return only up to num_results facts
                    return result.facts[:num_results]

            self._graph_wrapper = AgentGraphWrapper(agent)
            mode = "Novita" if "gemma" in self.setup_name else "Gemini"
            print(
                f"✅ RetrievalAgent-based Graph Wrapper initialized ({mode} LLM sufficiency, 5-15 facts)"
            )

            # 2. Initialize Vanilla Retriever
            vanilla_retriever = await create_vanilla_retriever(VANILLA_SETUP)

            # 3. Create Hybrid Retriever
            self._hybrid_retriever = HybridRetriever(
                graph_client=self._graph_wrapper,
                vanilla_retriever=vanilla_retriever,
                setup=HYBRID_SETUP,
            )
            print(f"✅ Hybrid Retriever initialized ({self.setup_name})")

    async def evaluate(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Run evaluation on queries"""
        assert self._queries is not None
        queries = self._queries[:limit] if limit else self._queries
        results = []

        print(f"\nEvaluating {len(queries)} queries...")

        # Phase 1: Retrieval
        for q in tqdm(queries, desc=f"Retrieval ({self.setup_name})"):
            result = await self._evaluate_single(q)
            results.append(result)

        # Phase 2: LLM Judge (if enabled)
        if self.with_llm_judge:
            print(f"\nRunning LLM Judge ({self.judge_model})...")
            from src.evaluation.metrics import context_sufficiency_llm_judge

            for result in tqdm(results, desc=f"LLM Judge ({self.setup_name})"):
                # Use Top 15 facts for Judge to capture more granular details
                # Agentic facts are small, so 15 is reasonable for context window
                context_facts = result.retrieved_facts[:15]
                context = "\n".join(context_facts)

                try:
                    judge_result = await context_sufficiency_llm_judge(
                        query=result.query,
                        expected_answer=result.expected_answer,
                        retrieved_context=context,
                        judge_model=self.judge_model,
                    )
                    result.context_sufficiency = judge_result.score
                    result.context_sufficiency_details = judge_result.details
                except Exception as e:
                    logger.warning(f"LLM Judge failed for {result.query_id}: {e}")
                    result.context_sufficiency = None

                await asyncio.sleep(1)  # Rate limit for judge

        # Calculate summary
        summary = self._calculate_summary(results)

        # Add cost tracking
        from src.utils.cost_tracker import get_cost_tracker

        cost_summary = get_cost_tracker().get_summary()

        summary_dict = asdict(summary)
        summary_dict["total_cost"] = cost_summary

        return {"summary": summary_dict, "results": [asdict(r) for r in results]}

    async def _evaluate_single(self, query_data: Dict) -> QueryResult:
        """Evaluate a single query"""
        start_time = time.time()
        query_timestamp = datetime.now().isoformat()  # Capture timestamp

        query = query_data["query"]

        # Initialize tracking variables for new fields
        iterations = 1
        sub_queries = []
        retrieval_metadata = {}

        # Search strategy depends on setup
        if "hybrid" in self.setup_name and hasattr(self, "_hybrid_retriever"):
            query_id = query_data.get("id", "")

            # Check if we can reuse cached agentic results
            if query_id in self._cached_agentic_results:
                # REUSE: Use cached agentic facts + fresh vanilla chunks
                cached_facts = self._cached_agentic_results[query_id]
                print(
                    f"[REUSE] Using {len(cached_facts)} cached agentic facts for: {query[:30]}..."
                )

                # Get fresh vanilla chunks
                vanilla_result = await self._hybrid_retriever.vanilla.retrieve(query)
                vanilla_chunks = (
                    [r.text for r in vanilla_result.results[:5]]
                    if vanilla_result
                    else []
                )

                # Combine: agentic facts (prefixed) + vanilla chunks
                retrieved_facts = [f"[FACT] {f}" for f in cached_facts[:15]] + [
                    f"[DETAIL] {c}" for c in vanilla_chunks
                ]
                print(
                    f"[REUSE] Combined: {len(cached_facts)} agentic + {len(vanilla_chunks)} vanilla = {len(retrieved_facts)} total"
                )

                retrieval_metadata = {
                    "source": "hybrid_cached",
                    "agentic_count": len(cached_facts),
                    "vanilla_count": len(vanilla_chunks),
                }
            else:
                # FRESH: Run full hybrid retrieval (fallback if no cache)
                print(f"[DEBUG] Starting Hybrid Retrieval for: {query[:30]}...")
                try:
                    hybrid_results = await self._hybrid_retriever.retrieve(
                        query=query, limit=10
                    )
                    print(
                        f"[DEBUG] Hybrid Retrieval done. Count: {len(hybrid_results)}"
                    )
                except Exception as e:
                    print(f"[ERROR] Hybrid Retrieval failed: {e}")
                    hybrid_results = []

                # Format using the hybrid formatter to get nice context strings
                retrieved_facts = [r.content for r in hybrid_results]

                # Count sources
                graph_count = sum(1 for r in hybrid_results if r.source_type == "graph")
                vanilla_count = sum(
                    1 for r in hybrid_results if r.source_type == "vanilla"
                )
                retrieval_metadata = {
                    "source": "hybrid_fresh",
                    "graph_count": graph_count,
                    "vanilla_count": vanilla_count,
                }

        elif self.setup_name in ("vanilla_gemini", "vanilla_gemma"):
            # Vanilla Search
            print(f"[DEBUG] Starting Vanilla Retrieval for: {query[:30]}...")
            vanilla_result = await self._vanilla_retriever.retrieve(query)
            retrieved_facts = [r.text for r in vanilla_result.results]

            retrieval_metadata = {
                "source": "vanilla",
                "result_count": len(retrieved_facts),
            }
        else:
            # Agentic Mode with RetrievalAgent (iterative multi-hop)
            if hasattr(self, "_retrieval_agent"):
                agent_result = await self._retrieval_agent.retrieve(query)
                retrieved_facts = [f.fact for f in agent_result.facts]

                # Capture agent metadata
                iterations = agent_result.iterations
                sub_queries = agent_result.metadata.get("plan", {}).get(
                    "search_queries", []
                )
                retrieval_metadata = {
                    "source": "agentic",
                    "query_type": agent_result.query_type.value
                    if hasattr(agent_result.query_type, "value")
                    else str(agent_result.query_type),
                    "entities_found": agent_result.entities,
                    "confidence": agent_result.confidence,
                    "requires_multi_hop": agent_result.metadata.get("plan", {}).get(
                        "requires_multi_hop", False
                    ),
                }

                # Log iterations for debugging
                if agent_result.iterations > 1:
                    print(
                        f"[DEBUG] Agent used {agent_result.iterations} iterations for: {query[:30]}..."
                    )
            else:
                assert self._graphiti is not None
                assert self._group_id is not None
                search_results = await self._graphiti.search(
                    query=query, group_ids=[self._group_id], num_results=10
                )
                retrieved_facts = (
                    [r.fact for r in search_results] if search_results else []
                )
                retrieval_metadata = {"source": "surreal_direct"}

        elapsed_ms = (time.time() - start_time) * 1000

        # Extract session IDs from retrieved facts for session-based HR/MRR
        # This makes agentic metrics apple-to-apple comparable with vanilla
        relevant_sessions = query_data.get("relevant_sessions", [])
        retrieved_sessions = await self._extract_session_ids_from_facts(retrieved_facts)

        # Calculate metrics using session-based matching (same as vanilla)
        hit_rate = self._calculate_hit_rate(retrieved_sessions, relevant_sessions)
        reciprocal_rank = self._calculate_reciprocal_rank(
            retrieved_sessions, relevant_sessions
        )

        return QueryResult(
            query_id=query_data["id"],
            query=query,
            query_type=query_data.get("type", "unknown"),
            difficulty=query_data.get("difficulty", "unknown"),
            expected_answer=query_data["expected_answer"],
            relevant_sessions=relevant_sessions,
            retrieved_facts=retrieved_facts[:15],
            retrieved_sessions=retrieved_sessions[:10],  # Store for debugging/analysis
            retrieval_time_ms=elapsed_ms,
            hit_rate=hit_rate,
            reciprocal_rank=reciprocal_rank,
            context_length=sum(len(f) for f in retrieved_facts),
            # New fields
            timestamp=query_timestamp,
            iterations=iterations,
            sub_queries=sub_queries if sub_queries else None,
            retrieval_metadata=retrieval_metadata if retrieval_metadata else None,
        )

    def _calculate_summary(self, results: List[QueryResult]) -> EvaluationSummary:
        """Calculate aggregate metrics"""
        total = len(results)

        mean_hr = sum(r.hit_rate for r in results) / total if total > 0 else 0
        mean_mrr = sum(r.reciprocal_rank for r in results) / total if total > 0 else 0
        mean_time = (
            sum(r.retrieval_time_ms for r in results) / total if total > 0 else 0
        )

        cs_results = [
            r.context_sufficiency for r in results if r.context_sufficiency is not None
        ]
        mean_cs = sum(cs_results) / len(cs_results) if cs_results else None

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
            setup_name=f"Agentic {self.setup_name.capitalize()}",
            total_queries=total,
            timestamp=datetime.now().isoformat(),
            mean_hit_rate=mean_hr,
            mean_mrr=mean_mrr,
            mean_retrieval_time_ms=mean_time,
            mean_context_sufficiency=mean_cs,
            metrics_by_type=by_type,
            metrics_by_difficulty=by_diff,
        )

    async def close(self):
        """Cleanup"""
        tc = getattr(self, "_tc", None)
        if tc is not None:
            try:
                await tc.close()
            except Exception:
                pass


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Agentic RAG")
    parser.add_argument(
        "--setup",
        choices=[
            "vanilla_gemini",
            "vanilla_gemma",
            "gemini",
            "gemma",
            "gemini_hybrid",
            "gemma_hybrid",
        ],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default="gemini-2.5-pro")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Batch API for 50%% cost reduction (async, results within 24h)",
    )
    parser.add_argument(
        "--reuse-agentic-from",
        type=str,
        default=None,
        help="Path to agentic results JSON to reuse for hybrid eval (avoids re-running agentic retrieval)",
    )

    args = parser.parse_args()

    # Handle batch mode - redirect to batch_evaluator
    if args.batch:
        print("\n" + "=" * 60)
        print("🚀 BATCH MODE ACTIVATED (50% Cost Savings)")
        print("=" * 60)
        print("Redirecting to batch_evaluator.py...")
        print("Note: Results will be delivered within 24 hours.")
        print("\nTo use batch mode manually:")
        print("  python scripts/batch_evaluator.py --help")
        print("=" * 60)
        return

    with_llm_judge = not args.no_llm_judge

    print("\n" + "=" * 60)
    print("AGENTIC RAG EVALUATION")
    print("=" * 60)
    print(f"Setup: {args.setup}")
    print(f"Limit: {args.limit or 'All 100 queries'}")
    print(f"LLM Judge: {'Enabled' if with_llm_judge else 'Disabled'}")
    if args.reuse_agentic_from:
        print(f"Reusing agentic results from: {args.reuse_agentic_from}")

    evaluator = AgenticEvaluator(
        setup_name=args.setup,
        with_llm_judge=with_llm_judge,
        judge_model=args.judge_model,
        reuse_agentic_from=args.reuse_agentic_from,
    )

    try:
        await evaluator.initialize()
        results = await evaluator.evaluate(limit=args.limit)

        # Save results
        filename = f"agentic_{args.setup}_results.json"
        with open(OUTPUT_DIR / filename, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {OUTPUT_DIR / filename}")

        # Print summary
        s = results["summary"]
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\n{s['setup_name']}:")
        print(f"  Hit Rate: {s['mean_hit_rate']:.3f}")
        print(f"  MRR:      {s['mean_mrr']:.3f}")
        if s.get("mean_context_sufficiency") is not None:
            print(f"  Context Sufficiency: {s['mean_context_sufficiency']:.3f}")
        print(f"  Avg Time: {s['mean_retrieval_time_ms']:.1f} ms")

        from src.utils.cost_tracker import get_cost_tracker

        print(f"  Total Cost: {get_cost_tracker().get_summary()}")

        if s.get("metrics_by_difficulty"):
            print("\n  By Difficulty:")
            for diff, metrics in s["metrics_by_difficulty"].items():
                print(
                    f"    {diff}: HR={metrics['hit_rate']:.3f}, MRR={metrics['mrr']:.3f} (n={metrics['count']})"
                )

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)

    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())
