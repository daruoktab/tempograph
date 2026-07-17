# src/retrieval/agent.py
from __future__ import annotations

"""
Retrieval Agent
===============
Agentic iterative retrieval untuk query kompleks dengan temporal reasoning.
"""

import logging
import asyncio
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from ..surreal.fact_graph import SearchResult
from ..config.settings import get_config, RetrievalConfig

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the agent can handle"""

    FACTUAL = "factual"  # "Siapa rekan kerja Aisha?"
    TEMPORAL = "temporal"  # "Kapan Aisha mulai project skincare?"
    CAUSAL = "causal"  # "Kenapa Aisha stress di bulan Januari?"
    AGGREGATION = "aggregation"  # "Berapa kali Aisha ke kafe?"
    COMPARISON = "comparison"  # "Apa bedanya project A dan B?"
    UNKNOWN = "unknown"


@dataclass
class RetrievalPlan:
    """Plan for retrieval execution"""

    query_type: QueryType
    entities_to_find: List[str] = field(default_factory=list)
    temporal_filter: Optional[Dict[str, datetime]] = None
    requires_multi_hop: bool = False
    search_queries: List[str] = field(default_factory=list)


@dataclass
class RetrievalState:
    """State during retrieval iteration"""

    iteration: int = 0
    retrieved_facts: List[SearchResult] = field(default_factory=list)
    entities_found: List[str] = field(default_factory=list)
    is_sufficient: bool = False
    confidence: float = 0.0


@dataclass
class RetrievalResult:
    """Final retrieval result"""

    facts: List[SearchResult]
    context: str
    iterations: int
    query_type: QueryType
    entities: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalAgent:
    """
    Agent untuk retrieval dengan kemampuan:
    - Query classification
    - Iterative retrieval with LLM-based sufficiency check
    - Temporal filtering
    - Multi-hop reasoning

    Args:
        graph_client: Surreal fact-graph client (vector + entity traversal)
        config: Konfigurasi retrieval (min/max facts, iterations)
        llm_client: Optional LLM client untuk sufficiency evaluation
    """

    def __init__(
        self,
        graph_client,
        config: Optional[RetrievalConfig] = None,
        llm_client=None,  # Optional: for LLM-based sufficiency check
    ):
        self.client = graph_client
        self.config = config or get_config().retrieval
        self.llm_client = llm_client  # LLM for "brain" evaluation

    def _detect_temporal_filter(
        self, query: str, query_type: QueryType
    ) -> Optional[Dict[str, datetime]]:
        """Light ISO-date window when temporal cues or explicit dates appear."""
        if not self.config.enable_temporal_filter:
            return None
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        if not m:
            return None
        temporalish = query_type in (
            QueryType.TEMPORAL,
            QueryType.FACTUAL,
            QueryType.CAUSAL,
            QueryType.COMPARISON,
        )
        if not temporalish:
            return None
        try:
            day = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
            start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {"after": start, "before": end}
        except ValueError:
            return None

    # ==========================================================================
    # QUERY CLASSIFICATION
    # ==========================================================================

    def classify_query(self, query: str) -> QueryType:
        """
        Classify query type based on keywords and patterns.

        For production, this should use an LLM classifier.
        """
        query_lower = query.lower()

        # Temporal indicators
        temporal_keywords = [
            "kapan",
            "tanggal",
            "waktu",
            "sejak",
            "sampai",
            "selama",
            "sebelum",
            "sesudah",
            "setelah",
            "bulan",
            "tahun",
            "minggu",
            "kemarin",
            "besok",
            "lusa",
            "dulu",
            "baru-baru ini",
        ]
        if any(kw in query_lower for kw in temporal_keywords):
            return QueryType.TEMPORAL

        # Causal indicators
        causal_keywords = [
            "kenapa",
            "mengapa",
            "karena",
            "sebab",
            "akibat",
            "penyebab",
            "alasan",
            "sehingga",
            "dampak",
        ]
        if any(kw in query_lower for kw in causal_keywords):
            return QueryType.CAUSAL

        # Aggregation indicators
        agg_keywords = [
            "berapa kali",
            "berapa banyak",
            "total",
            "jumlah",
            "seberapa sering",
            "rata-rata",
            "semua",
            "seluruh",
        ]
        if any(kw in query_lower for kw in agg_keywords):
            return QueryType.AGGREGATION

        # Comparison indicators
        comp_keywords = [
            "beda",
            "perbedaan",
            "sama",
            "dibanding",
            "versus",
            "lebih",
            "kurang",
            "mirip",
        ]
        if any(kw in query_lower for kw in comp_keywords):
            return QueryType.COMPARISON

        # Default to factual
        return QueryType.FACTUAL

    # ==========================================================================
    # PLANNING
    # ==========================================================================

    def create_plan(self, query: str, query_type: QueryType) -> RetrievalPlan:
        """
        Create retrieval plan based on query analysis.

        TODO: Use LLM for more sophisticated planning.
        """
        plan = RetrievalPlan(
            query_type=query_type,
            search_queries=[query],  # Start with original query
        )

        # Extract potential entity names (simple heuristic)
        # In production, use NER
        words = query.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                plan.entities_to_find.append(word)

        # Set multi-hop requirement based on query type
        if query_type in [QueryType.CAUSAL, QueryType.COMPARISON]:
            plan.requires_multi_hop = True

        plan.temporal_filter = self._detect_temporal_filter(query, query_type)
        return plan

    # ==========================================================================
    # EXECUTION
    # ==========================================================================

    async def execute_search(
        self, query: str, state: RetrievalState, plan: RetrievalPlan
    ) -> List[SearchResult]:
        """Execute a single search iteration"""
        if self.config.enable_temporal_filter and plan.temporal_filter:
            results = await self.client.search(
                query=query,
                num_results=self.config.num_results,
                valid_before=plan.temporal_filter.get("before"),
                valid_after=plan.temporal_filter.get("after"),
            )
        else:
            results = await self.client.search(
                query=query, num_results=self.config.num_results
            )

        # RRF scores are not cosine; skip threshold when RRF metadata is present
        if self.config.similarity_threshold > 0 and results:
            if not any((r.metadata or {}).get("rrf_score") is not None for r in results):
                results = [
                    r for r in results if r.score >= self.config.similarity_threshold
                ]

        return results

    async def expand_search(
        self, state: RetrievalState, plan: RetrievalPlan
    ) -> List[SearchResult]:
        """
        Expand search by following entity relationships.
        Uses graph ``entity`` → ``fact_involves`` → ``extracted_fact`` when available
        (TemporalGraphClient), with legacy fallback on ``entity_names`` arrays.
        """
        additional_results: List[SearchResult] = []
        seen_e: set[str] = set()
        ordered: List[str] = []
        for entity in list(plan.entities_to_find) + list(state.entities_found):
            if not entity or entity in seen_e:
                continue
            seen_e.add(entity)
            ordered.append(entity)
            if len(ordered) >= 5:
                break

        for entity in ordered:
            entity_facts = await self.client.get_entity_facts(entity)
            additional_results.extend(entity_facts)

        return additional_results

    # ==========================================================================
    # EVALUATION (LLM-based or Heuristic fallback)
    # ==========================================================================

    async def _ask_llm_sufficiency(
        self, query: str, facts: List[SearchResult]
    ) -> Tuple[bool, str]:
        """
        Ask LLM if the retrieved facts are sufficient to answer the query.
        Uses direct API call (bypassing any graph SDK wrapper) for Gemma compatibility.

        Returns:
            (is_sufficient, missing_info_hint)
        """
        if not self.llm_client:
            return True, ""  # No LLM = always sufficient (fallback)

        facts_text = "\n".join([f"- {f.fact}" for f in facts[:15]])

        prompt = f"""Kamu adalah evaluator untuk sistem RAG.

Pertanyaan user: "{query}"

Fakta yang sudah ditemukan:
{facts_text}

Apakah fakta-fakta di atas CUKUP untuk menjawab pertanyaan user secara lengkap?

Jawab dengan format:
CUKUP: [YA/TIDAK]
ALASAN: [penjelasan singkat]
INFO_KURANG: [jika TIDAK, sebutkan informasi apa yang masih kurang]"""

        try:
            # Try direct call with custom handling
            response = await self._call_llm_direct(prompt)

            if not response:
                logger.warning("Empty response from LLM, defaulting to sufficient")
                return True, ""

            is_sufficient = (
                "CUKUP: YA" in response.upper() or "CUKUP:YA" in response.upper()
            )

            # Extract missing info hint for next iteration
            missing_info = ""
            if "INFO_KURANG:" in response.upper():
                parts = response.split("INFO_KURANG:")[-1]
                missing_info = parts.strip().split("\n")[0]

            logger.info(
                f"LLM Sufficiency: {'CUKUP' if is_sufficient else 'KURANG'} - {missing_info[:50] if missing_info else 'N/A'}"
            )
            return is_sufficient, missing_info

        except Exception as e:
            logger.warning(f"LLM sufficiency check failed: {e}")
            return True, ""  # Fallback to sufficient on error

    async def _call_llm_direct(self, prompt: str) -> str:
        """
        Direct LLM call that works with both Gemini and Gemma.
        Calls the provider API directly to avoid SDK instruction quirks.
        """
        # Check if llm_client has config with model info
        client_cfg = getattr(self.llm_client, "config", None)
        if client_cfg is not None:
            model = getattr(client_cfg, "model", "") or ""
            base_url = getattr(client_cfg, "base_url", None)
            api_key = getattr(client_cfg, "api_key", "") or ""

            # If using Novita/Gemma, call directly via OpenAI-compatible API
            if base_url and "novita" in base_url.lower():
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3,
                )
                raw = response.choices[0].message.content
                return raw if raw is not None else ""

            # Gemini: Google GenAI SDK directly
            elif "gemini" in model.lower():
                from google import genai
                from google.genai import types as genai_types

                from ..config.settings import get_config

                key = api_key or get_config().gemini.api_key
                client = genai.Client(api_key=key)
                m = model or "gemini-2.5-flash"
                loop = asyncio.get_running_loop()

                def _call():
                    return client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=0.3, max_output_tokens=500
                        ),
                    )

                resp = await loop.run_in_executor(None, _call)
                return (getattr(resp, "text", None) or "").strip()

        # Fallback: try generate_response if present (legacy clients)
        try:
            if hasattr(self.llm_client, "generate_response"):
                response = await self.llm_client.generate_response(  # type: ignore[possibly-missing-attribute]
                    messages=[{"role": "user", "content": prompt}]
                )
                if isinstance(response, dict):
                    return str(response.get("content", ""))
                return str(response)
        except Exception as e:
            logger.debug(f"Fallback LLM call failed: {e}")
        return ""

    async def evaluate_sufficiency(
        self, query: str, state: RetrievalState, plan: RetrievalPlan
    ) -> Tuple[bool, float, str]:
        """
        Evaluate if retrieved context is sufficient.

        Returns (is_sufficient, confidence, missing_info_hint)
        """
        num_facts = len(state.retrieved_facts)

        # Hard limits
        if num_facts >= self.config.max_agent_facts:
            logger.info(f"Hit max_agent_facts limit ({self.config.max_agent_facts})")
            return True, 1.0, ""

        if state.iteration >= self.config.max_iterations:
            logger.info(f"Hit max_iterations limit ({self.config.max_iterations})")
            return True, 0.8, ""

        # Must have minimum facts before even checking
        if num_facts < self.config.min_agent_facts:
            return False, 0.0, "Need more facts"

        # Use LLM if available, otherwise heuristic
        if self.llm_client:
            is_sufficient, missing_info = await self._ask_llm_sufficiency(
                query, state.retrieved_facts
            )
            confidence = 0.9 if is_sufficient else 0.5
            return is_sufficient, confidence, missing_info
        else:
            # Fallback heuristic
            avg_score = sum(f.score for f in state.retrieved_facts) / num_facts
            use_rrf_scores = any(
                (f.metadata or {}).get("rrf_score") is not None for f in state.retrieved_facts
            )
            thr = 0.12 if use_rrf_scores else 0.7
            high_quality = sum(1 for f in state.retrieved_facts if f.score > thr)

            is_sufficient = high_quality >= 5 or (num_facts >= 10 and avg_score > (0.08 if use_rrf_scores else 0.6))
            confidence = min(1.0, (high_quality / 5) * max(avg_score, 0.05))

            return is_sufficient, confidence, ""

    # ==========================================================================
    # MAIN AGENT LOOP
    # ==========================================================================

    async def retrieve(self, query: str) -> RetrievalResult:
        """
        Main retrieval method with agentic loop.

        1. Classify query
        2. Create plan
        3. Execute iteratively until LLM says sufficient OR hit limits
        4. Return results (5-15 facts)
        """
        # Step 1: Classify
        query_type = self.classify_query(query)
        logger.info(f"Query classified as: {query_type.value}")

        # Step 2: Plan
        plan = self.create_plan(query, query_type)
        logger.debug(f"Created plan: {plan}")

        # Step 3: Iterative execution
        state = RetrievalState()
        missing_info_hint = ""
        for e in plan.entities_to_find:
            if e and e not in state.entities_found:
                state.entities_found.append(e)

        while state.iteration < self.config.max_iterations:
            state.iteration += 1
            logger.debug(f"Iteration {state.iteration}")

            # Track existing facts for deduplication
            existing_facts = {f.fact for f in state.retrieved_facts}

            # Build search queries for this iteration
            search_queries = plan.search_queries.copy()
            if missing_info_hint and state.iteration > 1:
                # Add refined query based on what LLM said was missing
                search_queries.append(missing_info_hint)
                logger.info(f"Added refined query: {missing_info_hint[:50]}...")

            # Execute search for each query
            for search_query in search_queries:
                results = await self.execute_search(search_query, state, plan)

                # Deduplicate and add results
                for r in results:
                    if (
                        r.fact not in existing_facts
                        and len(state.retrieved_facts) < self.config.max_agent_facts
                    ):
                        state.retrieved_facts.append(r)
                        existing_facts.add(r.fact)

                        # Track entities
                        if r.entity_name and r.entity_name not in state.entities_found:
                            state.entities_found.append(r.entity_name)

            # Graph expansion: multi-hop plans, or any query with heuristic entity candidates
            if (
                (plan.requires_multi_hop or bool(plan.entities_to_find))
                and (state.entities_found or plan.entities_to_find)
                and len(state.retrieved_facts) < self.config.max_agent_facts
            ):
                expansion = await self.expand_search(state, plan)
                for r in expansion:
                    if (
                        r.fact not in existing_facts
                        and len(state.retrieved_facts) < self.config.max_agent_facts
                    ):
                        state.retrieved_facts.append(r)

            # Evaluate sufficiency (async - may call LLM)
            (
                state.is_sufficient,
                state.confidence,
                missing_info_hint,
            ) = await self.evaluate_sufficiency(query, state, plan)

            if state.is_sufficient:
                logger.info(
                    f"Sufficient context found at iteration {state.iteration} with {len(state.retrieved_facts)} facts"
                )
                break

            logger.info(
                f"Iteration {state.iteration}: Not sufficient, need: {missing_info_hint[:50]}..."
            )

        # Step 4: Build result (cap at MAX_FACTS)
        final_facts = state.retrieved_facts[: self.config.max_agent_facts]
        context = "\n".join([f.fact for f in final_facts])

        return RetrievalResult(
            facts=final_facts,
            context=context,
            iterations=state.iteration,
            query_type=query_type,
            entities=state.entities_found,
            confidence=state.confidence,
            metadata={
                "plan": {
                    "search_queries": plan.search_queries,
                    "requires_multi_hop": plan.requires_multi_hop,
                    "temporal_filter": plan.temporal_filter,
                }
            },
        )

    async def retrieve_for_turn(
        self, turn_text: str, speaker: str, session_date: Optional[datetime] = None
    ) -> RetrievalResult:
        """
        Retrieve context for a conversation turn.

        For bot turns: retrieve relevant history to inform response
        For user turns: retrieve to understand references
        """
        # Use turn text as query
        result = await self.retrieve(turn_text)

        # Add metadata
        result.metadata["speaker"] = speaker
        if session_date:
            result.metadata["session_date"] = session_date.isoformat()

        return result


async def test_retrieval():
    """Test retrieval agent"""
    from ..surreal.fact_graph import TemporalGraphClient
    client = TemporalGraphClient()

    try:
        await client.initialize()

        agent = RetrievalAgent(client)

        # Test queries
        test_queries = [
            "Siapa rekan kerja Aisha?",
            "Kapan Aisha mulai project skincare?",
            "Kenapa Aisha stress?",
            "Berapa kali Aisha ke kafe?",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            result = await agent.retrieve(query)
            print(f"  Type: {result.query_type.value}")
            print(f"  Facts found: {len(result.facts)}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Confidence: {result.confidence:.2f}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_retrieval())
