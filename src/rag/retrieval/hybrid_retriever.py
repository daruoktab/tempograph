import asyncio
import json
import logging
from typing import List, Dict, Any, Protocol
from dataclasses import dataclass

from ...config.experiment_setups import ExperimentSetup
from ..surreal.fact_graph import SearchResult
from .vanilla_retriever import VanillaRetriever

logger = logging.getLogger(__name__)


class GraphSearchClient(Protocol):
    """Anything ``HybridRetriever`` calls for graph-side search (DB client or agent wrapper)."""

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]: ...


@dataclass
class HybridSearchResult:
    """Unified result format for hybrid retrieval"""

    content: str
    source_type: str  # "graph" or "vanilla"
    score: float
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    Combines Graph Retrieval (High-level, Entity-linked)
    with Vanilla Retrieval (Specific Details, Raw Text).
    """

    def __init__(
        self,
        graph_client: GraphSearchClient,
        vanilla_retriever: VanillaRetriever,
        setup: ExperimentSetup,
    ):
        self.graph = graph_client
        self.vanilla = vanilla_retriever
        self.setup = setup

    async def initialize(self):
        """Initialize both sub-retrievers"""
        # Graph client usually already initialized by caller
        # Vanilla is ALREADY initialized by create_vanilla_retriever()
        # DO NOT call self.vanilla.initialize() again - it corrupts the embedder!
        logger.info("Hybrid Retriever initialized")

    async def retrieve(self, query: str, limit: int = 10) -> List[HybridSearchResult]:
        """
        Agent-led Hybrid Retrieval.

        Strategy:
        - Graph component: Agent determines how many facts needed (5-15) based on query complexity
        - Vanilla component: Always add top 5 chunks as detail supplement

        Total result: 10-20 items (fair comparison with Vanilla's 10)
        """
        VANILLA_SUPPLEMENT = 5  # Fixed: always add 5 vanilla chunks for detail

        async def _graph_branch() -> List[SearchResult]:
            return await self.graph.search(query, num_results=15)

        async def _vanilla_branch():
            return await self.vanilla.retrieve(query)

        try:
            results_graph: List[SearchResult] = []
            result_vanilla = None
            vanilla_count = 0

            graph_task = asyncio.create_task(_graph_branch())
            vanilla_task = asyncio.create_task(_vanilla_branch())
            graph_out, vanilla_out = await asyncio.gather(
                graph_task, vanilla_task, return_exceptions=True
            )

            if isinstance(graph_out, Exception):
                logger.error(f"Graph search failed: {graph_out}")
                results_graph = []
            else:
                results_graph = graph_out
                graph_count = len(results_graph) if results_graph else 0
                logger.info(f"Graph (Agent-led): got {graph_count} facts")

            if isinstance(vanilla_out, Exception):
                logger.error(f"Vanilla search failed: {vanilla_out}")
                result_vanilla = None
            else:
                result_vanilla = vanilla_out
                vanilla_count = len(result_vanilla.results) if result_vanilla else 0
                logger.info(
                    f"Vanilla: got={vanilla_count}, will use={min(vanilla_count, VANILLA_SUPPLEMENT)}"
                )

            combined_results = []

            # 3. Add ALL Graph Results (Agent already decided sufficiency: 5-15)
            for r in results_graph:
                combined_results.append(
                    HybridSearchResult(
                        content=f"[FACT] {r.fact}",
                        source_type="graph",
                        score=r.score,
                        metadata={
                            "entity": r.entity_name,
                            "valid_at": r.valid_at,
                            "source_description": r.source_description,
                            "entity_names": (r.metadata or {}).get("entity_names"),
                        },
                    )
                )

            # 4. Add FIXED 5 Vanilla Results (detail supplement)
            if result_vanilla and result_vanilla.results:
                for r in result_vanilla.results[:VANILLA_SUPPLEMENT]:
                    combined_results.append(
                        HybridSearchResult(
                            content=f"[DETAIL] {r.text}",
                            source_type="vanilla",
                            score=r.score,
                            metadata=r.metadata,
                        )
                    )

            logger.info(
                f"Hybrid retrieval finished. Graph={len(results_graph)}, Vanilla={min(vanilla_count, VANILLA_SUPPLEMENT) if result_vanilla else 0}, Total={len(combined_results)}"
            )
            return combined_results

        except Exception as e:
            logger.error(f"Critical error in HybridRetriever.retrieve: {e}")
            return []

    def format_context(self, results: List[HybridSearchResult]) -> str:
        """Structured context for LLM (facts JSON + raw session passages)."""
        lines: List[str] = ["=== RETRIEVED CONTEXT ==="]

        graph_facts = [r for r in results if r.source_type == "graph"]
        vanilla_docs = [r for r in results if r.source_type == "vanilla"]

        if graph_facts:
            lines.append("\n<FACTS>")
            fact_rows = []
            for r in graph_facts:
                text = r.content.replace("[FACT] ", "", 1)
                md = r.metadata or {}
                fact_rows.append(
                    {
                        "fact": text,
                        "valid_at": md.get("valid_at"),
                        "source_description": md.get("source_description"),
                        "entity": md.get("entity"),
                        "entity_names": md.get("entity_names"),
                    }
                )
            lines.append(json.dumps(fact_rows, default=str, ensure_ascii=False, indent=2))
            lines.append("</FACTS>")

        if vanilla_docs:
            lines.append("\n<EPISODE_PASSAGES>")
            for r in vanilla_docs:
                lines.append(r.content.replace("[DETAIL] ", "", 1))
            lines.append("</EPISODE_PASSAGES>")

        return "\n".join(lines)
