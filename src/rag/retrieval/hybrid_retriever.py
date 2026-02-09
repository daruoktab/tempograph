
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..experiment_setups import ExperimentSetup
from ..graph_client import TemporalGraphClient, SearchResult
from .vanilla_retriever import VanillaRetriever, VanillaRetrievalResult

logger = logging.getLogger(__name__)

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
        graph_client: TemporalGraphClient,
        vanilla_retriever: VanillaRetriever,
        setup: ExperimentSetup
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
        
        try:
            # 1. Retrieve from Graph (Agent-led, returns 5-15 based on complexity)
            try:
                # Agent wrapper returns all facts it deemed necessary (5-15)
                results_graph = await self.graph.search(query, num_results=15)  # Max request
                graph_count = len(results_graph) if results_graph else 0
                logger.info(f"Graph (Agent-led): got {graph_count} facts")
            except Exception as e:
                logger.error(f"Graph search failed: {e}")
                results_graph = []

            # 2. Retrieve from Vanilla (Fixed 5 chunks for detail supplement)
            try:
                result_vanilla = await self.vanilla.retrieve(query)
                vanilla_count = len(result_vanilla.results) if result_vanilla else 0
                logger.info(f"Vanilla: got={vanilla_count}, will use={min(vanilla_count, VANILLA_SUPPLEMENT)}")
            except Exception as e:
                logger.error(f"Vanilla search failed: {e}")
                result_vanilla = None
            
            combined_results = []
            
            # 3. Add ALL Graph Results (Agent already decided sufficiency: 5-15)
            for r in results_graph:
                combined_results.append(HybridSearchResult(
                    content=f"[FACT] {r.fact}",
                    source_type="graph",
                    score=r.score,
                    metadata={"entity": r.entity_name, "valid_at": r.valid_at}
                ))
                
            # 4. Add FIXED 5 Vanilla Results (detail supplement)
            if result_vanilla and result_vanilla.results:
                for r in result_vanilla.results[:VANILLA_SUPPLEMENT]:
                    combined_results.append(HybridSearchResult(
                        content=f"[DETAIL] {r.text}",
                        source_type="vanilla",
                        score=r.score,
                        metadata=r.metadata
                    ))
            
            logger.info(f"Hybrid retrieval finished. Graph={len(results_graph)}, Vanilla={min(vanilla_count, VANILLA_SUPPLEMENT) if result_vanilla else 0}, Total={len(combined_results)}")
            return combined_results
            
        except Exception as e:
            logger.error(f"Critical error in HybridRetriever.retrieve: {e}")
            return []

    def format_context(self, results: List[HybridSearchResult]) -> str:
        """Format merged results into a single context string"""
        lines = ["=== RETRIEVED CONTEXT ==="]
        
        # Group by source for clarity
        graph_facts = [r for r in results if r.source_type == "graph"]
        vanilla_docs = [r for r in results if r.source_type == "vanilla"]
        
        if graph_facts:
            lines.append("\n--- STRUCTURAL FACTS (FROM KNOWLEDGE GRAPH) ---")
            for r in graph_facts:
                lines.append(f"- {r.content.replace('[FACT] ', '')}")
                
        if vanilla_docs:
            lines.append("\n--- SPECIFIC DETAILS (FROM RAW SESSIONS) ---")
            for r in vanilla_docs:
                lines.append(f"{r.content.replace('[DETAIL] ', '')}")
                
        return "\n".join(lines)
