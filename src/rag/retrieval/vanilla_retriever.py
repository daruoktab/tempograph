# src/retrieval/vanilla_retriever.py
"""
Vanilla Retriever
=================
Simple pure vector similarity search menggunakan SurrealDB.
Digunakan sebagai baseline comparison untuk Agentic RAG.

Perbedaan dengan Agentic:
- ❌ No fact extraction (raw turns)
- ❌ No graph structure
- ❌ No iterative retrieval
- ❌ No temporal reasoning
- ❌ No multi-hop
- ✅ Simple: embed query → vector search → rerank → return
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..vectordb import ChromaVectorDB, VanillaSearchResult
from ...config.experiment_setups import ExperimentSetup, RetrievalSettings

logger = logging.getLogger(__name__)


@dataclass
class VanillaRetrievalResult:
    """Result from vanilla retrieval"""

    results: List[VanillaSearchResult]
    context: str
    query: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def facts(self) -> List[VanillaSearchResult]:
        """Alias for compatibility with agentic retriever"""
        return self.results


class VanillaRetriever:
    """
    Simple pure vector retriever using SurrealDB.

    Flow:
    1. Embed query
    2. Vector search in ChromaDB (top-K candidates)
    3. Rerank (filter to final top-K)
    4. Return results

    No graph, no iteration, no fact extraction.
    """

    def __init__(
        self,
        chroma_db: ChromaVectorDB,
        settings: Optional[RetrievalSettings] = None,
        setup: Optional[ExperimentSetup] = None,
    ):
        """
        Initialize Vanilla Retriever.

        Args:
            chroma_db: Vector store client (SurrealVanillaVectorDB; param name legacy)
            settings: Retrieval settings (top-k, threshold)
            setup: Experiment setup for model configuration
        """
        self.db = chroma_db
        self.setup = setup

        # Use settings from setup or provided settings or defaults
        if settings:
            self.settings = settings
        elif setup:
            self.settings = setup.retrieval
        else:
            self.settings = RetrievalSettings()

        self._embedder = None
        self._llm_reranker = None

    async def initialize(self, embedder=None):
        """
        Initialize retriever with embedder.

        Args:
            embedder: Embedder instance for query embedding
        """
        self._embedder = embedder

        # Initialize LLM reranker if using LLM-based reranking
        if self.setup and self.setup.reranker_type == "llm":
            from .llm_reranker import LLMReranker

            rr = self.setup.reranker
            assert rr is not None, "LLM reranking requires setup.reranker"
            self._llm_reranker = LLMReranker(model_name=rr.name)
            await self._llm_reranker.initialize()
            logger.info(f"Vanilla Retriever initialized with LLM reranker: {rr.name}")
        else:
            logger.info("Vanilla Retriever initialized with embedding-based reranker")

    async def retrieve(self, query: str) -> VanillaRetrievalResult:
        """
        Simple single-shot retrieval from SurrealDB.

        Args:
            query: Query string

        Returns:
            VanillaRetrievalResult with retrieved documents
        """
        logger.debug(f"Vanilla retrieval for: {query[:50]}...")

        # Step 1: Embed query
        if self._embedder:
            query_embedding = await self._embedder.embed_single(query)
        else:
            # Use ChromaDB's internal embedder (if configured)
            query_embedding = None

        # Step 2: Vector search (get candidates)
        candidates = await self.db.search(
            query=query,
            n_results=self.settings.embedding_top_k,
            query_embedding=query_embedding,
        )

        logger.debug(f"Vector search returned {len(candidates)} candidates")

        # Step 3: Apply similarity threshold
        if self.settings.similarity_threshold > 0:
            candidates = [
                r for r in candidates if r.score >= self.settings.similarity_threshold
            ]
            logger.debug(f"After threshold filter: {len(candidates)} candidates")

        # Step 4: Rerank and get top-K
        if self._llm_reranker and len(candidates) > 0:
            # LLM-based reranking
            passages = [c.text for c in candidates]
            reranked = await self._llm_reranker.rank(query, passages)

            # Map back to VanillaSearchResult objects
            text_to_result = {c.text: c for c in candidates}
            final_results = []
            for passage, score in reranked[: self.settings.rerank_top_k]:
                if passage in text_to_result:
                    result = text_to_result[passage]
                    # Update score with rerank score
                    final_results.append(
                        VanillaSearchResult(
                            id=result.id,
                            text=result.text,
                            score=score,
                            metadata=result.metadata,
                        )
                    )
        else:
            # Already sorted by similarity from ChromaDB
            final_results = candidates[: self.settings.rerank_top_k]

        logger.debug(f"Final results: {len(final_results)} documents")

        # Build context string
        context = "\n".join([r.text for r in final_results])

        return VanillaRetrievalResult(
            results=final_results,
            context=context,
            query=query,
            metadata={
                "retriever_type": "vanilla",
                "storage_type": "surrealdb",
                "collection": self.db.collection_name,
                "embedding_top_k": self.settings.embedding_top_k,
                "rerank_top_k": self.settings.rerank_top_k,
                "candidates_found": len(candidates),
                "final_count": len(final_results),
                "reranker_type": self.setup.reranker_type
                if self.setup
                else "embedding",
            },
        )

    async def retrieve_batch(
        self, queries: List[str], show_progress: bool = True
    ) -> List[VanillaRetrievalResult]:
        """
        Retrieve for multiple queries.

        Args:
            queries: List of query strings
            show_progress: Show progress bar

        Returns:
            List of VanillaRetrievalResult
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(queries, desc="Vanilla Retrieval") if show_progress else queries

        for query in iterator:
            result = await self.retrieve(query)
            results.append(result)

            # Small delay for rate limiting (if using API embedder)
            await asyncio.sleep(0.05)

        return results


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


async def create_vanilla_retriever(setup: ExperimentSetup) -> VanillaRetriever:
    """
    Create and initialize vanilla retriever from experiment setup.

    Args:
        setup: Experiment setup (must be VANILLA type)

    Returns:
        Initialized VanillaRetriever
    """
    from ..vectordb import get_chroma_client
    from ...embedders import create_embedder, EmbedderType

    if setup.rag_type.value != "vanilla":
        raise ValueError(f"Expected VANILLA setup, got {setup.rag_type}")

    # Get ChromaDB client
    assert setup.storage.collection_name is not None
    chroma_db = get_chroma_client(
        collection_name=setup.storage.collection_name,
        persist_directory=setup.storage.persist_directory,
    )

    # Create embedder
    if setup.embedder.provider == "huggingface":
        embedder = create_embedder(
            embedder_type=EmbedderType.HUGGINGFACE, model_name=setup.embedder.name
        )
    else:
        from ...config.settings import get_config

        config = get_config()
        embedder = create_embedder(
            embedder_type=EmbedderType.GEMINI,
            model_name=setup.embedder.name,
            gemini_api_key=config.gemini.api_key,
        )

    await embedder.initialize()

    # Initialize ChromaDB with embedder
    await chroma_db.initialize(embedder=embedder)

    # Create retriever
    retriever = VanillaRetriever(chroma_db, setup=setup)
    await retriever.initialize(embedder=embedder)

    return retriever


# =============================================================================
# TEST
# =============================================================================


async def test_vanilla_retriever():
    """Test vanilla retriever"""
    from ...config.experiment_setups import SETUP_1V_VANILLA_GEMINI

    print("\n" + "=" * 60)
    print("TESTING VANILLA RETRIEVER")
    print("=" * 60)

    try:
        retriever = await create_vanilla_retriever(SETUP_1V_VANILLA_GEMINI)

        # Check if collection has data
        doc_count = retriever.db.count()
        print("\n✅ Retriever initialized")
        print(f"   Collection: {retriever.db.collection_name}")
        print(f"   Documents: {doc_count}")

        if doc_count == 0:
            print("\n⚠️ Collection is empty. Run ingestion first.")
            return

        # Test query
        query = "Kapan Aisha mulai project skincare?"
        result = await retriever.retrieve(query)

        print(f"\n📝 Query: {query}")
        print(f"   Results found: {len(result.results)}")
        print(f"   Metadata: {result.metadata}")

        print("\n   Top 3 results:")
        for i, r in enumerate(result.results[:3]):
            print(f"   {i + 1}. [{r.score:.3f}] {r.text[:60]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_vanilla_retriever())
