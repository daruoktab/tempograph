# src/graph_client.py
"""
Neo4j Graph Client Wrapper
Abstraksi untuk interaksi dengan Neo4j Temporal Knowledge Graph via Graphiti

Mendukung:
- Multiple embedding models (Gemini, HuggingFace)
- Multiple LLM providers (Gemini, OpenRouter, HuggingFace)
- Experiment-based data separation via group_id
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder as GraphitiGeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
import numpy as np

from .config import get_config, Neo4jConfig, GeminiConfig
from .experiment_setups import ExperimentSetup
from .embedders import create_embedder, EmbedderType, BaseEmbedder
from .llm_providers import create_llm_provider, LLMProviderType, BaseLLMProvider

logger = logging.getLogger(__name__)


class EmbeddingReranker(CrossEncoderClient):
    """
    Reranker menggunakan embedding similarity.
    Fallback jika cross-encoder model tidak tersedia.
    Mendukung berbagai embedder types.
    """
    def __init__(self, embedder: Any):  # Accept any embedder type
        self.embedder = embedder
        self._is_graphiti_embedder = hasattr(embedder, 'create_embedding')

    async def rank(self, query: str, passages: List[str]) -> List[tuple[str, float]]:
        if not passages:
            return []
        
        # Embed query and passages based on embedder type
        if self._is_graphiti_embedder:
            # Graphiti's GeminiEmbedder uses create_embedding
            query_result = await self.embedder.create_embedding(query)
            query_embedding = query_result.embeddings[0] if hasattr(query_result, 'embeddings') else query_result
            passage_embeddings = []
            for p in passages:
                p_result = await self.embedder.create_embedding(p)
                passage_embeddings.append(p_result.embeddings[0] if hasattr(p_result, 'embeddings') else p_result)
        elif hasattr(self.embedder, 'embed'):
            # Our custom embedder
            query_embedding = await self.embedder.embed(query)
            passage_embeddings = await self.embedder.embed_batch(passages)
        else:
            # GraphitiEmbedderWrapper
            query_embedding = (await self.embedder.embed([query]))[0]
            passage_embeddings = await self.embedder.embed(passages)
        
        # Normalize vectors
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v
        
        q_norm = normalize(np.array(query_embedding))
        
        # Calculate cosine similarity
        results = []
        for i, p_emb in enumerate(passage_embeddings):
            p_norm = normalize(np.array(p_emb))
            score = float(np.dot(q_norm, p_norm))
            results.append((passages[i], score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


@dataclass
class SearchResult:
    """Standardized search result"""
    fact: str
    score: float
    entity_name: Optional[str] = None
    created_at: Optional[datetime] = None
    valid_at: Optional[datetime] = None
    source_description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GraphitiEmbedderWrapper:
    """
    Wrapper to make our custom embedders compatible with Graphiti's expected interface.
    Graphiti expects: await embedder.embed(List[str]) -> List[List[float]]
    """
    
    def __init__(self, custom_embedder: Any):  # Accept any embedder
        self.custom_embedder = custom_embedder
        # Get dimension if available
        self.embedding_dim = getattr(custom_embedder, 'embedding_dim', 768)
        
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts - Graphiti compatible interface."""
        if hasattr(self.custom_embedder, 'embed_batch'):
            return await self.custom_embedder.embed_batch(texts)
        elif hasattr(self.custom_embedder, 'embed'):
            # Fall back to embedding one at a time
            results = []
            for text in texts:
                result = await self.custom_embedder.embed(text)
                results.append(result)
            return results
        else:
            raise ValueError("Custom embedder has no embed or embed_batch method")


class TemporalGraphClient:
    """
    Client untuk interaksi dengan Neo4j Temporal Knowledge Graph.
    Wrapper di atas Graphiti dengan fitur tambahan untuk temporal queries.
    
    Mendukung experiment-based data separation:
    - Setiap kombinasi embedding + LLM memiliki group_id unik
    - Data dipisahkan di Neo4j berdasarkan group_id
    """
    
    def __init__(
        self,
        neo4j_config: Optional[Neo4jConfig] = None,
        gemini_config: Optional[GeminiConfig] = None,
        group_id: Optional[str] = None,
        setup: Optional[ExperimentSetup] = None
    ):
        """
        Initialize TemporalGraphClient.
        
        Args:
            neo4j_config: Neo4j connection config
            gemini_config: Gemini API config (used as fallback if setup not provided)
            group_id: Manual group_id override
            setup: Experiment configuration (embedding + LLM model selection)
        """
        config = get_config()
        self.neo4j_config = neo4j_config or config.neo4j
        self.gemini_config = gemini_config or config.gemini
        self.setup = setup
        
        # Generate group_id based on experiment or fallback
        if group_id:
            self.group_id = group_id
        elif setup:
            self.group_id = setup.storage.group_id
        else:
            self.group_id = f"temporal_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._client: Optional[Graphiti] = None
        self._embedder: Optional[Union[GraphitiGeminiEmbedder, BaseEmbedder]] = None
        self._llm_provider: Optional[BaseLLMProvider] = None
        self._driver: Optional[Neo4jDriver] = None
        self._experiment_manager: Optional[ExperimentManager] = None
        
    @classmethod
    def from_setup(
        cls,
        setup: ExperimentSetup,
        neo4j_config: Optional[Neo4jConfig] = None
    ) -> "TemporalGraphClient":
        """
        Create client from an experiment setup.
        
        Args:
            setup: The experiment setup
            neo4j_config: Optional Neo4j config override
            
        Returns:
            Configured TemporalGraphClient
        """
        return cls(
            neo4j_config=neo4j_config,
            setup=setup
        )
        
    async def initialize(self):
        """Initialize all connections and clients"""
        logger.info(f"Initializing TemporalGraphClient with group_id: {self.group_id}")
        
        if self.setup:
            logger.info(f"Using setup: {self.setup.name}")
            logger.info(f"  Embedding: {self.setup.embedder.name}")
            if self.setup.llm_extraction:
                logger.info(f"  LLM Extraction: {self.setup.llm_extraction.name}")
        
        # Initialize embedder based on experiment setup or default
        if self.setup and self.setup.embedder.provider == "huggingface":
            self._embedder = create_embedder(
                embedder_type=EmbedderType.HUGGINGFACE,
                model_name=self.setup.embedder.name
            )
            await self._embedder.initialize()
        elif self.setup and self.setup.embedder.provider == "gemini":
            # Use Graphiti's Gemini embedder directly
            self._embedder = GraphitiGeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.gemini_config.api_key,
                    embedding_model=self.setup.embedder.name
                )
            )
        else:
            # Fallback to Graphiti's Gemini embedder
            self._embedder = GraphitiGeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.gemini_config.api_key,
                    embedding_model=self.gemini_config.embedding_model
                )
            )
        
        # Initialize LLM client
        # Note: Graphiti currently only supports GeminiClient internally
        # We use our custom LLM provider for other use cases
        if self.setup and self.setup.llm_extraction and self.setup.llm_extraction.provider != "gemini":
            self._llm_provider = create_llm_provider(
                provider_type=LLMProviderType(self.setup.llm_extraction.provider.upper()),
                model_name=self.setup.llm_extraction.name
            )
        
        # Graphiti still needs GeminiClient for graph operations
        # Prefer SETUP config, fallback to gemini_config only if necessary
        extraction_model = self.setup.llm_extraction.name if (self.setup and self.setup.llm_extraction) else self.gemini_config.model_hard
        small_model = self.setup.llm_small.name if (self.setup and self.setup.llm_small) else self.gemini_config.model_medium
        
        llm_client = GeminiClient(
            config=LLMConfig(
                api_key=self.gemini_config.api_key,
                model=extraction_model
            )
        )
        llm_client.small_model = small_model
        logger.debug(f"GraphClient initialized with Main: {extraction_model}, Small: {small_model}")
        
        # Initialize Neo4j driver
        self._driver = Neo4jDriver(
            uri=self.neo4j_config.uri,
            user=self.neo4j_config.user,
            password=self.neo4j_config.password,
            database=self.neo4j_config.database
        )
        
        # Create embedder wrapper for Graphiti if using custom embedder
        graphiti_embedder = self._create_graphiti_compatible_embedder()
        
        # Initialize Graphiti client
        self._client = Graphiti(
            uri="",  # Ignored when graph_driver is provided
            user="",
            password="",
            graph_driver=self._driver,
            llm_client=llm_client,
            embedder=graphiti_embedder,
            cross_encoder=EmbeddingReranker(graphiti_embedder)
        )
        
        # Build indices (idempotent operation)
        logger.info("Building indices and constraints...")
        await self._driver.build_indices_and_constraints()
        logger.info("Initialization complete")
        
    def _create_graphiti_compatible_embedder(self):
        """
        Create an embedder compatible with Graphiti's interface.
        If using custom embedder, wrap it to match Graphiti's expected interface.
        """
        if self._embedder is None:
            raise RuntimeError("Embedder not initialized")
        
        if isinstance(self._embedder, GraphitiGeminiEmbedder):
            return self._embedder
        
        # Wrap our custom embedder in Graphiti-compatible interface
        return GraphitiEmbedderWrapper(self._embedder)
        
    async def close(self):
        """Close all connections"""
        if self._driver:
            await self._driver.close()
            logger.info("Connections closed")
    
    @property
    def client(self) -> Graphiti:
        """Get underlying Graphiti client"""
        if self._client is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        return self._client
    
    @property
    def embedder(self) -> Union[GraphitiGeminiEmbedder, BaseEmbedder]:
        """Get embedder for external use"""
        if self._embedder is None:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        return self._embedder
    
    @property
    def llm_provider(self) -> Optional[BaseLLMProvider]:
        """Get LLM provider for external use (may be None if using default Gemini)"""
        return self._llm_provider
    
    @property
    def experiment_name(self) -> str:
        """Get experiment name"""
        if self.setup:
            return self.setup.name
        return f"default_{self.group_id}"
    
    # ==========================================================================
    # INGESTION METHODS
    # ==========================================================================
    
    async def add_episode(
        self,
        content: str,
        name: str,
        source_description: str,
        reference_time: Optional[datetime] = None,
        source_type: EpisodeType = EpisodeType.text
    ) -> str:
        """
        Ingest sebuah episode (turn percakapan) ke graph.
        
        Args:
            content: Teks dari turn percakapan
            name: Nama episode (e.g., "Session 1 Turn 0")
            source_description: Deskripsi sumber (e.g., "Speaker: user")
            reference_time: Waktu terjadinya percakapan
            source_type: Tipe episode
            
        Returns:
            Episode UUID
        """
        if reference_time is None:
            reference_time = datetime.now()
            
        result = await self.client.add_episode(
            name=name,
            group_id=self.group_id,
            episode_body=content,
            source=source_type,
            source_description=source_description,
            reference_time=reference_time
        )
        
        logger.debug(f"Added episode: {name}")
        return result.uuid if hasattr(result, 'uuid') else str(result)
    
    async def add_fact(
        self,
        fact: str,
        source_entity: str,
        target_entity: str,
        relation_name: str,
        valid_at: Optional[datetime] = None
    ):
        """
        Tambahkan fakta terstruktur ke graph.
        
        Args:
            fact: Statement fakta lengkap
            source_entity: Nama entity sumber
            target_entity: Nama entity target
            relation_name: Nama relasi (e.g., "bekerja_di", "teman_dengan")
            valid_at: Kapan fakta ini valid
        """
        # TODO: Implement direct fact insertion via Cypher
        # For now, we rely on Graphiti's automatic extraction
        pass
    
    # ==========================================================================
    # RETRIEVAL METHODS
    # ==========================================================================
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Semantic search di knowledge graph.
        
        Args:
            query: Query string
            num_results: Jumlah hasil yang diinginkan
            group_ids: Filter by group IDs (default: current group)
            
        Returns:
            List of SearchResult
        """
        if group_ids is None:
            group_ids = [self.group_id]
            
        results = await self.client.search(
            group_ids=group_ids,
            query=query,
            num_results=num_results
        )
        
        return [
            SearchResult(
                fact=r.fact,
                score=getattr(r, 'score', 0.0),
                entity_name=getattr(r, 'entity_name', None),
                created_at=getattr(r, 'created_at', None),
                valid_at=getattr(r, 'valid_at', None),
                source_description=getattr(r, 'source_description', None),
                metadata={}
            )
            for r in results
        ] if results else []
    
    async def search_with_temporal_filter(
        self,
        query: str,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        Semantic search dengan filter temporal.
        
        Args:
            query: Query string
            before: Filter fakta yang terjadi sebelum tanggal ini
            after: Filter fakta yang terjadi setelah tanggal ini
            num_results: Jumlah hasil
            
        Returns:
            Filtered search results
        """
        # Get all results first
        results = await self.search(query, num_results=num_results * 2)
        
        # Apply temporal filter
        filtered = []
        for r in results:
            if r.valid_at:
                if before and r.valid_at > before:
                    continue
                if after and r.valid_at < after:
                    continue
            filtered.append(r)
            
        return filtered[:num_results]
    
    async def get_entity_facts(
        self,
        entity_name: str,
        limit: int = 20
    ) -> List[SearchResult]:
        """
        Ambil semua fakta yang terkait dengan entity tertentu.
        
        Args:
            entity_name: Nama entity
            limit: Maksimum hasil
            
        Returns:
            List of facts about the entity
        """
        try:
            # Note: Neo4jDriver.execute_query only takes query string
            gid = self.group_id.replace("'", "\\'")
            name = entity_name.replace("'", "\\'")
            
            results = await self._driver.execute_query(
                f"MATCH (e:Entity {{name: '{name}', group_id: '{gid}'}})-[r:RELATES_TO]-(other) "
                f"RETURN r.fact as fact, r.created_at as created_at, r.valid_at as valid_at "
                f"LIMIT {limit}"
            )
            
            return [
                SearchResult(
                    fact=r['fact'],
                    score=1.0,
                    entity_name=entity_name,
                    created_at=r.get('created_at'),
                    valid_at=r.get('valid_at')
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Error getting entity facts: {e}")
            return []
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    async def get_stats(self) -> Dict[str, int]:
        """Get statistics about the graph"""
        try:
            # Note: Neo4jDriver.execute_query only takes query string
            # Parameters are embedded in the query for simplicity
            gid = self.group_id.replace("'", "\\'")
            
            entity_count = await self._driver.execute_query(
                f"MATCH (e:Entity {{group_id: '{gid}'}}) RETURN count(e) as count"
            )
            
            edge_count = await self._driver.execute_query(
                f"MATCH ()-[r:RELATES_TO {{group_id: '{gid}'}}]->() RETURN count(r) as count"
            )
            
            episode_count = await self._driver.execute_query(
                f"MATCH (e:Episodic {{group_id: '{gid}'}}) RETURN count(e) as count"
            )
            
            return {
                "entities": self._extract_count(entity_count),
                "edges": self._extract_count(edge_count),
                "episodes": self._extract_count(episode_count)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"entities": 0, "edges": 0, "episodes": 0}
    
    def _extract_count(self, result) -> int:
        """Extract count from various result formats"""
        try:
            if result is None:
                return 0
            if isinstance(result, int):
                return result
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if isinstance(first, dict):
                    return first.get('count', 0)
                if hasattr(first, 'count'):
                    return first.count
            if hasattr(result, 'single'):
                record = result.single()
                return record['count'] if record else 0
            return 0
        except Exception:
            return 0
    
    async def clear_group(self):
        """Clear all data for current group_id"""
        logger.warning(f"Clearing all data for group: {self.group_id}")
        gid = self.group_id.replace("'", "\\'")
        
        await self._driver.execute_query(
            f"MATCH (n {{group_id: '{gid}'}}) DETACH DELETE n"
        )


# Convenience function for quick testing
async def test_connection():
    """Test Neo4j connection"""
    client = TemporalGraphClient()
    try:
        await client.initialize()
        stats = await client.get_stats()
        print(f"✅ Connection successful!")
        print(f"   Group ID: {client.group_id}")
        print(f"   Stats: {stats}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    finally:
        await client.close()





if __name__ == "__main__":
    asyncio.run(test_connection())
