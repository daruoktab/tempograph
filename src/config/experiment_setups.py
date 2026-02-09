# src/experiment_setups.py
"""
Experiment Setups untuk Evaluasi RAG System
============================================

4 Setup dengan 4 DATABASE TERPISAH:

VANILLA RAG (ChromaDB - Pure Vector):
- Setup 1V: vanilla_gemini (ChromaDB collection)
- Setup 2V: vanilla_gemma (ChromaDB collection)

AGENTIC RAG (Neo4j - Graph + Vector):
- Setup 1A: agentic_gemini (Neo4j group_id)
- Setup 2A: agentic_gemma (Neo4j group_id)

KEY DIFFERENCES:
- Vanilla: Raw sessions → embed → ChromaDB → vector search
- Agentic: Raw sessions → LLM extract facts → embed → Neo4j → graph + vector search

INGESTION UNIT: Per-Session (bukan per-turn)
- Real-world: dalam 1 sesi, semua turns masih di context window
- RAG hanya untuk retrieve dari sesi LAIN
- Lebih efisien: 100 API calls vs 1143 calls
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class SetupType(str, Enum):
    """Available experiment setups"""
    VANILLA_GEMINI = "vanilla_gemini"   # Setup 1V
    VANILLA_GEMMA = "vanilla_gemma"     # Setup 2V
    AGENTIC_GEMINI = "agentic_gemini"   # Setup 1A
    AGENTIC_GEMMA = "agentic_gemma"     # Setup 2A
    AGENTIC_GEMINI_HYBRID = "agentic_gemini_hybrid" # Setup 1H (Hybrid Gemini)
    AGENTIC_GEMMA_HYBRID = "agentic_gemma_hybrid"   # Setup 2H (Hybrid Gemma)


class RAGType(str, Enum):
    """Type of RAG system"""
    VANILLA = "vanilla"   # Pure vector search (ChromaDB)
    AGENTIC = "agentic"   # Graph + vector (Neo4j)


class ModelStack(str, Enum):
    """Model stack type"""
    GEMINI = "gemini"
    GEMMA = "gemma"


class StorageType(str, Enum):
    """Storage backend type"""
    CHROMADB = "chromadb"   # For Vanilla RAG
    NEO4J = "neo4j"         # For Agentic RAG


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# ChromaDB collections for Vanilla RAG
CHROMA_COLLECTIONS = {
    SetupType.VANILLA_GEMINI: "vanilla_gemini",
    SetupType.VANILLA_GEMMA: "vanilla_gemma",
}

# Neo4j group_ids for Agentic RAG
NEO4J_GROUP_IDS = {
    SetupType.AGENTIC_GEMINI: "agentic_gemini",
    SetupType.AGENTIC_GEMMA: "agentic_gemma",
}

# ChromaDB storage path
CHROMA_PERSIST_DIR = "./data/chroma"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: str  # "gemini", "gemma", "huggingface"
    description: str
    rpm: Optional[int] = None   # Requests per minute
    tpm: Optional[int] = None   # Tokens per minute
    rpd: Optional[int] = None   # Requests per day


@dataclass
class RetrievalSettings:
    """Retrieval configuration - SAME for all setups for fair comparison
    
    Reduced top_k to 5 for:
    - More rigorous evaluation (tests retrieval quality)
    - Lower input token cost (~50% reduction)
    - More realistic RAG scenario (production usually uses top 3-5)
    """
    embedding_top_k: int = 20       # Candidates from vector search
    rerank_top_k: int = 5           # Final results after rerank
    similarity_threshold: float = 0.5
    max_iterations: int = 3         # Only for Agentic (ignored in Vanilla)



@dataclass
class StorageConfig:
    """Storage configuration"""
    storage_type: StorageType
    # For ChromaDB
    collection_name: Optional[str] = None
    persist_directory: str = CHROMA_PERSIST_DIR
    # For Neo4j
    group_id: Optional[str] = None


@dataclass
class ExperimentSetup:
    """Complete configuration for an experiment setup"""
    name: str
    setup_type: SetupType
    rag_type: RAGType
    model_stack: ModelStack
    description: str
    
    # Storage configuration
    storage: StorageConfig
    
    # Embedding model (used for both ingestion and retrieval)
    embedder: ModelConfig
    
    # LLM for fact extraction (Agentic only)
    llm_extraction: Optional[ModelConfig] = None
    
    # LLM for retrieval operations (classification, etc) - Agentic only
    llm_small: Optional[ModelConfig] = None
    
    # Reranker configuration
    reranker: Optional[ModelConfig] = None
    reranker_type: str = "embedding"  # "embedding" or "llm"
    
    # Retrieval settings (SAME for fair comparison)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    
    # LLM Judge (always gemini-2.5-pro for consistency)
    llm_judge: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="gemini-2.5-pro",
        provider="gemini",
        description="LLM Judge for context sufficiency evaluation",
        rpm=15,
        tpm=1_000_000,
        rpd=300
    ))
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/saving"""
        return {
            "setup_name": self.name,
            "setup_type": self.setup_type.value,
            "rag_type": self.rag_type.value,
            "model_stack": self.model_stack.value,
            "storage": {
                "type": self.storage.storage_type.value,
                "collection_name": self.storage.collection_name,
                "group_id": self.storage.group_id,
            },
            "description": self.description,
            "embedder": self.embedder.name,
            "llm_extraction": self.llm_extraction.name if self.llm_extraction else None,
            "llm_small": self.llm_small.name if self.llm_small else None,
            "reranker": self.reranker.name if self.reranker is not None else None,
            "reranker_type": self.reranker_type,
            "retrieval": {
                "embedding_top_k": self.retrieval.embedding_top_k,
                "rerank_top_k": self.retrieval.rerank_top_k,
                "similarity_threshold": self.retrieval.similarity_threshold,
                "max_iterations": self.retrieval.max_iterations,
            },
            "llm_judge": self.llm_judge.name
        }


# =============================================================================
# FAIR RETRIEVAL SETTINGS (SAME FOR ALL)
# =============================================================================

# =============================================================================
# FAIR RETRIEVAL SETTINGS (SAME FOR ALL)
# =============================================================================

FAIR_RETRIEVAL_SETTINGS = RetrievalSettings(
    embedding_top_k=50,          # Search 50 candidates (increased for per-turn chunks)
    rerank_top_k=10,             # Return 10 final results
    similarity_threshold=0.5,    # Minimum similarity
    max_iterations=3             # For Agentic only
)

# Optimized settings for Pure Vector Search (Direct 10)
DIRECT_RETRIEVAL_SETTINGS = RetrievalSettings(
    embedding_top_k=10,          # Direct 10, no reranking needed
    rerank_top_k=10,
    similarity_threshold=0.5,
    max_iterations=3
)


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Gemini Models
GEMINI_EMBEDDER = ModelConfig(
    name="gemini-embedding-001",
    provider="gemini",
    description="Gemini embedding (API-based, 768 dim)",
    rpm=1500, tpm=10_000_000, rpd=None
)

GEMINI_LLM_EXTRACTION = ModelConfig(
    name="gemini-2.5-flash",
    provider="gemini",
    description="LLM for fact extraction (Agentic only)",
    rpm=1000, tpm=1_000_000, rpd=10_000
)

# NOTE: flash-lite confirmed NOT used by Graphiti ingestion (tested 2024-12-17)
# COMMENTED OUT FOR COST AUDIT - testing if code works without this model
# GEMINI_LLM_SMALL = ModelConfig(
#     name="gemini-2.5-flash-lite",
#     provider="gemini",
#     description="Small LLM for classification (currently unused by Graphiti)",
#     rpm=4000, tpm=4_000_000, rpd=None
# )
GEMINI_LLM_SMALL = None  # Disabled for cost audit

GEMINI_RERANKER = ModelConfig(
    name="gemini-embedding-001",
    provider="gemini",
    description="Embedding-based reranker (cosine similarity)",
    rpm=1500, tpm=10_000_000, rpd=None
)

# Used by both SETUP_1A (Agentic) and SETUP_1H (Hybrid)
GEMINI_LLM_EXTRACTION_HD = ModelConfig(
    name="gemini-2.5-flash",
    provider="gemini",
    description="LLM for high-detail fact extraction (Agentic Gemini)",
    rpm=1000, tpm=1_000_000, rpd=None
)

# Gemma Models (via Novita AI - OpenAI-compatible API)
GEMMA_EMBEDDER = ModelConfig(
    name="google/embeddinggemma-300m",
    provider="huggingface",
    description="EmbeddingGemma (HuggingFace LOCAL, 768 dim)",
    rpm=None, tpm=None, rpd=None  # Local, no rate limit
)

GEMMA_LLM_EXTRACTION = ModelConfig(
    name="google/gemma-3-27b-it",
    provider="novita",  # Via Novita AI (OpenAI-compatible)
    description="Gemma 3 27B IT for fact extraction (Agentic) via Novita AI",
    rpm=100, tpm=500_000, rpd=None
)


# =============================================================================
# SETUP 1V: VANILLA + GEMINI (ChromaDB)
# =============================================================================

SETUP_1V_VANILLA_GEMINI = ExperimentSetup(
    name="Setup 1V: Vanilla Gemini",
    setup_type=SetupType.VANILLA_GEMINI,
    rag_type=RAGType.VANILLA,
    model_stack=ModelStack.GEMINI,
    description="Vanilla RAG with Gemini embedding - pure vector search, no graph",
    
    storage=StorageConfig(
        storage_type=StorageType.CHROMADB,
        collection_name=CHROMA_COLLECTIONS[SetupType.VANILLA_GEMINI],
        persist_directory=CHROMA_PERSIST_DIR
    ),
    
    embedder=GEMINI_EMBEDDER,
    llm_extraction=None,  # No fact extraction in Vanilla
    llm_small=None,       # No classification in Vanilla
    reranker=GEMINI_RERANKER,
    reranker_type="embedding",
    retrieval=FAIR_RETRIEVAL_SETTINGS
)


# =============================================================================
# SETUP 2V: VANILLA + GEMMA (ChromaDB)
# =============================================================================

SETUP_2V_VANILLA_GEMMA = ExperimentSetup(
    name="Setup 2V: Vanilla Gemma",
    setup_type=SetupType.VANILLA_GEMMA,
    rag_type=RAGType.VANILLA,
    model_stack=ModelStack.GEMMA,
    description="Vanilla RAG with Gemma embedding - pure vector search, no graph",
    
    storage=StorageConfig(
        storage_type=StorageType.CHROMADB,
        collection_name=CHROMA_COLLECTIONS[SetupType.VANILLA_GEMMA],
        persist_directory=CHROMA_PERSIST_DIR
    ),
    
    embedder=GEMMA_EMBEDDER,
    llm_extraction=None,  # No fact extraction in Vanilla
    llm_small=None,       # No classification in Vanilla
    reranker=None,  # Use embedding for reranking (pure vector search)
    reranker_type="embedding",
    retrieval=DIRECT_RETRIEVAL_SETTINGS
)


# =============================================================================
# SETUP 2A: AGENTIC + GEMMA (Neo4j via Novita AI)
# =============================================================================

SETUP_2A_AGENTIC_GEMMA = ExperimentSetup(
    name="Setup 2A: Agentic Gemma (Novita)",
    setup_type=SetupType.AGENTIC_GEMMA,
    rag_type=RAGType.AGENTIC,
    model_stack=ModelStack.GEMMA,
    description="Agentic RAG with Gemma 3 27B IT via Novita AI - graph + vector, iterative retrieval",
    
    storage=StorageConfig(
        storage_type=StorageType.NEO4J,
        group_id=NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMMA]
    ),
    
    embedder=GEMMA_EMBEDDER,
    llm_extraction=GEMMA_LLM_EXTRACTION,  # Uses Novita AI provider
    llm_small=None,  # Not used
    reranker=None,  # Use embedding for reranking (pure vector search)
    reranker_type="embedding",
    retrieval=DIRECT_RETRIEVAL_SETTINGS
)


# =============================================================================
# SETUP 1A: AGENTIC GEMINI (High Detail - Primary)
# =============================================================================

# Note: This replaces the old "V2" setup as the default Agentic Gemini
SETUP_1A_AGENTIC_GEMINI = ExperimentSetup(
    name="Setup 1A: Agentic Gemini",
    setup_type=SetupType.AGENTIC_GEMINI,
    rag_type=RAGType.AGENTIC,
    model_stack=ModelStack.GEMINI,
    description="Agentic RAG with Gemini 2.5 Flash - high-detail fact extraction",
    
    storage=StorageConfig(
        storage_type=StorageType.NEO4J,
        group_id=NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMINI]
    ),
    
    embedder=GEMINI_EMBEDDER,
    llm_extraction=GEMINI_LLM_EXTRACTION_HD,
    llm_small=GEMINI_LLM_SMALL,  # Set to None for cost audit
    reranker=GEMINI_EMBEDDER,
    reranker_type="embedding",
    retrieval=DIRECT_RETRIEVAL_SETTINGS
)


# =============================================================================
# SETUP 1H: HYBRID GEMINI (Graph + Vanilla)
# =============================================================================

SETUP_1H_HYBRID_GEMINI = ExperimentSetup(
    name="Setup 1H: Hybrid Gemini (Graph + Vanilla)",
    setup_type=SetupType.AGENTIC_GEMINI_HYBRID,
    rag_type=RAGType.AGENTIC,  # Use Agentic pipeline as base
    model_stack=ModelStack.GEMINI,
    description="Hybrid RAG: Agentic Graph + Vanilla Vector combined",
    
    # Primary Storage (Graph) - shares with SETUP_1A
    storage=StorageConfig(
        storage_type=StorageType.NEO4J,
        group_id=NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMINI]
    ),
    
    embedder=GEMINI_EMBEDDER,
    llm_extraction=GEMINI_LLM_EXTRACTION_HD,
    llm_small=GEMINI_LLM_SMALL,  # Set to None for cost audit
    reranker=GEMINI_EMBEDDER,
    reranker_type="embedding",
    retrieval=DIRECT_RETRIEVAL_SETTINGS
)


# =============================================================================
# SETUP 2H: HYBRID GEMMA (Graph + Vanilla via Novita AI)
# =============================================================================

SETUP_2H_HYBRID_GEMMA = ExperimentSetup(
    name="Setup 2H: Hybrid Gemma (Graph + Vanilla)",
    setup_type=SetupType.AGENTIC_GEMMA_HYBRID,
    rag_type=RAGType.AGENTIC,  # Use Agentic pipeline as base
    model_stack=ModelStack.GEMMA,
    description="Hybrid RAG: Agentic Graph (Gemma via Novita) + Vanilla Vector combined",
    
    # Primary Storage (Graph) - shares with SETUP_2A
    storage=StorageConfig(
        storage_type=StorageType.NEO4J,
        group_id=NEO4J_GROUP_IDS[SetupType.AGENTIC_GEMMA]
    ),
    
    embedder=GEMMA_EMBEDDER,  # HuggingFace local
    llm_extraction=GEMMA_LLM_EXTRACTION,  # Novita AI
    llm_small=None,
    reranker=None,  # Use embedding for reranking
    reranker_type="embedding",
    retrieval=DIRECT_RETRIEVAL_SETTINGS
)

# ... (Hybrid specific config is handled by the Retriever class logic)


# =============================================================================
# SETUP REGISTRY
# =============================================================================

EXPERIMENT_SETUPS: Dict[SetupType, ExperimentSetup] = {
    SetupType.VANILLA_GEMINI: SETUP_1V_VANILLA_GEMINI,
    SetupType.VANILLA_GEMMA: SETUP_2V_VANILLA_GEMMA,
    SetupType.AGENTIC_GEMINI: SETUP_1A_AGENTIC_GEMINI,
    SetupType.AGENTIC_GEMMA: SETUP_2A_AGENTIC_GEMMA,
    SetupType.AGENTIC_GEMINI_HYBRID: SETUP_1H_HYBRID_GEMINI,
    SetupType.AGENTIC_GEMMA_HYBRID: SETUP_2H_HYBRID_GEMMA,
}


def get_setup(setup_type: SetupType) -> ExperimentSetup:
    """Get experiment setup by type"""
    return EXPERIMENT_SETUPS[setup_type]


def get_vanilla_setups() -> List[ExperimentSetup]:
    """Get all vanilla setups"""
    return [s for s in EXPERIMENT_SETUPS.values() if s.rag_type == RAGType.VANILLA]


def get_agentic_setups() -> List[ExperimentSetup]:
    """Get all agentic setups"""
    return [s for s in EXPERIMENT_SETUPS.values() if s.rag_type == RAGType.AGENTIC]


def list_setups() -> None:
    """Print all available setups"""
    print("\n" + "=" * 70)
    print("AVAILABLE EXPERIMENT SETUPS")
    print("=" * 70)
    
    print("\n📦 VANILLA RAG (ChromaDB - Pure Vector)")
    print("-" * 40)
    for setup in get_vanilla_setups():
        print(f"\n  📋 {setup.name}")
        print(f"     Storage: ChromaDB collection '{setup.storage.collection_name}'")
        print(f"     Embedder: {setup.embedder.name}")
        reranker_name = setup.reranker.name if setup.reranker else "N/A"
        print(f"     Reranker: {reranker_name} ({setup.reranker_type})")
    
    print("\n📦 AGENTIC RAG (Neo4j - Graph + Vector)")
    print("-" * 40)
    for setup in get_agentic_setups():
        print(f"\n  🤖 {setup.name}")
        print(f"     Storage: Neo4j group_id '{setup.storage.group_id}'")
        llm_name = setup.llm_extraction.name if setup.llm_extraction else "N/A"
        print(f"     LLM Extraction: {llm_name}")
        print(f"     Embedder: {setup.embedder.name}")
        reranker_name = setup.reranker.name if setup.reranker else "N/A"
        print(f"     Reranker: {reranker_name} ({setup.reranker_type})")
    
    print("\n" + "=" * 70)


def print_comparison_table() -> None:
    """Print comparison table of all 4 setups"""
    print("""
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              4 EXPERIMENT SETUPS COMPARISON                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  VANILLA RAG (ChromaDB)                │  AGENTIC RAG (Neo4j)                              │
│  ══════════════════════                │  ════════════════════                             │
│                                        │                                                    │
│  What's stored: RAW TURNS              │  What's stored: EXTRACTED FACTS                   │
│  Search: Pure vector similarity        │  Search: Graph + Vector + Iteration               │
│                                        │                                                    │
│  ┌──────────────────────────────────┐  │  ┌──────────────────────────────────┐             │
│  │  1V: vanilla_gemini (ChromaDB)   │  │  │  1A: agentic_gemini (Neo4j)      │             │
│  │                                  │  │  │                                  │             │
│  │  • 100 sessions (full text)      │  │  │  • ~800-1200 extracted facts     │             │
│  │  • gemini-embedding-001          │  │  │  • gemini-2.5-flash (extraction) │             │
│  │  • Embedding reranker            │  │  │  • gemini-embedding-001          │             │
│  │                                  │  │  │  • Iterative + graph traverse    │             │
│  └──────────────────────────────────┘  │  └──────────────────────────────────┘             │
│                                        │                                                    │
│  ┌──────────────────────────────────┐  │  ┌──────────────────────────────────┐             │
│  │  2V: vanilla_gemma (ChromaDB)    │  │  │  2A: agentic_gemma (Neo4j)       │             │
│  │                                  │  │  │                                  │             │
│  │  • 100 sessions (full text)      │  │  │  • ~800-1200 extracted facts     │             │
│  │  • embeddinggemma-300m (LOCAL)   │  │  │  • gemma-3-27b-it (extraction)   │             │
│  │  • LLM reranker (gemma-4b)       │  │  │  • embeddinggemma-300m (LOCAL)   │             │
│  │                                  │  │  │  • Iterative + graph traverse    │             │
│  └──────────────────────────────────┘  │  └──────────────────────────────────┘             │
│                                        │                                                    │
│  ════════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                             │
│  RETRIEVAL SETTINGS (SAME FOR ALL - FAIR COMPARISON):                                      │
│  • Top-K candidates: 30                                                                     │
│  • Final results: 10                                                                        │
│  • LLM Judge: gemini-2.5-pro                                                               │
│                                                                                             │
│  ════════════════════════════════════════════════════════════════════════════════════════  │
│                                                                                             │
│  COMPARISON VALUE:                                                                          │
│  • 1V vs 1A: Is graph + fact extraction worth it? (Gemini stack)                           │
│  • 2V vs 2A: Is graph + fact extraction worth it? (Gemma stack)                            │
│  • 1V vs 2V: Gemini vs Gemma embedding quality (Vanilla)                                   │
│  • 1A vs 2A: Gemini vs Gemma full stack (Agentic)                                          │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
""")


def print_database_info() -> None:
    """Print database configuration info"""
    print("""
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATABASE CONFIGURATION                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  CHROMADB (Vanilla RAG)                                                                    │
│  ══════════════════════                                                                    │
│  Location: ./data/chroma/                                                                  │
│                                                                                             │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐          │
│  │  Collection: vanilla_gemini         │  │  Collection: vanilla_gemma          │          │
│  │                                     │  │                                     │          │
│  │  Content:                           │  │  Content:                           │          │
│  │  • 100 sessions (full text each)    │  │  • 100 sessions (full text each)    │          │
│  │  • Embedded with gemini-embedding   │  │  • Embedded with embeddinggemma     │          │
│  │  • Metadata: session_id, turn_count │  │  • Metadata: session_id, turn_count │          │
│  └─────────────────────────────────────┘  └─────────────────────────────────────┘          │
│                                                                                             │
│  NEO4J (Agentic RAG)                                                                       │
│  ═══════════════════                                                                       │
│  Connection: bolt://localhost:7687                                                          │
│                                                                                             │
│  ┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐          │
│  │  Group ID: agentic_gemini           │  │  Group ID: agentic_gemma            │          │
│  │                                     │  │                                     │          │
│  │  Content:                           │  │  Content:                           │          │
│  │  • Entities (nodes)                 │  │  • Entities (nodes)                 │          │
│  │  • Facts (edges) + embeddings       │  │  • Facts (edges) + embeddings       │          │
│  │  • Temporal metadata                │  │  • Temporal metadata                │          │
│  │  • Extracted by gemini-2.5-flash    │  │  • Extracted by gemma-3-27b-it      │          │
│  └─────────────────────────────────────┘  └─────────────────────────────────────┘          │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    list_setups()
    print_comparison_table()
    print_database_info()
