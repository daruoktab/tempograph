# src/vectordb/chroma_client.py
"""
ChromaDB Client for Vanilla RAG
===============================
Pure vector similarity search tanpa graph features.
Digunakan sebagai baseline comparison untuk Agentic RAG.

Storage:
- Local persistent storage di ./data/chroma/
- 2 collections: vanilla_gemini, vanilla_gemma
"""

import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings

from ...config.settings import get_config

logger = logging.getLogger(__name__)

# Default storage path
CHROMA_PERSIST_DIR = "./data/chroma"

# Collection names
COLLECTION_VANILLA_GEMINI = "vanilla_gemini"
COLLECTION_VANILLA_GEMMA = "vanilla_gemma"


@dataclass
class VanillaDocument:
    """Document for vanilla RAG (raw turn)"""

    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VanillaSearchResult:
    """Search result from vanilla RAG"""

    id: str
    text: str
    score: float  # Similarity score (higher = more similar)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChromaVectorDB:
    """
    ChromaDB client for Vanilla RAG.

    Features:
    - Local persistent storage
    - Support for custom embeddings (Gemini or Gemma)
    - Simple API: add, search, delete
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = CHROMA_PERSIST_DIR,
        embedding_function: Optional[Any] = None,
    ):
        """
        Initialize ChromaDB client.

        Args:
            collection_name: Name of the collection (vanilla_gemini or vanilla_gemma)
            persist_directory: Directory for persistent storage
            embedding_function: Custom embedding function (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._embedding_function = embedding_function

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self._collection = None
        self._embedder = None

    async def initialize(self, embedder=None):
        """
        Initialize collection and embedder.

        Args:
            embedder: Custom embedder instance (from src/embedders)
        """
        self._embedder = embedder

        # Get or create collection
        # Note: We don't use ChromaDB's built-in embedding function
        # because we want to use our own embedders (Gemini/Gemma)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(f"ChromaDB collection '{self.collection_name}' initialized")
        logger.info(f"  Persist directory: {self.persist_directory}")
        logger.info(f"  Current document count: {self._collection.count()}")

    async def add_documents(
        self, documents: List[VanillaDocument], show_progress: bool = True
    ) -> int:
        """
        Add documents to collection.

        Args:
            documents: List of VanillaDocument to add
            show_progress: Show progress bar

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        from tqdm import tqdm

        ids = []
        texts = []
        embeddings = []
        metadatas = []

        iterator = (
            tqdm(documents, desc="Adding documents") if show_progress else documents
        )

        for doc in iterator:
            ids.append(doc.id)
            texts.append(doc.text)
            metadatas.append(doc.metadata)

            # Get embedding
            if doc.embedding:
                embeddings.append(doc.embedding)
            elif self._embedder:
                # Generate embedding using our embedder
                embedding = await self._embedder.embed(doc.text)
                embeddings.append(embedding)
            else:
                raise ValueError("No embedding provided and no embedder configured")

        # Add to collection
        self._collection.add(  # type: ignore[possibly-missing-attribute]
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

        logger.info(f"Added {len(documents)} documents to '{self.collection_name}'")
        return len(documents)

    async def add_document(self, document: VanillaDocument) -> bool:
        """Add single document"""
        count = await self.add_documents([document], show_progress=False)
        return count > 0

    async def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[VanillaSearchResult]:
        """
        Search for similar documents.

        Args:
            query: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            query_embedding: Pre-computed query embedding (optional)

        Returns:
            List of VanillaSearchResult sorted by similarity
        """
        # Get query embedding
        if query_embedding:
            q_embedding = query_embedding
        elif self._embedder:
            q_embedding = await self._embedder.embed(query)
        else:
            raise ValueError("No query embedding provided and no embedder configured")

        # Search
        import functools

        loop = asyncio.get_running_loop()
        func = functools.partial(
            self._collection.query,  # type: ignore[possibly-missing-attribute]
            query_embeddings=[q_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = await loop.run_in_executor(None, func)

        # Convert to VanillaSearchResult
        search_results = []

        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                search_results.append(
                    VanillaSearchResult(
                        id=doc_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        score=similarity,
                        metadata=results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                    )
                )

        return search_results

    async def get_document(self, doc_id: str) -> Optional[VanillaDocument]:
        """Get document by ID"""
        result = self._collection.get(  # type: ignore[possibly-missing-attribute]
            ids=[doc_id], include=["documents", "metadatas", "embeddings"]
        )

        if result and result["ids"]:
            return VanillaDocument(
                id=result["ids"][0],
                text=result["documents"][0] if result["documents"] else "",
                embedding=result["embeddings"][0] if result["embeddings"] else None,  # type: ignore[invalid-argument-type]
                metadata=result["metadatas"][0] if result["metadatas"] else {},  # type: ignore[invalid-argument-type]
            )
        return None

    def count(self) -> int:
        """Get document count"""
        return self._collection.count()  # type: ignore[possibly-missing-attribute]

    async def clear(self):
        """Clear all documents from collection"""
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.warning(f"Cleared collection '{self.collection_name}'")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "collection_name": self.collection_name,
            "document_count": self._collection.count(),  # type: ignore[possibly-missing-attribute]
            "persist_directory": self.persist_directory,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_chroma_instances: Dict[str, ChromaVectorDB] = {}


def get_chroma_client(
    collection_name: str, persist_directory: str = CHROMA_PERSIST_DIR
) -> ChromaVectorDB:
    """
    Get or create ChromaDB client instance.

    Args:
        collection_name: Collection name (vanilla_gemini or vanilla_gemma)
        persist_directory: Storage directory

    Returns:
        ChromaVectorDB instance
    """
    key = f"{persist_directory}:{collection_name}"

    if key not in _chroma_instances:
        _chroma_instances[key] = ChromaVectorDB(
            collection_name=collection_name, persist_directory=persist_directory
        )

    return _chroma_instances[key]


def get_vanilla_gemini_db() -> ChromaVectorDB:
    """Get ChromaDB for Vanilla Gemini setup"""
    return get_chroma_client(COLLECTION_VANILLA_GEMINI)


def get_vanilla_gemma_db() -> ChromaVectorDB:
    """Get ChromaDB for Vanilla Gemma setup"""
    return get_chroma_client(COLLECTION_VANILLA_GEMMA)


# =============================================================================
# TEST
# =============================================================================


async def test_chroma():
    """Test ChromaDB setup"""
    print("\n" + "=" * 60)
    print("TESTING CHROMADB")
    print("=" * 60)

    # Create test client
    db = ChromaVectorDB(
        collection_name="test_collection", persist_directory="./data/chroma_test"
    )

    # Initialize without embedder (we'll provide embeddings manually)
    await db.initialize()

    print(f"\n✅ Collection created: {db.collection_name}")
    print(f"   Document count: {db.count()}")

    # Add test documents with dummy embeddings
    test_docs = [
        VanillaDocument(
            id="turn_1",
            text="Aisha cerita tentang project skincare barunya",
            embedding=[0.1] * 768,  # Dummy 768-dim embedding
            metadata={"session_id": 1, "turn_id": 0, "speaker": "user"},
        ),
        VanillaDocument(
            id="turn_2",
            text="Project skincare dimulai bulan Januari",
            embedding=[0.2] * 768,
            metadata={"session_id": 1, "turn_id": 1, "speaker": "assistant"},
        ),
        VanillaDocument(
            id="turn_3",
            text="Dewi adalah partner bisnis Aisha",
            embedding=[0.3] * 768,
            metadata={"session_id": 2, "turn_id": 0, "speaker": "user"},
        ),
    ]

    await db.add_documents(test_docs, show_progress=False)
    print(f"\n✅ Added {len(test_docs)} documents")
    print(f"   New count: {db.count()}")

    # Search with dummy query embedding
    query_embedding = [0.15] * 768  # Should be closest to turn_1
    results = await db.search(
        query="project skincare", n_results=3, query_embedding=query_embedding
    )

    print(f"\n✅ Search results:")
    for i, r in enumerate(results):
        print(f"   {i + 1}. [{r.score:.4f}] {r.text[:50]}...")

    # Get stats
    stats = db.get_stats()
    print(f"\n📊 Stats: {stats}")

    # Clear test collection
    await db.clear()
    print(f"\n🗑️ Cleared collection. Count: {db.count()}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_chroma())
