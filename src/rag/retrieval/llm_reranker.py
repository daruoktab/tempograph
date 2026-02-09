# src/retrieval/llm_reranker.py
"""
LLM-based Reranker
==================
Reranking menggunakan LLM untuk scoring relevansi yang lebih akurat.
Digunakan di Setup 2 (Gemma) dengan gemma-3-4b-it.
"""

import asyncio
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Prompt template untuk reranking
RERANK_PROMPT = """Anda adalah sistem penilaian relevansi dokumen.

TUGAS: Berikan skor relevansi (0.0 - 1.0) untuk setiap passage terhadap query.

QUERY: {query}

PASSAGES:
{passages}

KRITERIA PENILAIAN:
- 1.0: Sangat relevan, langsung menjawab query
- 0.8-0.9: Relevan, mengandung informasi yang dibutuhkan
- 0.5-0.7: Cukup relevan, ada hubungan dengan query
- 0.2-0.4: Sedikit relevan, hanya menyinggung topik
- 0.0-0.1: Tidak relevan

Berikan output dalam format JSON array:
[
  {{"passage_id": 0, "score": 0.85, "reason": "..."}},
  {{"passage_id": 1, "score": 0.62, "reason": "..."}},
  ...
]

Hanya berikan JSON array, tanpa teks lain."""


@dataclass
class RerankResult:
    """Result of reranking"""

    passage: str
    score: float
    original_rank: int
    reason: Optional[str] = None


class LLMReranker:
    """
    Reranker menggunakan LLM untuk scoring.

    Keuntungan dibanding embedding-based:
    - Lebih akurat untuk query kompleks
    - Bisa memahami konteks dan nuansa
    - Bisa memberikan reasoning

    Kekurangan:
    - Lebih lambat (API call per batch)
    - Lebih mahal (token usage)
    """

    def __init__(
        self,
        model_name: str = "gemma-3-4b-it",
        max_passages_per_call: int = 10,
        temperature: float = 0.1,
    ):
        """
        Initialize LLM Reranker.

        Args:
            model_name: LLM model untuk reranking
            max_passages_per_call: Max passages per API call (untuk token limit)
            temperature: Temperature untuk LLM (rendah = lebih konsisten)
        """
        self.model_name = model_name
        self.max_passages_per_call = max_passages_per_call
        self.temperature = temperature
        self._client = None

    async def initialize(self):
        """Initialize LLM client"""
        from ...utils.gemma_client import GemmaClient

        self._client = GemmaClient(model=self.model_name)
        logger.info(f"LLM Reranker initialized with {self.model_name}")

    async def rank(
        self, query: str, passages: List[str], return_reasons: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Rerank passages berdasarkan relevansi dengan query.

        Args:
            query: Query string
            passages: List of passages to rerank
            return_reasons: Include reasoning in result

        Returns:
            List of (passage, score) sorted by score descending
        """
        if not passages:
            return []

        if len(passages) == 1:
            return [(passages[0], 1.0)]

        # Process in batches if too many passages
        all_results = []

        for i in range(0, len(passages), self.max_passages_per_call):
            batch = passages[i : i + self.max_passages_per_call]
            batch_results = await self._rank_batch(query, batch, i)
            all_results.extend(batch_results)

        # Sort by score descending
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Return as tuples
        return [(r.passage, r.score) for r in all_results]

    async def _rank_batch(
        self, query: str, passages: List[str], start_idx: int
    ) -> List[RerankResult]:
        """Rank a batch of passages"""
        import json

        # Format passages
        passages_text = "\n".join(
            [
                f"[{i}] {p[:500]}..." if len(p) > 500 else f"[{i}] {p}"
                for i, p in enumerate(passages)
            ]
        )

        # Create prompt
        prompt = RERANK_PROMPT.format(query=query, passages=passages_text)

        try:
            # Call LLM
            response = await self._client.generate(  # type: ignore[possibly-missing-attribute]
                prompt=prompt, temperature=self.temperature, max_output_tokens=1000
            )

            # Parse response
            result_text = response.text.strip()

            # Handle markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            scores = json.loads(result_text)

            # Build results
            results = []
            for item in scores:
                passage_id = item.get("passage_id", 0)
                if passage_id < len(passages):
                    results.append(
                        RerankResult(
                            passage=passages[passage_id],
                            score=float(item.get("score", 0.5)),
                            original_rank=start_idx + passage_id,
                            reason=item.get("reason"),
                        )
                    )

            # Add any missing passages with default score
            scored_ids = {item.get("passage_id") for item in scores}
            for i, passage in enumerate(passages):
                if i not in scored_ids:
                    results.append(
                        RerankResult(
                            passage=passage,
                            score=0.5,  # Default score
                            original_rank=start_idx + i,
                            reason="Not scored by LLM",
                        )
                    )

            return results

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM rerank response: {e}")
            # Fallback: return with default scores based on original order
            return [
                RerankResult(
                    passage=p,
                    score=1.0 - (i * 0.05),  # Decreasing scores
                    original_rank=start_idx + i,
                    reason="Parse error fallback",
                )
                for i, p in enumerate(passages)
            ]

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            # Return original order with default scores
            return [
                RerankResult(
                    passage=p,
                    score=0.5,
                    original_rank=start_idx + i,
                    reason=f"Error: {str(e)}",
                )
                for i, p in enumerate(passages)
            ]


class CrossEncoderLLMWrapper:
    """
    Wrapper to make LLMReranker compatible with Graphiti's CrossEncoderClient interface.
    """

    def __init__(self, llm_reranker: LLMReranker):
        self.llm_reranker = llm_reranker

    async def rank(self, query: str, passages: List[str]) -> List[Tuple[str, float]]:
        """Graphiti-compatible rank interface"""
        return await self.llm_reranker.rank(query, passages)


# Factory function
def create_llm_reranker(model_name: str = "gemma-3-4b-it") -> LLMReranker:
    """Create LLM reranker instance"""
    return LLMReranker(model_name=model_name)


# Test
if __name__ == "__main__":

    async def test():
        reranker = LLMReranker(model_name="gemma-3-4b-it")
        await reranker.initialize()

        query = "Kapan Aisha mulai project skincare?"
        passages = [
            "Aisha bekerja sebagai content creator di Bandung",
            "Project skincare dimulai awal Januari 2024",
            "Dewi adalah partner bisnis Aisha",
            "Aisha sedang menyiapkan campaign untuk klien baru produk skincare",
        ]

        results = await reranker.rank(query, passages)

        print(f"\nQuery: {query}\n")
        print("Reranked results:")
        for i, (passage, score) in enumerate(results):
            print(f"  {i + 1}. [{score:.2f}] {passage[:60]}...")

    asyncio.run(test())
