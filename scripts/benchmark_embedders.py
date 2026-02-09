# benchmark_embedders.py
"""
Benchmark Script untuk Embedding Models
Membandingkan berbagai model embedding untuk analisis skripsi
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.config import get_config
from src.embedders import (
    create_embedder_by_name,
    get_available_embedders,
    BaseEmbedder,
    EmbedderType
)
from src.embedders.factory import benchmark_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_texts(dataset_path: str, num_samples: int = 100) -> List[str]:
    """Load sample texts from dataset for benchmarking"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    texts = []
    for session in dataset.get("sessions", []):
        for turn in session.get("turns", []):
            texts.append(turn.get("user_query", ""))
            texts.append(turn.get("assistant_response", "")[:500])  # Truncate long responses
            
            if len(texts) >= num_samples:
                break
        if len(texts) >= num_samples:
            break
    
    return texts[:num_samples]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


async def evaluate_retrieval_quality(
    embedder: BaseEmbedder,
    queries: List[str],
    passages: List[str],
    relevance_labels: List[List[int]]  # For each query, list of relevant passage indices
) -> Dict[str, float]:
    """
    Evaluate retrieval quality using embedding similarity.
    
    Returns:
        MRR, Recall@k metrics
    """
    query_results = await embedder.embed(queries)
    passage_results = await embedder.embed(passages)
    
    query_embeddings = query_results.embeddings
    passage_embeddings = passage_results.embeddings
    
    mrr_sum = 0.0
    recall_at_5 = 0.0
    recall_at_10 = 0.0
    
    for i, q_emb in enumerate(query_embeddings):
        # Calculate similarities
        similarities = [
            cosine_similarity(q_emb, p_emb)
            for p_emb in passage_embeddings
        ]
        
        # Rank passages by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        
        # Calculate MRR
        relevant = set(relevance_labels[i])
        for rank, idx in enumerate(ranked_indices):
            if idx in relevant:
                mrr_sum += 1.0 / (rank + 1)
                break
        
        # Calculate Recall@k
        top_5 = set(ranked_indices[:5])
        top_10 = set(ranked_indices[:10])
        
        recall_at_5 += len(relevant & top_5) / len(relevant)
        recall_at_10 += len(relevant & top_10) / len(relevant)
    
    n_queries = len(queries)
    return {
        "mrr": mrr_sum / n_queries,
        "recall@5": recall_at_5 / n_queries,
        "recall@10": recall_at_10 / n_queries
    }


async def run_benchmark(
    embedder_names: List[str],
    test_texts: List[str],
    output_dir: str = "output/embedder_benchmark"
):
    """
    Run comprehensive benchmark on multiple embedders.
    """
    config = get_config()
    results = []
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name in embedder_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create embedder
            embedder = create_embedder_by_name(
                name,
                gemini_api_key=config.gemini.api_key,
                device=config.embedder_experiment.hf_device
            )
            
            # Initialize
            logger.info("Initializing embedder...")
            await embedder.initialize()
            
            # Benchmark latency
            logger.info("Running latency benchmark...")
            bench_result = await benchmark_embedder(
                embedder,
                test_texts[:50],  # Use 50 texts for latency benchmark
                num_runs=3
            )
            
            # Test with different batch sizes
            logger.info("Testing with different batch sizes...")
            for batch_size in [1, 10, 32]:
                subset = test_texts[:batch_size]
                result = await embedder.embed(subset)
                logger.info(f"  Batch {batch_size}: {result.latency_ms:.2f}ms")
            
            # Get final metrics
            metrics = embedder.metrics.to_dict()
            
            result = {
                **bench_result,
                "metrics": metrics,
                "status": "success"
            }
            
            logger.info(f"✅ {name}: {bench_result['avg_latency_ms']:.2f}ms avg, "
                       f"dim={embedder.dimension}")
            
            # Cleanup
            await embedder.close()
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            result = {
                "model_name": name,
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    # Save results
    output_file = Path(output_dir) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_test_texts": len(test_texts),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print a comparison table of results"""
    print("\n" + "="*80)
    print("EMBEDDING MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<40} {'Dim':>6} {'Avg Latency':>12} {'Throughput':>12}")
    print("-"*80)
    
    for r in results:
        if r.get("status") == "success":
            print(f"{r['model_name']:<40} {r['dimension']:>6} "
                  f"{r['avg_latency_ms']:>10.2f}ms "
                  f"{r['throughput_texts_per_sec']:>10.1f}/s")
        else:
            print(f"{r['model_name']:<40} {'ERROR':<6} {r.get('error', 'Unknown')[:30]}")
    
    print("="*80)


async def main():
    """Main benchmark entry point"""
    config = get_config()
    
    # Print available embedders
    print("\nAvailable embedders:")
    for name, desc in get_available_embedders().items():
        print(f"  - {name}: {desc}")
    
    # Load test texts
    logger.info(f"\nLoading test texts from {config.dataset_path}")
    test_texts = load_test_texts(config.dataset_path, num_samples=100)
    logger.info(f"Loaded {len(test_texts)} test texts")
    
    # Get embedders to test
    embedders_to_test = config.embedder_experiment.embedders_to_test
    logger.info(f"\nWill benchmark: {embedders_to_test}")
    
    # Run benchmark
    results = await run_benchmark(
        embedder_names=embedders_to_test,
        test_texts=test_texts
    )
    
    # Print comparison
    print_comparison_table(results)


if __name__ == "__main__":
    asyncio.run(main())
