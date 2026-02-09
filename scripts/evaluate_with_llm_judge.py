#!/usr/bin/env python
# scripts/evaluate_with_llm_judge.py
"""
LLM-as-Judge Evaluation Script using existing metrics
========================================================

This script adds LLM-as-Judge evaluation to existing retrieval results.
Uses context_sufficiency_llm_judge from src/evaluation/metrics.py.

Metrics added:
- Context Sufficiency (LLM Judge): Is the retrieved context sufficient to answer?
  - information_presence: Is required info present?
  - completeness: Is context complete for the answer?
  - temporal_info: Is temporal info available (if needed)?
  - no_contradiction: Does context not contradict expected answer?

Usage:
    python scripts/evaluate_with_llm_judge.py --setup vanilla_gemini
    python scripts/evaluate_with_llm_judge.py --setup vanilla_gemma
    python scripts/evaluate_with_llm_judge.py --setup all
"""

import asyncio
import argparse
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.config import get_config
from src.evaluation.metrics import context_sufficiency_llm_judge, MetricResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path("output/evaluation_results")
OUTPUT_DIR = Path("output/evaluation_results")

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"


class LLMJudgeEvaluator:
    """Add LLM Judge evaluation to existing retrieval results"""
    
    def __init__(self, setup_name: str, judge_model: str = "gemini-2.5-pro"):
        self.setup_name = setup_name
        self.judge_model = judge_model
        self.config = get_config()
        
    def _load_retrieval_results(self) -> Dict:
        """Load retrieval results from previous evaluation"""
        results_path = RESULTS_DIR / f"{self.setup_name}_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Retrieval results not found: {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def evaluate(self) -> Dict[str, Any]:
        """Run LLM-as-judge evaluation using existing metrics"""
        
        # Load retrieval results
        retrieval_data = self._load_retrieval_results()
        results_list = retrieval_data.get("results", [])
        
        logger.info(f"Evaluating {len(results_list)} queries with LLM judge ({self.judge_model})...")
        
        judge_results = []
        successful_evals = 0
        
        for item in tqdm(results_list, desc=f"Judging ({self.setup_name})"):
            query = item["query"]
            expected = item["expected_answer"]
            
            # Get full context from top_k_texts
            # The texts are truncated in results, but we can use them
            context = "\n\n".join(item.get("top_k_texts", []))
            
            # If context is very short, try to expand from results
            if len(context) < 100:
                # Just use what we have
                pass
            
            # Use the existing context_sufficiency_llm_judge
            try:
                result: MetricResult = await context_sufficiency_llm_judge(
                    query=query,
                    retrieved_context=context,
                    expected_answer=expected,
                    judge_model=self.judge_model
                )
                
                judge_results.append({
                    "query_id": item["query_id"],
                    "query": query,
                    "expected_answer": expected,
                    "hit_rate": item.get("hit_rate", 0),
                    "mrr": item.get("reciprocal_rank", 0),
                    "context_sufficiency_score": result.score,
                    "context_sufficiency_details": result.details
                })
                
                if result.score > 0:
                    successful_evals += 1
                    
            except Exception as e:
                logger.warning(f"Judge failed for {item['query_id']}: {e}")
                judge_results.append({
                    "query_id": item["query_id"],
                    "query": query,
                    "expected_answer": expected,
                    "hit_rate": item.get("hit_rate", 0),
                    "mrr": item.get("reciprocal_rank", 0),
                    "context_sufficiency_score": 0.0,
                    "context_sufficiency_details": {"error": str(e)[:100]}
                })
            
            # Rate limit protection
            await asyncio.sleep(0.5)
        
        # Calculate aggregates
        valid_results = [r for r in judge_results if r["context_sufficiency_score"] > 0]
        
        summary = {
            "setup_name": self.setup_name,
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results_list),
            "successful_judgments": successful_evals,
            "judge_model": self.judge_model,
            
            # Retrieval metrics (from original)
            "mean_hit_rate": retrieval_data["summary"].get("mean_hit_rate", 0),
            "mean_mrr": retrieval_data["summary"].get("mean_mrr", 0),
            
            # LLM Judge metrics
            "mean_context_sufficiency": sum(r["context_sufficiency_score"] for r in valid_results) / len(valid_results) if valid_results else 0,
            
            # Detailed breakdowns (averages)
            "mean_information_presence": sum(r["context_sufficiency_details"].get("information_presence", 0) for r in valid_results) / len(valid_results) if valid_results else 0,
            "mean_completeness": sum(r["context_sufficiency_details"].get("completeness", 0) for r in valid_results) / len(valid_results) if valid_results else 0,
            "mean_temporal_info": sum(r["context_sufficiency_details"].get("temporal_info", 0) for r in valid_results) / len(valid_results) if valid_results else 0,
            "mean_no_contradiction": sum(r["context_sufficiency_details"].get("no_contradiction", 0) for r in valid_results) / len(valid_results) if valid_results else 0,
        }
        
        return {"summary": summary, "results": judge_results}
    
    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved: {output_path}")


async def evaluate_setup(setup_name: str, judge_model: str = "gemini-2.5-pro"):
    """Evaluate a single setup with LLM judge"""
    evaluator = LLMJudgeEvaluator(setup_name, judge_model)
    results = await evaluator.evaluate()
    
    output_path = OUTPUT_DIR / f"{setup_name}_llm_judge_results.json"
    evaluator.save_results(results, output_path)
    
    return results["summary"]


async def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge RAG Evaluation")
    parser.add_argument("--setup", choices=["vanilla_gemini", "vanilla_gemma", "all"], 
                       default="all", help="Setup to evaluate")
    parser.add_argument("--judge-model", default="gemini-2.5-pro",
                       help="Model to use as judge (default: gemini-2.5-pro)")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE EVALUATION (Context Sufficiency)")
    print("=" * 60)
    print(f"Judge Model: {args.judge_model}")
    
    setups = []
    if args.setup == "all":
        setups = ["vanilla_gemini", "vanilla_gemma"]
    else:
        setups = [args.setup]
    
    summaries = {}
    for setup in setups:
        print(f"\n--- Evaluating {setup} ---")
        try:
            summary = await evaluate_setup(setup, args.judge_model)
            summaries[setup] = summary
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {setup}: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for name, summary in summaries.items():
        print(f"\n{name}:")
        print(f"  Retrieval Metrics:")
        print(f"    Hit Rate: {summary['mean_hit_rate']:.3f}")
        print(f"    MRR:      {summary['mean_mrr']:.3f}")
        print(f"\n  Context Sufficiency (LLM Judge, 0-1 scale):")
        print(f"    Overall Score:        {summary['mean_context_sufficiency']:.3f}")
        print(f"    Information Presence: {summary['mean_information_presence']:.3f}")
        print(f"    Completeness:         {summary['mean_completeness']:.3f}")
        print(f"    Temporal Info:        {summary['mean_temporal_info']:.3f}")
        print(f"    No Contradiction:     {summary['mean_no_contradiction']:.3f}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
