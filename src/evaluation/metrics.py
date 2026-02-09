# src/evaluation/metrics.py
"""
Evaluation Metrics
==================
Comprehensive metrics untuk evaluasi RAG system dengan temporal capabilities.

FAIR COMPARISON METRICS:
- All metrics are normalized (0-1 range)
- Independent of retrieved context size
- Focus on retrieval QUALITY not QUANTITY

Metrics:
1. Context Recall: Required facts found / Total required
2. Context Precision: Relevant facts / Total retrieved  
3. Hit Rate: Binary - is answer present in context?
4. MRR: Mean Reciprocal Rank - position of correct answer
5. Temporal Recall: Temporal facts correctly retrieved
6. Context Sufficiency: LLM Judge - is context sufficient to answer?
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np

from ..utils import get_rate_limiter, log_token_usage

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric calculation"""
    name: str
    score: float
    details: Dict[str, Any]
    

def normalize_text(text: str) -> Set[str]:
    """Normalize text to set of words for comparison"""
    return set(text.lower().split())


# =============================================================================
# CONTEXT RECALL
# =============================================================================

def context_recall(
    retrieved_facts: List[str],
    required_facts: List[str],
    use_semantic: bool = False
) -> MetricResult:
    """
    Mengukur berapa persen fakta yang dibutuhkan berhasil di-retrieve.
    
    Formula: |retrieved ∩ required| / |required|
    
    Args:
        retrieved_facts: List of facts retrieved by RAG
        required_facts: List of facts from ground truth
        use_semantic: Use semantic similarity instead of exact match
        
    Returns:
        MetricResult with score 0-1
    """
    if not required_facts:
        return MetricResult(
            name="context_recall",
            score=1.0,  # No requirements = perfect recall
            details={"message": "No required facts"}
        )
    
    if not retrieved_facts:
        return MetricResult(
            name="context_recall",
            score=0.0,
            details={"message": "No facts retrieved", "required_count": len(required_facts)}
        )
    
    if use_semantic:
        # TODO: Implement semantic similarity matching
        pass
    
    # Simple word overlap approach
    retrieved_words = set()
    for fact in retrieved_facts:
        retrieved_words.update(normalize_text(fact))
    
    matched_count = 0
    for req_fact in required_facts:
        req_words = normalize_text(req_fact)
        
        # Calculate overlap ratio
        if not req_words:
            continue
            
        overlap = len(req_words & retrieved_words) / len(req_words)
        if overlap > 0.5:  # Consider matched if >50% overlap
            matched_count += 1
    
    score = matched_count / len(required_facts)
    
    return MetricResult(
        name="context_recall",
        score=score,
        details={
            "matched": matched_count,
            "required": len(required_facts),
            "retrieved": len(retrieved_facts)
        }
    )


# =============================================================================
# CONTEXT PRECISION
# =============================================================================

def context_precision(
    retrieved_facts: List[str],
    relevant_facts: List[str]
) -> MetricResult:
    """
    Mengukur berapa persen fakta yang di-retrieve relevan.
    
    Formula: |retrieved ∩ relevant| / |retrieved|
    
    Args:
        retrieved_facts: List of facts retrieved by RAG
        relevant_facts: List of facts that are actually relevant
        
    Returns:
        MetricResult with score 0-1
    """
    if not retrieved_facts:
        return MetricResult(
            name="context_precision",
            score=0.0,
            details={"message": "No facts retrieved"}
        )
    
    if not relevant_facts:
        return MetricResult(
            name="context_precision",
            score=0.0,
            details={"message": "No relevant facts defined"}
        )
    
    # Build relevant word set
    relevant_words = set()
    for fact in relevant_facts:
        relevant_words.update(normalize_text(fact))
    
    relevant_count = 0
    for ret_fact in retrieved_facts:
        ret_words = normalize_text(ret_fact)
        
        if not ret_words:
            continue
            
        overlap = len(ret_words & relevant_words) / len(ret_words)
        if overlap > 0.3:  # Lower threshold for precision
            relevant_count += 1
    
    score = relevant_count / len(retrieved_facts)
    
    return MetricResult(
        name="context_precision",
        score=score,
        details={
            "relevant_retrieved": relevant_count,
            "total_retrieved": len(retrieved_facts)
        }
    )


# =============================================================================
# TEMPORAL PRECISION
# =============================================================================

def temporal_precision(
    predicted_order: List[str],
    actual_order: List[str]
) -> MetricResult:
    """
    Mengukur akurasi temporal ordering dalam fakta yang di-retrieve.
    
    Uses Kendall's Tau correlation coefficient.
    
    Args:
        predicted_order: Events/facts in predicted temporal order
        actual_order: Events/facts in actual temporal order
        
    Returns:
        MetricResult with Kendall's Tau score (-1 to 1, normalized to 0-1)
    """
    if len(predicted_order) < 2 or len(actual_order) < 2:
        return MetricResult(
            name="temporal_precision",
            score=1.0,
            details={"message": "Insufficient items for ordering comparison"}
        )
    
    # Find common elements
    common = set(predicted_order) & set(actual_order)
    
    if len(common) < 2:
        return MetricResult(
            name="temporal_precision",
            score=0.5,  # Neutral score
            details={"message": "Less than 2 common elements"}
        )
    
    # Get positions of common elements
    pred_positions = {item: i for i, item in enumerate(predicted_order) if item in common}
    actual_positions = {item: i for i, item in enumerate(actual_order) if item in common}
    
    # Calculate concordant and discordant pairs
    items = list(common)
    concordant = 0
    discordant = 0
    
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item_i, item_j = items[i], items[j]
            
            pred_diff = pred_positions[item_i] - pred_positions[item_j]
            actual_diff = actual_positions[item_i] - actual_positions[item_j]
            
            if (pred_diff > 0) == (actual_diff > 0):
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = concordant + discordant
    if total_pairs == 0:
        tau = 0.0
    else:
        tau = (concordant - discordant) / total_pairs
    
    # Normalize to 0-1 range
    score = (tau + 1) / 2
    
    return MetricResult(
        name="temporal_precision",
        score=score,
        details={
            "kendall_tau": tau,
            "concordant_pairs": concordant,
            "discordant_pairs": discordant,
            "common_items": len(common)
        }
    )


# =============================================================================
# TEMPORAL RECALL
# =============================================================================

def temporal_recall(
    retrieved_temporal_facts: List[Dict[str, Any]],
    ground_truth_temporal: List[Dict[str, Any]]
) -> MetricResult:
    """
    Mengukur berapa persen fakta temporal berhasil di-retrieve dengan benar.
    
    Args:
        retrieved_temporal_facts: Retrieved facts with temporal info
        ground_truth_temporal: Ground truth temporal facts
        
    Returns:
        MetricResult with score 0-1
    """
    if not ground_truth_temporal:
        return MetricResult(
            name="temporal_recall",
            score=1.0,
            details={"message": "No temporal ground truth"}
        )
    
    if not retrieved_temporal_facts:
        return MetricResult(
            name="temporal_recall",
            score=0.0,
            details={"message": "No temporal facts retrieved"}
        )
    
    # Match temporal facts
    matched = 0
    for gt in ground_truth_temporal:
        gt_text = gt.get('text', gt.get('value', ''))
        gt_type = gt.get('type', 'unknown')
        
        for ret in retrieved_temporal_facts:
            ret_text = str(ret)
            if gt_text.lower() in ret_text.lower():
                matched += 1
                break
    
    score = matched / len(ground_truth_temporal)
    
    return MetricResult(
        name="temporal_recall",
        score=score,
        details={
            "matched": matched,
            "total_temporal": len(ground_truth_temporal)
        }
    )


# =============================================================================
# FACT COVERAGE
# =============================================================================

def fact_coverage(
    retrieved_context: str,
    ground_truth_facts: List[Dict[str, Any]],
    use_embeddings: bool = False,
    embedder=None
) -> MetricResult:
    """
    Mengukur seberapa baik retrieved context mencakup ground truth facts.
    
    Args:
        retrieved_context: Combined retrieved text
        ground_truth_facts: List of ground truth fact objects
        use_embeddings: Use embedding similarity
        embedder: Embedder instance (required if use_embeddings=True)
        
    Returns:
        MetricResult with score 0-1
    """
    if not ground_truth_facts:
        return MetricResult(
            name="fact_coverage",
            score=1.0,
            details={"message": "No ground truth facts"}
        )
    
    if not retrieved_context:
        return MetricResult(
            name="fact_coverage",
            score=0.0,
            details={"message": "No context retrieved"}
        )
    
    retrieved_words = normalize_text(retrieved_context)
    
    covered_count = 0
    coverage_details = []
    
    for gt_fact in ground_truth_facts:
        fact_text = gt_fact.get('fact', '')
        if not fact_text:
            continue
            
        fact_words = normalize_text(fact_text)
        
        if not fact_words:
            continue
        
        # Calculate coverage
        overlap = len(fact_words & retrieved_words)
        coverage = overlap / len(fact_words)
        
        coverage_details.append({
            "fact": fact_text[:50] + "..." if len(fact_text) > 50 else fact_text,
            "coverage": coverage
        })
        
        if coverage > 0.5:
            covered_count += 1
    
    total_facts = len([f for f in ground_truth_facts if f.get('fact')])
    score = covered_count / total_facts if total_facts > 0 else 0.0
    
    return MetricResult(
        name="fact_coverage",
        score=score,
        details={
            "covered": covered_count,
            "total": total_facts,
            "fact_details": coverage_details[:5]  # Limit details
        }
    )


# =============================================================================
# HIT RATE (Binary metric)
# =============================================================================

def hit_rate(
    retrieved_facts: List[str],
    expected_answer: str,
    threshold: float = 0.5
) -> MetricResult:
    """
    Binary metric: apakah jawaban yang diharapkan ada di retrieved context?
    
    Formula: 1 if answer found in context else 0
    
    Args:
        retrieved_facts: List of retrieved facts
        expected_answer: Expected answer string
        threshold: Word overlap threshold to consider "found"
        
    Returns:
        MetricResult with score 0 or 1
    """
    if not expected_answer:
        return MetricResult(
            name="hit_rate",
            score=0.0,
            details={"message": "No expected answer to check (cannot determine hit)"}
        )
    
    if not retrieved_facts:
        return MetricResult(
            name="hit_rate",
            score=0.0,
            details={"message": "No facts retrieved", "answer_found": False}
        )
    
    expected_words = normalize_text(expected_answer)
    
    if not expected_words:
        return MetricResult(
            name="hit_rate",
            score=1.0,
            details={"message": "Empty expected answer"}
        )
    
    # Check each fact for answer presence
    best_overlap = 0.0
    best_fact_idx = -1
    
    for idx, fact in enumerate(retrieved_facts):
        fact_words = normalize_text(fact)
        if not fact_words:
            continue
            
        overlap = len(expected_words & fact_words) / len(expected_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_fact_idx = idx
    
    found = best_overlap >= threshold
    
    return MetricResult(
        name="hit_rate",
        score=1.0 if found else 0.0,
        details={
            "answer_found": found,
            "best_overlap": best_overlap,
            "best_fact_position": best_fact_idx + 1 if best_fact_idx >= 0 else None,
            "threshold": threshold
        }
    )


# =============================================================================
# MRR - MEAN RECIPROCAL RANK
# =============================================================================

def mrr(
    retrieved_facts: List[str],
    expected_answer: str,
    threshold: float = 0.5
) -> MetricResult:
    """
    Mean Reciprocal Rank: 1/rank of first relevant fact.
    
    Mengukur seberapa TINGGI posisi jawaban yang benar dalam hasil retrieval.
    
    Formula: 1 / rank_of_first_relevant_fact
    
    Args:
        retrieved_facts: List of retrieved facts (ORDERED by relevance)
        expected_answer: Expected answer string
        threshold: Word overlap threshold to consider "relevant"
        
    Returns:
        MetricResult with score 0-1
    """
    if not expected_answer:
        return MetricResult(
            name="mrr",
            score=0.0,
            details={"message": "No expected answer to check (cannot determine rank)"}
        )
    
    if not retrieved_facts:
        return MetricResult(
            name="mrr",
            score=0.0,
            details={"message": "No facts retrieved", "first_relevant_rank": None}
        )
    
    expected_words = normalize_text(expected_answer)
    
    if not expected_words:
        return MetricResult(
            name="mrr",
            score=1.0,
            details={"message": "Empty expected answer"}
        )
    
    # Find first relevant fact
    first_relevant_rank = None
    
    for rank, fact in enumerate(retrieved_facts, start=1):
        fact_words = normalize_text(fact)
        if not fact_words:
            continue
            
        overlap = len(expected_words & fact_words) / len(expected_words)
        if overlap >= threshold:
            first_relevant_rank = rank
            break
    
    if first_relevant_rank is None:
        score = 0.0
    else:
        score = 1.0 / first_relevant_rank
    
    return MetricResult(
        name="mrr",
        score=score,
        details={
            "first_relevant_rank": first_relevant_rank,
            "total_facts": len(retrieved_facts),
            "threshold": threshold
        }
    )


# =============================================================================
# CONTEXT SUFFICIENCY (LLM Judge)
# =============================================================================

# LLM Judge prompt for CONTEXT SUFFICIENCY evaluation
# This evaluates if the retrieved context is SUFFICIENT to answer the query
# NOT evaluating a generated answer (which doesn't exist in retrieval-only evaluation)
CONTEXT_SUFFICIENCY_PROMPT = """Anda adalah hakim yang mengevaluasi KECUKUPAN KONTEKS untuk menjawab pertanyaan.

TUGAS: Evaluasi apakah KONTEKS yang di-retrieve CUKUP untuk menjawab PERTANYAAN.

PERTANYAAN:
{query}

KONTEKS YANG DI-RETRIEVE:
{retrieved_context}

JAWABAN YANG DIHARAPKAN (Ground Truth):
{expected_answer}

KRITERIA PENILAIAN:
1. **Information Presence**: Apakah informasi yang dibutuhkan ADA di konteks?
2. **Completeness**: Apakah konteks LENGKAP untuk menjawab pertanyaan?
3. **Temporal Info**: Jika pertanyaan tentang waktu, apakah info temporal tersedia?
4. **No Contradiction**: Apakah konteks TIDAK bertentangan dengan expected answer?

SKALA PENILAIAN (0.0 - 1.0):
- 1.0: Sempurna - konteks memiliki SEMUA informasi untuk menjawab dengan benar
- 0.8-0.9: Sangat baik - konteks hampir lengkap, hanya minor detail yang kurang
- 0.6-0.7: Cukup - konteks memiliki informasi utama tapi tidak lengkap
- 0.4-0.5: Kurang - konteks memiliki sebagian informasi, perlu inferensi banyak
- 0.2-0.3: Sangat kurang - konteks hanya memiliki sedikit informasi relevan
- 0.0-0.1: Tidak cukup - konteks tidak memiliki informasi yang dibutuhkan

PENTING:
- Evaluasi KONTEKS, bukan jawaban yang di-generate
- Fokus pada: "Apakah seseorang BISA menjawab pertanyaan dengan konteks ini?"
- Jika expected answer adalah temporal (tanggal/waktu), pastikan info temporal ada

Berikan respons dalam format JSON:
{{
    "score": <float 0.0-1.0>,
    "information_presence": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "temporal_info": <float 0.0-1.0>,
    "no_contradiction": <float 0.0-1.0>,
    "reasoning": "<penjelasan singkat>",
    "missing_info": ["<info yang kurang 1>", "<info yang kurang 2>"]
}}
"""

# Legacy prompt for backward compatibility
LLM_JUDGE_PROMPT = CONTEXT_SUFFICIENCY_PROMPT


async def context_sufficiency_llm_judge(
    query: str,
    retrieved_context: str,
    expected_answer: str,
    judge_model: str = "gemini-2.5-pro"
) -> MetricResult:
    """
    Evaluasi KECUKUPAN KONTEKS menggunakan LLM sebagai hakim.
    
    Berbeda dengan answer_accuracy yang mengevaluasi jawaban yang di-generate,
    context_sufficiency mengevaluasi apakah KONTEKS yang di-retrieve CUKUP
    untuk menjawab pertanyaan.
    
    Ini lebih fair untuk evaluasi retrieval karena:
    - Tidak memerlukan answer generation step
    - Fokus pada kualitas retrieval, bukan generation
    - Semua setup dinilai dengan standar yang sama
    
    Args:
        query: Original query/question
        retrieved_context: Combined retrieved facts/context
        expected_answer: Expected answer from ground truth
        judge_model: Model to use as judge (default: gemini-2.5-pro)
        
    Returns:
        MetricResult with score 0-1 and detailed breakdown
    """
    import json
    import google.generativeai as genai
    from ..config import get_config
    
    if not retrieved_context:
        return MetricResult(
            name="context_sufficiency",
            score=0.0,
            details={"message": "No context retrieved", "method": "llm_judge"}
        )
    
    if not expected_answer:
        return MetricResult(
            name="context_sufficiency",
            score=0.5,
            details={"message": "No expected answer to compare", "method": "llm_judge"}
        )
    
    try:
        # Configure Gemini
        config = get_config()
        genai.configure(api_key=config.gemini.api_key)
        
        # Rate limiting for LLM Judge
        rate_limiter = get_rate_limiter()
        await rate_limiter.wait_if_needed(judge_model, estimated_tokens=2000)
        
        # Use gemini-2.5-pro as the judge
        model = genai.GenerativeModel(judge_model)
        
        # Format prompt
        prompt = CONTEXT_SUFFICIENCY_PROMPT.format(
            query=query,
            retrieved_context=retrieved_context,
            expected_answer=expected_answer
        )
        
        # Generate evaluation
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent evaluation
                response_mime_type="application/json"
            )
        )
        
        # Log token usage and record success
        if response.usage_metadata:
            log_token_usage(response.usage_metadata, judge_model)
            total_tokens = (
                response.usage_metadata.prompt_token_count + 
                response.usage_metadata.candidates_token_count
            )
            rate_limiter.record_success(judge_model, total_tokens)
            
            # Track Cost
            try:
                from ..utils.cost_tracker import get_cost_tracker
                tracker = get_cost_tracker()
                await tracker.track(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count,
                    judge_model
                )
            except ImportError:
                pass
        else:
            rate_limiter.record_success(judge_model)
        
        # Parse response
        result_text = response.text.strip()
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        
        return MetricResult(
            name="context_sufficiency",
            score=float(result.get("score", 0.0)),
            details={
                "method": "llm_judge",
                "judge_model": judge_model,
                "information_presence": result.get("information_presence", 0.0),
                "completeness": result.get("completeness", 0.0),
                "temporal_info": result.get("temporal_info", 0.0),
                "no_contradiction": result.get("no_contradiction", 0.0),
                "reasoning": result.get("reasoning", ""),
                "missing_info": result.get("missing_info", [])
            }
        )
        
    except RuntimeError as e:
        # Rate limit reached - propagate error
        logger.error(f"Rate limit error in LLM Judge: {e}")
        raise
        
    except Exception as e:
        logger.error(f"LLM Judge failed: {e}")
        # Record rate limit if it's a 429 error
        if "429" in str(e) or "ResourceExhausted" in str(e):
            rate_limiter = get_rate_limiter()
            rate_limiter.record_rate_limit_error(judge_model)
        
        # Fallback to simple word overlap method
        return context_sufficiency_simple(
            retrieved_context=retrieved_context,
            expected_answer=expected_answer
        )


def context_sufficiency_simple(
    retrieved_context: str,
    expected_answer: str
) -> MetricResult:
    """
    Simple fallback for context sufficiency using word overlap.
    Used when LLM Judge fails.
    """
    if not retrieved_context or not expected_answer:
        return MetricResult(
            name="context_sufficiency",
            score=0.0,
            details={"message": "Missing context or expected answer", "method": "word_overlap"}
        )
    
    context_words = normalize_text(retrieved_context)
    answer_words = normalize_text(expected_answer)
    
    if not answer_words:
        return MetricResult(
            name="context_sufficiency",
            score=1.0,
            details={"message": "Empty expected answer", "method": "word_overlap"}
        )
    
    overlap = len(context_words & answer_words) / len(answer_words)
    
    return MetricResult(
        name="context_sufficiency",
        score=overlap,
        details={
            "method": "word_overlap",
            "overlap_ratio": overlap,
            "answer_words": len(answer_words),
            "context_words": len(context_words)
        }
    )


# Legacy function for backward compatibility
async def answer_accuracy_llm_judge(
    generated_answer: str,
    expected_answer: str,
    question_context: str = "",
    llm_client=None,
    judge_model: str = "gemini-2.5-pro"
) -> MetricResult:
    """
    DEPRECATED: Use context_sufficiency_llm_judge instead.
    
    This function is kept for backward compatibility but now redirects
    to context_sufficiency_llm_judge for fair retrieval evaluation.
    """
    logger.warning(
        "answer_accuracy_llm_judge is deprecated. "
        "Use context_sufficiency_llm_judge for retrieval evaluation."
    )
    
    return await context_sufficiency_llm_judge(
        query=question_context,
        retrieved_context=generated_answer,  # In old code, this was the retrieved context
        expected_answer=expected_answer,
        judge_model=judge_model
    )


async def answer_accuracy(
    generated_answer: str,
    expected_answer: str,
    use_llm_judge: bool = True,
    llm_client=None,
    question_context: str = "",
    judge_model: str = "gemini-2.5-pro"
) -> MetricResult:
    """
    Mengukur akurasi jawaban yang di-generate.
    
    Args:
        generated_answer: Answer generated by RAG
        expected_answer: Expected answer from ground truth
        use_llm_judge: Use LLM for evaluation (recommended)
        llm_client: LLM client (optional, will use default if not provided)
        question_context: Original question for context
        judge_model: Model to use as judge (default: gemini-2.5-pro)
        
    Returns:
        MetricResult with score 0-1
    """
    if not generated_answer:
        return MetricResult(
            name="answer_accuracy",
            score=0.0,
            details={"message": "No answer generated"}
        )
    
    if not expected_answer:
        return MetricResult(
            name="answer_accuracy",
            score=0.5,
            details={"message": "No expected answer to compare"}
        )
    
    # Use LLM Judge if enabled
    if use_llm_judge:
        return await answer_accuracy_llm_judge(
            generated_answer=generated_answer,
            expected_answer=expected_answer,
            question_context=question_context,
            llm_client=llm_client,
            judge_model=judge_model
        )
    
    # Simple word overlap for now
    gen_words = normalize_text(generated_answer)
    exp_words = normalize_text(expected_answer)
    
    if not exp_words:
        return MetricResult(
            name="answer_accuracy",
            score=0.0,
            details={"message": "Expected answer is empty"}
        )
    
    overlap = len(gen_words & exp_words)
    
    # F1-style score
    precision = overlap / len(gen_words) if gen_words else 0
    recall = overlap / len(exp_words) if exp_words else 0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return MetricResult(
        name="answer_accuracy",
        score=f1,
        details={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "word_overlap": overlap
        }
    )


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

async def calculate_all_metrics(
    retrieved_facts: List[str],
    retrieved_context: str,
    ground_truth: Dict[str, Any],
    query: Optional[str] = None,
    expected_answer: Optional[str] = None,
    use_llm_judge: bool = True,
    generated_answer: Optional[str] = None,
    question_context: Optional[str] = None
) -> Dict[str, MetricResult]:
    """
    Calculate all evaluation metrics.
    
    FAIR COMPARISON: All metrics are normalized and independent of context size.
    
    Args:
        retrieved_facts: List of retrieved fact strings (ORDERED by relevance)
        retrieved_context: Combined context string
        ground_truth: Ground truth object with factual, entities, temporal info
        query: Original query/question for evaluation
        expected_answer: Expected answer from ground truth
        use_llm_judge: Whether to use LLM for context sufficiency evaluation
        generated_answer: (Optional) Generated answer or retrieved context used as answer
        question_context: (Optional) Alias for query
        
    Returns:
        Dict mapping metric name to MetricResult
    """
    # Handle aliases
    if query is None and question_context is not None:
        query = question_context
        
    results = {}
    
    # Extract ground truth components
    gt_facts = ground_truth.get('factual', [])
    gt_temporal = ground_truth.get('temporal_references', [])
    retrieval_required = ground_truth.get('retrieval_required', [])
    
    # Required facts from retrieval_required
    required_facts = [req.get('description', '') for req in retrieval_required]
    
    # 1. Context Recall (normalized: required facts found / total required)
    results['context_recall'] = context_recall(
        retrieved_facts=retrieved_facts,
        required_facts=required_facts
    )
    
    # 2. Context Precision (normalized: relevant facts / total retrieved)
    relevant_facts = [f.get('fact', '') for f in gt_facts]
    results['context_precision'] = context_precision(
        retrieved_facts=retrieved_facts,
        relevant_facts=relevant_facts
    )
    
    # 3. Hit Rate (binary: is answer in context?)
    if expected_answer:
        results['hit_rate'] = hit_rate(
            retrieved_facts=retrieved_facts,
            expected_answer=expected_answer
        )
    
    # 4. MRR - Mean Reciprocal Rank (position of correct answer)
    if expected_answer:
        results['mrr'] = mrr(
            retrieved_facts=retrieved_facts,
            expected_answer=expected_answer
        )
    
    # 5. Temporal Recall
    results['temporal_recall'] = temporal_recall(
        retrieved_temporal_facts=[{"text": f} for f in retrieved_facts],
        ground_truth_temporal=gt_temporal
    )
    
    # 6. Context Sufficiency (LLM Judge: is context sufficient to answer?)
    if use_llm_judge and query and expected_answer:
        results['context_sufficiency'] = await context_sufficiency_llm_judge(
            query=query,
            retrieved_context=retrieved_context,
            expected_answer=expected_answer
        )
    elif expected_answer:
        # Fallback to simple word overlap
        results['context_sufficiency'] = context_sufficiency_simple(
            retrieved_context=retrieved_context,
            expected_answer=expected_answer
        )
    
    # Legacy: Fact Coverage (for backward compatibility)
    results['fact_coverage'] = fact_coverage(
        retrieved_context=retrieved_context,
        ground_truth_facts=gt_facts
    )
    
    return results


async def calculate_retrieval_metrics(
    retrieved_facts: List[str],
    expected_answer: str,
    query: str,
    use_llm_judge: bool = True
) -> Dict[str, MetricResult]:
    """
    Calculate ONLY retrieval-focused metrics for fair comparison.
    
    This is the recommended function for comparing Vanilla vs Agentic RAG.
    
    Args:
        retrieved_facts: List of retrieved facts (ORDERED by relevance)
        expected_answer: Expected answer from ground truth
        query: Original query
        use_llm_judge: Use LLM for context sufficiency
        
    Returns:
        Dict with hit_rate, mrr, context_sufficiency
    """
    results = {}
    
    # Combined context
    retrieved_context = "\n".join(retrieved_facts)
    
    # 1. Hit Rate (binary)
    results['hit_rate'] = hit_rate(
        retrieved_facts=retrieved_facts,
        expected_answer=expected_answer
    )
    
    # 2. MRR
    results['mrr'] = mrr(
        retrieved_facts=retrieved_facts,
        expected_answer=expected_answer
    )
    
    # 3. Context Sufficiency
    if use_llm_judge:
        results['context_sufficiency'] = await context_sufficiency_llm_judge(
            query=query,
            retrieved_context=retrieved_context,
            expected_answer=expected_answer
        )
    else:
        results['context_sufficiency'] = context_sufficiency_simple(
            retrieved_context=retrieved_context,
            expected_answer=expected_answer
        )
    
    return results


if __name__ == "__main__":
    import asyncio
    
    async def test_metrics():
        """Test evaluation metrics"""
        print("Testing evaluation metrics...")
        print("=" * 60)
        
        # Sample data - simulating retrieval results
        retrieved = [
            "Aisha bekerja sebagai content creator di Bandung",
            "Dewi adalah partner bisnis Aisha",
            "Project skincare dimulai awal Januari 2024"
        ]
        
        expected_answer = "Project skincare dimulai awal Januari 2024"
        query = "Kapan project skincare dimulai?"
        
        print(f"\nQuery: {query}")
        print(f"Expected Answer: {expected_answer}")
        print(f"\nRetrieved Facts ({len(retrieved)}):")
        for i, fact in enumerate(retrieved, 1):
            print(f"  {i}. {fact}")
        
        # Test individual metrics
        print("\n" + "-" * 60)
        print("INDIVIDUAL METRICS:")
        print("-" * 60)
        
        # Hit Rate
        hr = hit_rate(retrieved, expected_answer)
        print(f"\n1. Hit Rate: {hr.score:.2f}")
        print(f"   Details: {hr.details}")
        
        # MRR
        mrr_result = mrr(retrieved, expected_answer)
        print(f"\n2. MRR: {mrr_result.score:.4f}")
        print(f"   Details: {mrr_result.details}")
        
        # Context Sufficiency (simple - no LLM)
        cs = context_sufficiency_simple(
            retrieved_context="\n".join(retrieved),
            expected_answer=expected_answer
        )
        print(f"\n3. Context Sufficiency (simple): {cs.score:.4f}")
        print(f"   Details: {cs.details}")
        
        # Full metrics test
        print("\n" + "-" * 60)
        print("FULL METRICS (calculate_retrieval_metrics):")
        print("-" * 60)
        
        results = await calculate_retrieval_metrics(
            retrieved_facts=retrieved,
            expected_answer=expected_answer,
            query=query,
            use_llm_judge=False  # Don't use LLM for quick test
        )
        
        print("\nResults:")
        for name, result in results.items():
            print(f"  {name}: {result.score:.4f}")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
    
    asyncio.run(test_metrics())
