# src/evaluation/query_schema.py
"""
Pydantic Schema untuk Evaluation Queries
=========================================
Memastikan format konsisten untuk semua 100 evaluation queries.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class QueryType(str, Enum):
    """Jenis pertanyaan evaluasi"""

    FACTUAL_RECALL = "factual_recall"
    INFERENCE = "inference"
    MULTI_HOP = "multi_hop"
    TEMPORAL_REASONING = "temporal_reasoning"
    COUNTERFACTUAL = "counterfactual"


class Difficulty(str, Enum):
    """Tingkat kesulitan pertanyaan"""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationQuery(BaseModel):
    """Schema untuk satu evaluation query"""

    id: str = Field(
        ...,
        description="Unique identifier (format: q001, q002, ...)",
        pattern=r"^q\d{3}$",
    )

    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Pertanyaan evaluasi dalam bahasa Indonesia",
    )

    type: QueryType = Field(..., description="Jenis pertanyaan")

    difficulty: Difficulty = Field(..., description="Tingkat kesulitan")

    expected_answer: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Jawaban yang diharapkan berdasarkan percakapan",
    )

    required_context: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Konteks dari percakapan yang dibutuhkan untuk menjawab",
    )

    reasoning_steps: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Langkah-langkah penalaran untuk sampai ke jawaban",
    )

    relevant_sessions: List[int] = Field(
        ..., min_length=1, description="List session IDs yang relevan (1-100)"
    )

    is_cross_session: bool = Field(
        default=False,
        description="True jika query membutuhkan informasi dari 2+ sessions",
    )

    source: Literal["original_dataset", "auto_generated", "manual"] = Field(
        default="auto_generated", description="Sumber query"
    )

    @field_validator("relevant_sessions")
    @classmethod
    def validate_sessions(cls, v):
        """Validate session IDs are in range 1-100"""
        for sid in v:
            if not 1 <= sid <= 100:
                raise ValueError(f"Session ID must be between 1-100, got {sid}")
        return v

    @field_validator("is_cross_session", mode="before")
    @classmethod
    def auto_detect_cross_session(cls, v, info):
        """Auto-detect cross-session based on relevant_sessions"""
        if "relevant_sessions" in info.data:
            return len(info.data["relevant_sessions"]) > 1
        return v

    class Config:
        use_enum_values = True


class EvaluationQuerySet(BaseModel):
    """Schema untuk kumpulan evaluation queries"""

    metadata: dict = Field(
        default_factory=dict, description="Metadata tentang query set"
    )

    queries: List[EvaluationQuery] = Field(
        ..., min_length=1, description="List of evaluation queries"
    )

    def get_by_session(self, session_id: int) -> List[EvaluationQuery]:
        """Get queries for a specific session"""
        return [q for q in self.queries if session_id in q.relevant_sessions]

    def get_cross_session_queries(self) -> List[EvaluationQuery]:
        """Get all cross-session queries"""
        return [q for q in self.queries if q.is_cross_session]

    def get_by_type(self, query_type: QueryType) -> List[EvaluationQuery]:
        """Get queries by type"""
        return [q for q in self.queries if q.type == query_type]

    def get_by_difficulty(self, difficulty: Difficulty) -> List[EvaluationQuery]:
        """Get queries by difficulty"""
        return [q for q in self.queries if q.difficulty == difficulty]

    def get_stats(self) -> dict:
        """Get statistics about the query set"""
        by_type = {}
        by_difficulty = {}
        by_session = {}
        cross_session_count = 0

        for q in self.queries:
            # By type
            t = q.type
            by_type[t] = by_type.get(t, 0) + 1

            # By difficulty
            d = q.difficulty
            by_difficulty[d] = by_difficulty.get(d, 0) + 1

            # By session
            for sid in q.relevant_sessions:
                by_session[sid] = by_session.get(sid, 0) + 1

            # Cross-session
            if q.is_cross_session:
                cross_session_count += 1

        return {
            "total_queries": len(self.queries),
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "sessions_covered": len(by_session),
            "cross_session_count": cross_session_count,
            "cross_session_percent": cross_session_count / len(self.queries) * 100
            if self.queries
            else 0,
        }


# JSON Schema untuk LLM output (tanpa id, akan di-assign otomatis)
class LLMGeneratedQuery(BaseModel):
    """Schema untuk query yang di-generate LLM (tanpa id)"""

    query: str = Field(..., description="Pertanyaan evaluasi")
    type: QueryType = Field(..., description="Jenis pertanyaan")
    difficulty: Difficulty = Field(..., description="Tingkat kesulitan")
    expected_answer: str = Field(..., description="Jawaban yang diharapkan")
    required_context: str = Field(..., description="Konteks yang dibutuhkan")
    reasoning_steps: str = Field(..., description="Langkah penalaran")
    relevant_sessions: List[int] = Field(..., description="Session IDs yang relevan")
    is_cross_session: bool = Field(default=False, description="Apakah cross-session")

    class Config:
        use_enum_values = True


class LLMQueryResponse(BaseModel):
    """Schema untuk response dari LLM"""

    queries: List[LLMGeneratedQuery]


def convert_to_evaluation_query(
    llm_query: LLMGeneratedQuery, query_id: str, source: str = "auto_generated"
) -> EvaluationQuery:
    """Convert LLM-generated query to full EvaluationQuery"""
    return EvaluationQuery(
        id=query_id,
        query=llm_query.query,
        type=llm_query.type,
        difficulty=llm_query.difficulty,
        expected_answer=llm_query.expected_answer,
        required_context=llm_query.required_context,
        reasoning_steps=llm_query.reasoning_steps,
        relevant_sessions=llm_query.relevant_sessions,
        is_cross_session=len(llm_query.relevant_sessions) > 1,
        source=source,  # type: ignore[invalid-argument-type]
    )


if __name__ == "__main__":
    # Test schema
    sample = EvaluationQuery(
        id="q001",
        query="Apa tantangan terbesar Aisha dalam campaign skincare?",
        type=QueryType.INFERENCE,
        difficulty=Difficulty.HARD,
        expected_answer="Tantangan terbesar adalah menentukan visual dan tone of voice.",
        required_context="Campaign skincare dengan target market spesifik.",
        reasoning_steps="1. Identifikasi tantangan. 2. Hubungkan dengan konteks.",
        relevant_sessions=[1],
        is_cross_session=False,
        source="original_dataset",
    )

    print("Sample query:")
    print(sample.model_dump_json(indent=2))
