"""
Shared Pydantic data models — the API contract between all components.
Every other module imports from here; changes require cross-component coordination.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AnswerType(str, Enum):
    YES = "yes"
    NO = "no"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class Requirement(BaseModel):
    """A single compliance question extracted from a review form."""

    id: int  # 1-indexed position in the review form
    text: str  # Full question text, e.g. "Does the P&P state that..."
    reference: str | None = None  # APL reference, e.g. "APL 25-008, page 2"
    category: str | None = None  # Optional grouping (for narrative docs)


class Passage(BaseModel):
    """A chunk retrieved from the policy vector store."""

    text: str  # The chunk text
    source_file: str  # e.g. "GG/GG.1508_CEO20250129_v20241231.pdf"
    page_number: int  # 1-indexed page within the PDF
    relevance_score: float  # Cosine similarity or re-rank score (0–1)


class Evaluation(BaseModel):
    """The agent's answer for one requirement — the core output of Phase 3."""

    requirement_id: int
    answer: AnswerType
    citation_text: str  # Exact excerpt from the policy doc
    source_file: str  # Which policy PDF
    page_number: int  # Page within that PDF
    confidence: float = Field(..., ge=0.0, le=1.0)  # 0.0 – 1.0
    reasoning: str  # Why the agent chose this answer
    needs_human_review: bool = False  # Flagged by critic or low confidence
    status: Literal["pending", "approved", "edited", "rejected"] = "pending"
    reviewer_notes: str = ""  # Human reviewer's notes


class ReviewSession(BaseModel):
    """Groups all evaluations for one uploaded review form."""

    id: str  # UUID
    filename: str  # Original uploaded filename
    doc_type: str  # "structured" or "narrative"
    requirements: list[Requirement]
    evaluations: list[Evaluation] = []
    created_at: datetime
    status: Literal["extracting", "reviewing", "critic", "complete"] = "extracting"
    progress: int = 0  # Number of evaluations completed so far

    @model_validator(mode="after")
    def _validate_evaluation_requirement_ids(self) -> ReviewSession:
        valid_ids = {r.id for r in self.requirements}
        for evaluation in self.evaluations:
            if evaluation.requirement_id not in valid_ids:
                raise ValueError(
                    f"Evaluation references unknown requirement_id "
                    f"{evaluation.requirement_id!r}. "
                    f"Valid ids: {sorted(valid_ids)}"
                )
        return self


class SSEEvent(BaseModel):
    """Wrapper for server-sent events streamed to the frontend."""

    event: Literal["evaluation", "progress", "critic_complete", "error"]
    data: dict[str, Any]  # Payload (usually a serialized Evaluation)


# ---------------------------------------------------------------------------
# API request / response types
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    doc_type: str
    requirements: list[Requirement]


class StartReviewRequest(BaseModel):
    session_id: str


class UpdateEvaluationRequest(BaseModel):
    answer: AnswerType | None = None
    citation_text: str | None = None
    reviewer_notes: str | None = None
    status: Literal["approved", "edited", "rejected"] | None = None


class BulkApproveRequest(BaseModel):
    requirement_ids: list[int]
