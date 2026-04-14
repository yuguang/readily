"""
Unit tests for backend/models/schemas.py

Covers:
- Field validation rules (confidence bounds, enum values, status literals)
- requirement_id cross-reference validation inside ReviewSession
- Serialization / deserialization roundtrip for every model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.models.schemas import (
    AnswerType,
    BulkApproveRequest,
    ComplianceRequirement,
    Evaluation,
    Passage,
    Requirement,
    ReviewSession,
    SSEEvent,
    StartReviewRequest,
    UpdateEvaluationRequest,
    UploadResponse,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_requirement(id: int = 1) -> Requirement:
    return Requirement(id=id, text="Does the P&P state that…", reference="APL 25-008, page 2")


def make_evaluation(requirement_id: int = 1, confidence: float = 0.9) -> Evaluation:
    return Evaluation(
        requirement_id=requirement_id,
        answer=AnswerType.YES,
        citation_text="The policy states…",
        source_file="GG/GG.1508.pdf",
        page_number=3,
        confidence=confidence,
        reasoning="Found explicit mention.",
    )


def make_session(extra_evaluations: list[Evaluation] | None = None) -> ReviewSession:
    req = make_requirement(1)
    evals = [make_evaluation(1)] + (extra_evaluations or [])
    return ReviewSession(
        id="abc-123",
        filename="review.pdf",
        doc_type="structured",
        requirements=[req],
        evaluations=evals,
        created_at=NOW,
    )


# ---------------------------------------------------------------------------
# AnswerType
# ---------------------------------------------------------------------------


class TestAnswerType:
    def test_valid_values(self):
        assert AnswerType("yes") is AnswerType.YES
        assert AnswerType("no") is AnswerType.NO
        assert AnswerType("partial") is AnswerType.PARTIAL

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            AnswerType("maybe")

    def test_is_str_subclass(self):
        assert isinstance(AnswerType.YES, str)


# ---------------------------------------------------------------------------
# Requirement
# ---------------------------------------------------------------------------


class TestRequirement:
    def test_minimal_valid(self):
        r = Requirement(id=1, text="Question?")
        assert r.reference is None
        assert r.category is None

    def test_full_fields(self):
        r = Requirement(id=2, text="Q?", reference="APL 1-001, p1", category="Access")
        assert r.reference == "APL 1-001, p1"
        assert r.category == "Access"

    def test_roundtrip(self):
        r = Requirement(id=1, text="Q?", reference="ref", category="cat")
        assert Requirement.model_validate(json.loads(r.model_dump_json())) == r


# ---------------------------------------------------------------------------
# ComplianceRequirement
# ---------------------------------------------------------------------------


class TestComplianceRequirement:
    def test_is_requirement_subclass(self):
        cr = ComplianceRequirement(id=1, text="Q?")
        assert isinstance(cr, Requirement)

    def test_all_optional_fields_default_to_none(self):
        cr = ComplianceRequirement(id=1, text="Q?")
        for field in (
            "obligation_type", "obligation_level", "actor", "action_required",
            "condition", "timeframe", "evidence_needed", "risk_area",
            "parent_id", "exact_quote", "section_heading",
        ):
            assert getattr(cr, field) is None

    def test_inherits_base_fields(self):
        cr = ComplianceRequirement(id=2, text="Q?", reference="APL 1", category="Privacy")
        assert cr.reference == "APL 1"
        assert cr.category == "Privacy"

    def test_full_fields(self):
        cr = ComplianceRequirement(
            id=3,
            text="Does the P&P state that…",
            obligation_type="mandatory",
            obligation_level="conditional_mandatory",
            actor="Compliance Officer",
            action_required="submit annual report",
            condition="if PHI is involved",
            timeframe="within 30 days",
            evidence_needed="audit logs",
            risk_area="Privacy",
            parent_id=1,
            exact_quote="The plan shall…",
            section_heading="Section 4.2",
        )
        assert cr.obligation_type == "mandatory"
        assert cr.parent_id == 1
        assert cr.risk_area == "Privacy"

    def test_roundtrip(self):
        cr = ComplianceRequirement(
            id=1,
            text="Q?",
            obligation_type="prohibition",
            actor="MCP",
            timeframe="annually",
        )
        assert ComplianceRequirement.model_validate(json.loads(cr.model_dump_json())) == cr

    def test_backward_compatible_with_base_fields_only(self):
        """A plain Requirement dict can be validated as ComplianceRequirement."""
        data = {"id": 1, "text": "Q?", "reference": None, "category": None}
        cr = ComplianceRequirement.model_validate(data)
        assert cr.id == 1
        assert cr.obligation_type is None


# ---------------------------------------------------------------------------
# Passage
# ---------------------------------------------------------------------------


class TestPassage:
    def test_valid(self):
        p = Passage(text="chunk", source_file="file.pdf", page_number=1, relevance_score=0.85)
        assert p.relevance_score == 0.85

    def test_roundtrip(self):
        p = Passage(text="t", source_file="f.pdf", page_number=2, relevance_score=0.5)
        assert Passage.model_validate(json.loads(p.model_dump_json())) == p


# ---------------------------------------------------------------------------
# Evaluation — confidence validation
# ---------------------------------------------------------------------------


class TestEvaluationConfidence:
    def test_confidence_lower_bound(self):
        e = make_evaluation(confidence=0.0)
        assert e.confidence == 0.0

    def test_confidence_upper_bound(self):
        e = make_evaluation(confidence=1.0)
        assert e.confidence == 1.0

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            make_evaluation(confidence=-0.01)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            make_evaluation(confidence=1.001)


# ---------------------------------------------------------------------------
# Evaluation — status validation
# ---------------------------------------------------------------------------


class TestEvaluationStatus:
    @pytest.mark.parametrize("status", ["pending", "approved", "edited", "rejected"])
    def test_valid_statuses(self, status):
        e = make_evaluation()
        e2 = e.model_copy(update={"status": status})
        assert e2.status == status

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            Evaluation(
                requirement_id=1,
                answer=AnswerType.YES,
                citation_text="c",
                source_file="f.pdf",
                page_number=1,
                confidence=0.5,
                reasoning="r",
                status="unknown",
            )

    def test_default_status_is_pending(self):
        e = make_evaluation()
        assert e.status == "pending"


# ---------------------------------------------------------------------------
# Evaluation — answer enum
# ---------------------------------------------------------------------------


class TestEvaluationAnswer:
    @pytest.mark.parametrize("answer", [AnswerType.YES, AnswerType.NO, AnswerType.PARTIAL])
    def test_valid_answers(self, answer):
        e = make_evaluation()
        e2 = e.model_copy(update={"answer": answer})
        assert e2.answer == answer

    def test_invalid_answer_raises(self):
        with pytest.raises(ValidationError):
            Evaluation(
                requirement_id=1,
                answer="maybe",
                citation_text="c",
                source_file="f.pdf",
                page_number=1,
                confidence=0.5,
                reasoning="r",
            )


# ---------------------------------------------------------------------------
# Evaluation — roundtrip
# ---------------------------------------------------------------------------


class TestEvaluationRoundtrip:
    def test_roundtrip(self):
        e = make_evaluation()
        assert Evaluation.model_validate(json.loads(e.model_dump_json())) == e


# ---------------------------------------------------------------------------
# ReviewSession — requirement_id cross-reference
# ---------------------------------------------------------------------------


class TestReviewSessionRequirementIds:
    def test_valid_session(self):
        s = make_session()
        assert len(s.evaluations) == 1

    def test_invalid_requirement_id_raises(self):
        """Evaluation with a requirement_id that has no matching Requirement."""
        req = make_requirement(1)
        bad_eval = make_evaluation(requirement_id=99)
        with pytest.raises(ValidationError, match="requirement_id"):
            ReviewSession(
                id="x",
                filename="f.pdf",
                doc_type="structured",
                requirements=[req],
                evaluations=[bad_eval],
                created_at=NOW,
            )

    def test_empty_evaluations_is_valid(self):
        s = ReviewSession(
            id="x",
            filename="f.pdf",
            doc_type="structured",
            requirements=[make_requirement(1)],
            created_at=NOW,
        )
        assert s.evaluations == []

    def test_multiple_requirements_and_evaluations(self):
        reqs = [make_requirement(i) for i in range(1, 4)]
        evals = [make_evaluation(i) for i in range(1, 4)]
        s = ReviewSession(
            id="x",
            filename="f.pdf",
            doc_type="structured",
            requirements=reqs,
            evaluations=evals,
            created_at=NOW,
        )
        assert len(s.evaluations) == 3


# ---------------------------------------------------------------------------
# ReviewSession — status validation
# ---------------------------------------------------------------------------


class TestReviewSessionStatus:
    @pytest.mark.parametrize("status", ["extracting", "reviewing", "critic", "complete"])
    def test_valid_statuses(self, status):
        s = ReviewSession(
            id="x",
            filename="f.pdf",
            doc_type="structured",
            requirements=[make_requirement(1)],
            created_at=NOW,
            status=status,
        )
        assert s.status == status

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            ReviewSession(
                id="x",
                filename="f.pdf",
                doc_type="structured",
                requirements=[make_requirement(1)],
                created_at=NOW,
                status="done",
            )

    def test_default_status_is_extracting(self):
        s = make_session()
        assert s.status == "extracting"


# ---------------------------------------------------------------------------
# ReviewSession — roundtrip
# ---------------------------------------------------------------------------


class TestReviewSessionRoundtrip:
    def test_roundtrip(self):
        s = make_session()
        assert ReviewSession.model_validate(json.loads(s.model_dump_json())) == s


# ---------------------------------------------------------------------------
# SSEEvent
# ---------------------------------------------------------------------------


class TestSSEEvent:
    @pytest.mark.parametrize("event", ["evaluation", "progress", "critic_complete", "error"])
    def test_valid_events(self, event):
        e = SSEEvent(event=event, data={"key": "value"})
        assert e.event == event

    def test_invalid_event_raises(self):
        with pytest.raises(ValidationError):
            SSEEvent(event="unknown", data={})

    def test_roundtrip(self):
        e = SSEEvent(event="evaluation", data={"foo": 1})
        assert SSEEvent.model_validate(json.loads(e.model_dump_json())) == e


# ---------------------------------------------------------------------------
# API request/response types
# ---------------------------------------------------------------------------


class TestUploadResponse:
    def test_complete_status(self):
        r = UploadResponse(
            session_id="s1",
            filename="f.pdf",
            doc_type="structured",
            extraction_status="complete",
            requirements=[make_requirement(1)],
        )
        assert r.extraction_status == "complete"
        assert len(r.requirements) == 1

    def test_processing_status_with_empty_requirements(self):
        """Long docs may return processing status with no requirements yet."""
        r = UploadResponse(
            session_id="s2",
            filename="long.pdf",
            doc_type="narrative",
            extraction_status="processing",
            requirements=[],
        )
        assert r.extraction_status == "processing"
        assert r.requirements == []

    def test_invalid_extraction_status_raises(self):
        with pytest.raises(ValidationError):
            UploadResponse(
                session_id="s1",
                filename="f.pdf",
                doc_type="structured",
                extraction_status="unknown",
                requirements=[],
            )

    def test_roundtrip(self):
        r = UploadResponse(
            session_id="s1",
            filename="f.pdf",
            doc_type="narrative",
            extraction_status="complete",
            requirements=[make_requirement(1)],
        )
        assert UploadResponse.model_validate(json.loads(r.model_dump_json())) == r


class TestStartReviewRequest:
    def test_valid(self):
        r = StartReviewRequest(session_id="abc")
        assert r.session_id == "abc"

    def test_roundtrip(self):
        r = StartReviewRequest(session_id="abc")
        assert StartReviewRequest.model_validate(json.loads(r.model_dump_json())) == r


class TestUpdateEvaluationRequest:
    def test_all_none_is_valid(self):
        r = UpdateEvaluationRequest()
        assert r.answer is None
        assert r.status is None

    def test_valid_status_values(self):
        for status in ("approved", "edited", "rejected"):
            r = UpdateEvaluationRequest(status=status)
            assert r.status == status

    def test_pending_not_allowed_in_update(self):
        """'pending' is not a valid update status — only approved/edited/rejected."""
        with pytest.raises(ValidationError):
            UpdateEvaluationRequest(status="pending")

    def test_roundtrip(self):
        r = UpdateEvaluationRequest(answer=AnswerType.NO, reviewer_notes="looks good")
        assert UpdateEvaluationRequest.model_validate(json.loads(r.model_dump_json())) == r


class TestBulkApproveRequest:
    def test_valid(self):
        r = BulkApproveRequest(requirement_ids=[1, 2, 3])
        assert r.requirement_ids == [1, 2, 3]

    def test_empty_list(self):
        r = BulkApproveRequest(requirement_ids=[])
        assert r.requirement_ids == []

    def test_roundtrip(self):
        r = BulkApproveRequest(requirement_ids=[1, 2])
        assert BulkApproveRequest.model_validate(json.loads(r.model_dump_json())) == r
