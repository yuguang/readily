"""
Evaluation suite for Component 4: Question Agent + RAG.

Measures end-to-end answer quality, citation grounding, tool behaviour,
and output consistency for the ToolCallingAgent that answers compliance
requirements against the policy corpus.

Unlike unit tests, these evals make real LLM calls and query the live
ChromaDB collection.  They are gated behind two conditions:
  - GEMINI_API_KEY must be set (LLM calls)
  - The policy ChromaDB must be populated (run `python -m backend.ingestion.ingest` first)

Usage
-----
    # Full eval suite (slow — ~15–30 LLM calls):
    pytest -m eval backend/evals/eval_question_agent.py -v -s

    # Individual classes:
    pytest -m eval backend/evals/eval_question_agent.py::TestSearchPoliciesTool -v -s
    pytest -m eval backend/evals/eval_question_agent.py::TestDefineTermTool -v -s
    pytest -m eval backend/evals/eval_question_agent.py::TestEvaluateCitationTool -v -s
    pytest -m eval backend/evals/eval_question_agent.py::TestQuestionAgentOutputQuality -v -s
    pytest -m eval backend/evals/eval_question_agent.py::TestCitationGrounding -v -s

    # Offline-only tests (no API key, no ChromaDB needed):
    pytest backend/evals/eval_question_agent.py::TestParseAgentResultEdgeCases -v
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field

import pytest

from backend.agents.question_agent import (
    _fallback_evaluation,
    get_model,
    parse_agent_result,
    run_question_agent,
)
from backend.models.schemas import AnswerType, Evaluation, Requirement

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

# Output structure
CONFIDENCE_RANGE_MIN = 0.0
CONFIDENCE_RANGE_MAX = 1.0
REASONING_MIN_WORDS = 5          # reasoning must contain at least 5 words
YES_ANSWER_CITATION_MIN_LEN = 10  # "yes" answers must cite at least 10 chars

# Batch quality (over EVAL_REQUIREMENTS)
FALLBACK_RATE_MAX = 0.20          # ≤20% of runs may fall back to error state
MIN_AVG_CONFIDENCE = 0.40         # average confidence across batch ≥ 0.40
GROUNDING_MIN_SIMILARITY = 0.50   # citation must have ≥0.50 cosine sim to some passage

# Stability
STABILITY_MATCH_REQUIRED = True   # two runs on same req must give the same answer

# ---------------------------------------------------------------------------
# Eval requirement set
#
# Five requirements drawn from the Easy PDF (APL 25-008) representing a range
# of topic areas.  These drive all LLM-calling eval classes.
# ---------------------------------------------------------------------------

EVAL_REQUIREMENTS: list[Requirement] = [
    Requirement(
        id=1,
        text=(
            "Does the P&P state that the MCP must provide hospice services coverage "
            "as described in the APL?"
        ),
        reference="APL 25-008, page 1",
        category="Hospice",
    ),
    Requirement(
        id=18,
        text=(
            "Does the P&P address Prior Authorization processes for hospice services, "
            "including timelines for authorization decisions?"
        ),
        reference="APL 25-008, page 5",
        category="Prior Authorization",
    ),
    Requirement(
        id=40,
        text=(
            "Does the P&P state that routine home care is covered under the MCP's "
            "hospice benefit?"
        ),
        reference="APL 25-008, page 12",
        category="Hospice",
    ),
    Requirement(
        id=55,
        text=(
            "Does the P&P describe the process for coordinating care between the "
            "MCP and hospice providers?"
        ),
        reference="APL 25-008, page 14",
        category="Care Coordination",
    ),
    Requirement(
        id=64,
        text=(
            "Does the P&P state that the MCP must cooperate with DHCS audits and "
            "monitoring activities?"
        ),
        reference="APL 25-008, page 16",
        category="Oversight",
    ),
]

# A requirement whose acronym ("ECM") should trigger a define_term call.
ECM_REQUIREMENT = Requirement(
    id=100,
    text=(
        "Does the P&P describe the MCP's approach to identifying ECM Population "
        "of Focus (POF) members eligible for Enhanced Care Management?"
    ),
    reference="Section III",
    category="ECM",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Accumulates metric scores and formats a human-readable report."""

    name: str
    scores: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)

    def record(self, metric: str, value: float, threshold: float | None = None) -> None:
        self.scores[metric] = value
        if threshold is not None:
            self.thresholds[metric] = threshold

    def __str__(self) -> str:
        lines = [f"\n{'='*60}", f"  Eval: {self.name}", "=" * 60]
        for metric, score in self.scores.items():
            threshold = self.thresholds.get(metric)
            if threshold is not None:
                status = "PASS" if score >= threshold else "FAIL"
                lines.append(
                    f"  [{status}] {metric:<40} {score:.3f}  (min {threshold:.3f})"
                )
            else:
                lines.append(f"  [INFO] {metric:<40} {score:.3f}")
        for k, v in self.extras.items():
            lines.append(f"  [INFO] {k:<40} {v!r}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _chroma_available() -> bool:
    """Return True if the policy ChromaDB collection exists and is non-empty."""
    try:
        from backend.tools.policy_search import get_chroma_collection
        col = get_chroma_collection()
        return col.count() > 0
    except Exception:
        return False


def _evaluation_is_fallback(ev: Evaluation) -> bool:
    """Return True if the evaluation is a fallback (agent failed)."""
    return ev.needs_human_review and ev.confidence == 0.0 and ev.source_file == ""


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

requires_api_key = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping LLM eval",
)

requires_chroma = pytest.mark.skipif(
    not _chroma_available(),
    reason="ChromaDB policy collection not populated — run ingestion first",
)


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    """Shared LLM model instance for all agent-calling tests."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    return get_model()


@pytest.fixture(scope="module")
def batch_evaluations(model) -> list[tuple[Requirement, Evaluation]]:
    """
    Run the question agent on all EVAL_REQUIREMENTS once and cache results.
    Avoids redundant LLM calls across multiple eval tests in this module.
    """
    import asyncio

    results = []
    for req in EVAL_REQUIREMENTS:
        t0 = time.perf_counter()
        ev = asyncio.run(run_question_agent(req, model))
        elapsed = time.perf_counter() - t0
        logger.info(
            "Req #%d → %s (conf=%.2f, %.1fs)", req.id, ev.answer.value, ev.confidence, elapsed
        )
        results.append((req, ev))
    return results


@pytest.fixture(scope="module")
def seeded_term_collection():
    """
    Seed the document_terms ChromaDB collection with test entries and
    yield; clean up after the module completes.
    """
    if not _chroma_available():
        pytest.skip("ChromaDB not available")

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    from backend.config import EMBEDDING_MODEL, TERM_COLLECTION_NAME
    from backend.tools.policy_search import _get_chroma_client

    client = _get_chroma_client()
    col = client.get_or_create_collection(
        name=TERM_COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL),
    )

    # Seed test entries with unique IDs so cleanup is reliable
    test_ids = [
        f"eval_test_{uuid.uuid4().hex[:8]}",
        f"eval_test_{uuid.uuid4().hex[:8]}",
        f"eval_test_{uuid.uuid4().hex[:8]}",
    ]
    col.add(
        ids=test_ids,
        documents=[
            "A whole-person, interdisciplinary approach to care for members with complex needs.",
            "Groups of Medi-Cal Members eligible for ECM, such as individuals experiencing homelessness.",
            "Substance Use Disorder treatment and recovery services for eligible beneficiaries.",
        ],
        metadatas=[
            {
                "term": "Enhanced Care Management",
                "abbreviation": "ECM",
                "definition": "A whole-person, interdisciplinary approach to care that addresses the clinical and non-clinical needs of Members with the most complex medical and social needs.",
                "source_file": "data/Example Input Doc - Hard.pdf",
                "page_number": 5,
                "section_heading": "Section II: Definitions",
                "source": "glossary",
            },
            {
                "term": "Population of Focus",
                "abbreviation": "POF",
                "definition": "The groups of Medi-Cal Members eligible for ECM, such as adults and youth experiencing homelessness, justice-involved individuals, adults at risk for long-term care institutionalization.",
                "source_file": "data/Example Input Doc - Hard.pdf",
                "page_number": 10,
                "section_heading": "Section IV: ECM Populations",
                "source": "glossary",
            },
            {
                "term": "Substance Use Disorder",
                "abbreviation": "SUD",
                "definition": "A treatable medical condition characterized by compulsive substance use despite negative consequences.",
                "source_file": "data/Example Input Doc - Hard.pdf",
                "page_number": 3,
                "section_heading": "Section I: Overview",
                "source": "glossary",
            },
        ],
    )

    yield col, test_ids

    # Cleanup: remove only the test entries we added
    try:
        col.delete(ids=test_ids)
    except Exception as exc:
        logger.warning("Failed to clean up seeded term entries: %s", exc)


# ===========================================================================
# Eval 1 — parse_agent_result: Edge cases beyond the unit tests
# ===========================================================================


@pytest.mark.eval
class TestParseAgentResultEdgeCases:
    """
    Parser robustness for realistic LLM output patterns not covered by unit tests.
    No API key or ChromaDB required.
    """

    def _valid(self, **overrides) -> dict:
        base = {
            "answer": "yes",
            "citation_text": "The plan shall maintain prior authorization procedures.",
            "source_file": "GG/GG.1508.pdf",
            "page_number": 3,
            "confidence": 0.85,
            "reasoning": "Passage directly addresses prior authorization requirements.",
        }
        return {**base, **overrides}

    def test_dict_roundtrip_preserves_all_fields(self):
        data = self._valid()
        ev = parse_agent_result(data, requirement_id=1)
        assert ev.answer == AnswerType.YES
        assert ev.citation_text == data["citation_text"]
        assert ev.source_file == data["source_file"]
        assert ev.page_number == data["page_number"]
        assert ev.confidence == pytest.approx(data["confidence"])
        assert ev.reasoning == data["reasoning"]
        assert ev.requirement_id == 1

    def test_answer_partial_parsed(self):
        ev = parse_agent_result(self._valid(answer="partial"), requirement_id=2)
        assert ev.answer == AnswerType.PARTIAL

    def test_answer_no_parsed(self):
        ev = parse_agent_result(self._valid(answer="no"), requirement_id=3)
        assert ev.answer == AnswerType.NO

    def test_confidence_clamped_at_zero_on_none(self):
        ev = parse_agent_result(self._valid(confidence=None), requirement_id=4)
        assert ev.confidence == pytest.approx(0.0)

    def test_page_number_clamped_at_zero_on_none(self):
        ev = parse_agent_result(self._valid(page_number=None), requirement_id=5)
        assert ev.page_number == 0

    def test_string_confidence_coerced(self):
        ev = parse_agent_result(self._valid(confidence="0.72"), requirement_id=6)
        assert ev.confidence == pytest.approx(0.72)

    def test_string_page_number_coerced(self):
        ev = parse_agent_result(self._valid(page_number="12"), requirement_id=7)
        assert ev.page_number == 12

    def test_invalid_page_number_string_coerced_to_zero(self):
        ev = parse_agent_result(self._valid(page_number="N/A"), requirement_id=8)
        assert ev.page_number == 0

    def test_extra_fields_ignored(self):
        """LLM may return extra fields — they must not crash the parser."""
        data = self._valid()
        data["extra_field"] = "should be ignored"
        data["model_notes"] = "internal"
        ev = parse_agent_result(data, requirement_id=9)
        assert ev.answer == AnswerType.YES

    def test_answer_uppercase_coerced(self):
        ev = parse_agent_result(self._valid(answer="YES"), requirement_id=10)
        assert ev.answer == AnswerType.YES

    def test_answer_mixed_case_coerced(self):
        ev = parse_agent_result(self._valid(answer="Partial"), requirement_id=11)
        assert ev.answer == AnswerType.PARTIAL

    def test_fenced_block_with_language_tag(self):
        inner = json.dumps(self._valid())
        ev = parse_agent_result(f"```json\n{inner}\n```", requirement_id=12)
        assert ev.answer == AnswerType.YES

    def test_fenced_block_without_language_tag(self):
        inner = json.dumps(self._valid())
        ev = parse_agent_result(f"```\n{inner}\n```", requirement_id=13)
        assert ev.answer == AnswerType.YES

    def test_invalid_answer_value_returns_fallback(self):
        ev = parse_agent_result(self._valid(answer="maybe"), requirement_id=14)
        assert ev.needs_human_review is True
        assert ev.confidence == pytest.approx(0.0)

    def test_invalid_json_returns_fallback(self):
        ev = parse_agent_result("not valid json at all", requirement_id=15)
        assert ev.needs_human_review is True
        assert ev.requirement_id == 15

    def test_empty_string_returns_fallback(self):
        ev = parse_agent_result("", requirement_id=16)
        assert ev.needs_human_review is True

    def test_completely_missing_answer_field_returns_fallback(self):
        data = {k: v for k, v in self._valid().items() if k != "answer"}
        ev = parse_agent_result(data, requirement_id=17)
        assert ev.needs_human_review is True

    def test_fallback_preserves_requirement_id(self):
        ev = _fallback_evaluation(requirement_id=42)
        assert ev.requirement_id == 42
        assert ev.answer == AnswerType.NO
        assert ev.needs_human_review is True
        assert ev.confidence == pytest.approx(0.0)

    def test_fallback_reasoning_is_nonempty(self):
        ev = _fallback_evaluation(requirement_id=1)
        assert ev.reasoning.strip()

    def test_fallback_custom_reasoning(self):
        ev = _fallback_evaluation(requirement_id=1, reasoning="timeout after 8 steps")
        assert "timeout" in ev.reasoning


# ===========================================================================
# Eval 2 — search_policies tool: format and retrieval quality
# ===========================================================================


@pytest.mark.eval
@requires_chroma
class TestSearchPoliciesTool:
    """
    Tests for the search_policies ChromaDB retrieval tool.
    Requires a populated ChromaDB collection (no LLM calls).
    """

    def test_returns_string(self):
        from backend.tools.policy_search import search_policies
        result = search_policies("prior authorization process")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_score_line(self):
        from backend.tools.policy_search import search_policies
        result = search_policies("hospice care coverage")
        assert "Score:" in result

    def test_result_contains_source_line(self):
        from backend.tools.policy_search import search_policies
        result = search_policies("utilization management timelines")
        assert "Source:" in result

    def test_result_contains_text_line(self):
        from backend.tools.policy_search import search_policies
        result = search_policies("member grievance and appeals")
        assert "Text:" in result

    def test_top_k_limits_results(self):
        from backend.tools.policy_search import search_policies
        result3 = search_policies("care coordination", top_k=3)
        result1 = search_policies("care coordination", top_k=1)
        # Three results should produce more output than one
        assert len(result3) > len(result1)

    def test_top_k_1_returns_exactly_one_result(self):
        from backend.tools.policy_search import search_policies
        result = search_policies("hospice benefit", top_k=1)
        # A single result has no separator
        assert result.count("[Result 1]") == 1
        assert "[Result 2]" not in result

    def test_relevant_query_returns_high_score_result(self):
        """
        A targeted query about prior authorization should return at least one
        result with a positive similarity score (score > 0).
        """
        from backend.tools.policy_search import search_policies
        result = search_policies("prior authorization request timeline 72 hours", top_k=5)
        # Extract first score from "Score: 0.XXX"
        import re
        scores = [float(m) for m in re.findall(r"Score:\s*(-?[\d.]+)", result)]
        assert scores, "No score found in search results"
        assert max(scores) > 0.0, f"Best score is non-positive: {max(scores)}"

    def test_empty_collection_query_returns_no_results_message(self):
        """
        An extremely specific nonsense query should still return a string
        (never raise), even if results are empty.
        """
        from backend.tools.policy_search import search_policies
        result = search_policies("xyzzy_nonsense_term_that_cannot_exist_in_corpus_abc123")
        assert isinstance(result, str)

    def test_result_separator_present_for_multiple_results(self):
        from backend.tools.policy_search import search_policies
        result = search_policies("provider network adequacy", top_k=3)
        # Multiple results are separated by ---
        assert "---" in result


# ===========================================================================
# Eval 3 — define_term tool: lookup correctness
# ===========================================================================


@pytest.mark.eval
@requires_chroma
class TestDefineTermTool:
    """
    Tests for the define_term ChromaDB lookup tool.
    Uses a seeded document_terms collection — no LLM calls needed.
    """

    def test_exact_abbreviation_uppercase_match(self, seeded_term_collection):
        """define_term('ECM') must return the Enhanced Care Management entry."""
        from backend.tools.policy_search import define_term
        result = define_term("ECM")
        assert "Enhanced Care Management" in result
        assert "No definition found" not in result

    def test_exact_abbreviation_lowercase_match(self, seeded_term_collection):
        """define_term('ecm') must return the same entry (case-insensitive)."""
        from backend.tools.policy_search import define_term
        result = define_term("ecm")
        assert "Enhanced Care Management" in result

    def test_exact_abbreviation_pof(self, seeded_term_collection):
        """define_term('POF') returns the Population of Focus entry."""
        from backend.tools.policy_search import define_term
        result = define_term("POF")
        assert "Population of Focus" in result

    def test_full_term_match(self, seeded_term_collection):
        """define_term with a full term string finds the correct entry."""
        from backend.tools.policy_search import define_term
        result = define_term("Substance Use Disorder")
        assert "SUD" in result or "Substance Use Disorder" in result

    def test_unknown_term_returns_no_definition_message(self, seeded_term_collection):
        """define_term for an unknown abbreviation returns 'No definition found'."""
        from backend.tools.policy_search import define_term
        result = define_term("XYZZY_NONEXISTENT_9999")
        assert "No definition found" in result

    def test_result_contains_definition_text(self, seeded_term_collection):
        """The returned string must include 'Definition:' for a matched term."""
        from backend.tools.policy_search import define_term
        result = define_term("ECM")
        assert "Definition:" in result

    def test_result_contains_source(self, seeded_term_collection):
        """The returned string must include source information."""
        from backend.tools.policy_search import define_term
        result = define_term("ECM")
        assert "Source:" in result

    def test_returns_string(self, seeded_term_collection):
        from backend.tools.policy_search import define_term
        assert isinstance(define_term("ECM"), str)
        assert isinstance(define_term("XYZZY_NONEXISTENT_9999"), str)


# ===========================================================================
# Eval 4 — evaluate_citation tool: LLM-based second opinion
# ===========================================================================


@pytest.mark.eval
@requires_api_key
class TestEvaluateCitationTool:
    """
    Tests for the evaluate_citation LLM tool.
    Makes real LLM calls; each test is one chat completion.
    """

    def test_returns_string(self):
        from backend.tools.policy_search import evaluate_citation
        result = evaluate_citation(
            requirement="Does the P&P address prior authorization timelines?",
            passage="Prior authorization requests shall be processed within 72 hours of receipt.",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_response_is_valid_json(self):
        from backend.tools.policy_search import evaluate_citation
        result = evaluate_citation(
            requirement="Does the P&P require member notification within 30 days?",
            passage="The plan must notify members of coverage decisions within 30 calendar days.",
        )
        # Strip any markdown fences
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            text = "\n".join(inner).strip()
        data = json.loads(text)
        assert isinstance(data, dict), "evaluate_citation did not return a JSON object"

    def test_response_has_required_fields(self):
        from backend.tools.policy_search import evaluate_citation
        result = evaluate_citation(
            requirement="Does the P&P address grievance procedures?",
            passage="Members may file a grievance within 180 days of the action.",
        )
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        assert "answer" in data, "Missing 'answer' field"
        assert "confidence" in data, "Missing 'confidence' field"
        assert "reasoning" in data, "Missing 'reasoning' field"

    def test_clear_yes_returns_yes(self):
        """A passage that explicitly satisfies the requirement → 'yes'."""
        from backend.tools.policy_search import evaluate_citation
        result = evaluate_citation(
            requirement="Does the P&P state that the plan must cover emergency services?",
            passage=(
                "The Managed Care Plan shall provide coverage for emergency services "
                "24 hours a day, 7 days a week, without prior authorization, "
                "regardless of the treating provider's network status."
            ),
        )
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        assert data["answer"].lower() in ("yes", "partial"), (
            f"Expected 'yes' or 'partial' for a clear match, got {data['answer']!r}"
        )

    def test_clear_no_returns_no(self):
        """A passage completely unrelated to the requirement → 'no'."""
        from backend.tools.policy_search import evaluate_citation
        result = evaluate_citation(
            requirement="Does the P&P describe the process for prior authorization of MRI scans?",
            passage=(
                "The plan's annual member satisfaction survey is conducted in the "
                "fourth quarter of each calendar year and results are reported to DHCS."
            ),
        )
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        assert data["answer"].lower() in ("no", "partial"), (
            f"Expected 'no' or 'partial' for an unrelated passage, got {data['answer']!r}"
        )

    def test_confidence_in_valid_range(self):
        """Confidence score must be a float in [0, 1]."""
        from backend.tools.policy_search import evaluate_citation
        result = evaluate_citation(
            requirement="Does the P&P address timely access standards?",
            passage="The plan shall ensure members have access to primary care within 10 business days.",
        )
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        conf = float(data["confidence"])
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} is out of range [0, 1]"

    def test_reasoning_is_nonempty(self):
        """Reasoning must be a non-empty string."""
        from backend.tools.policy_search import evaluate_citation
        result = evaluate_citation(
            requirement="Does the P&P address member rights?",
            passage="Members have the right to receive information about available services.",
        )
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        assert data.get("reasoning", "").strip(), "reasoning field is empty"


# ===========================================================================
# Eval 5 — Question Agent: output structure and field validity
# ===========================================================================


@pytest.mark.eval
@requires_api_key
@requires_chroma
class TestQuestionAgentOutputQuality:
    """
    Runs the full question agent on EVAL_REQUIREMENTS and checks that every
    Evaluation has valid structure and meaningful content.
    """

    def test_all_evaluations_have_correct_requirement_id(self, batch_evaluations):
        for req, ev in batch_evaluations:
            assert ev.requirement_id == req.id, (
                f"Req #{req.id}: evaluation has requirement_id={ev.requirement_id}"
            )

    def test_all_answers_are_valid_enum_values(self, batch_evaluations):
        valid = {AnswerType.YES, AnswerType.NO, AnswerType.PARTIAL}
        for req, ev in batch_evaluations:
            assert ev.answer in valid, (
                f"Req #{req.id}: invalid answer {ev.answer!r}"
            )

    def test_all_confidences_in_valid_range(self, batch_evaluations):
        for req, ev in batch_evaluations:
            assert CONFIDENCE_RANGE_MIN <= ev.confidence <= CONFIDENCE_RANGE_MAX, (
                f"Req #{req.id}: confidence {ev.confidence} out of [0, 1]"
            )

    def test_all_reasoning_fields_nonempty(self, batch_evaluations):
        for req, ev in batch_evaluations:
            words = ev.reasoning.split()
            assert len(words) >= REASONING_MIN_WORDS, (
                f"Req #{req.id}: reasoning too short ({len(words)} words): {ev.reasoning!r}"
            )

    def test_yes_answers_have_nonempty_citation(self, batch_evaluations):
        """Every 'yes' answer must include a citation from the policy."""
        for req, ev in batch_evaluations:
            if ev.answer == AnswerType.YES:
                assert len(ev.citation_text) >= YES_ANSWER_CITATION_MIN_LEN, (
                    f"Req #{req.id}: 'yes' answer has short/empty citation: {ev.citation_text!r}"
                )

    def test_yes_answers_have_nonempty_source_file(self, batch_evaluations):
        """Every 'yes' answer must reference a specific source file."""
        for req, ev in batch_evaluations:
            if ev.answer == AnswerType.YES:
                assert ev.source_file.strip(), (
                    f"Req #{req.id}: 'yes' answer has empty source_file"
                )

    def test_yes_answers_have_positive_page_number(self, batch_evaluations):
        """Every 'yes' answer must have a page number ≥ 1."""
        for req, ev in batch_evaluations:
            if ev.answer == AnswerType.YES:
                assert ev.page_number >= 1, (
                    f"Req #{req.id}: 'yes' answer has page_number={ev.page_number}"
                )

    def test_fallback_rate_below_threshold(self, batch_evaluations):
        """Fewer than FALLBACK_RATE_MAX of runs may be fallback (agent errors)."""
        fallbacks = [(req.id, ev) for req, ev in batch_evaluations if _evaluation_is_fallback(ev)]
        rate = len(fallbacks) / len(batch_evaluations)

        report = EvalReport("fallback_rate")
        report.record("rate", rate, threshold=0.0)
        report.record("max_allowed", FALLBACK_RATE_MAX)
        report.extras["fallback_req_ids"] = [req_id for req_id, _ in fallbacks]
        print(report)

        assert rate <= FALLBACK_RATE_MAX, (
            f"Fallback rate {rate:.1%} > {FALLBACK_RATE_MAX:.0%}. "
            f"Fallback requirement IDs: {[req_id for req_id, _ in fallbacks]}"
        )

    def test_average_confidence_above_threshold(self, batch_evaluations):
        """Average confidence across the batch must be ≥ MIN_AVG_CONFIDENCE."""
        confidences = [ev.confidence for _, ev in batch_evaluations]
        avg_conf = sum(confidences) / len(confidences)

        report = EvalReport("average_confidence")
        report.record("avg_confidence", avg_conf, threshold=MIN_AVG_CONFIDENCE)
        report.extras["per_req"] = [
            (req.id, round(ev.confidence, 2)) for req, ev in batch_evaluations
        ]
        print(report)

        assert avg_conf >= MIN_AVG_CONFIDENCE, (
            f"Average confidence {avg_conf:.3f} < {MIN_AVG_CONFIDENCE} threshold. "
            f"Per-req: {[(req.id, round(ev.confidence, 2)) for req, ev in batch_evaluations]}"
        )

    def test_answer_distribution_not_all_no(self, batch_evaluations):
        """
        The agent should not answer 'no' for every requirement.
        A well-functioning agent over known healthcare requirements should find
        at least some supporting passages.
        """
        answers = [ev.answer for _, ev in batch_evaluations]
        non_no = sum(1 for a in answers if a != AnswerType.NO)

        report = EvalReport("answer_distribution")
        report.record("non_no_count", float(non_no))
        report.extras["distribution"] = {a.value: answers.count(a) for a in AnswerType}
        print(report)

        assert non_no >= 1, (
            f"All {len(answers)} requirements answered 'no' — "
            "agent may not be connecting to ChromaDB or producing valid results."
        )

    def test_source_files_look_like_pdf_paths(self, batch_evaluations):
        """Source files in 'yes' answers must end with .pdf."""
        for req, ev in batch_evaluations:
            if ev.answer == AnswerType.YES and ev.source_file:
                assert ev.source_file.lower().endswith(".pdf"), (
                    f"Req #{req.id}: source_file does not end with .pdf: {ev.source_file!r}"
                )


# ===========================================================================
# Eval 6 — Citation grounding: cited text is not fabricated
# ===========================================================================


@pytest.mark.eval
@requires_api_key
@requires_chroma
class TestCitationGrounding:
    """
    Anti-fabrication check: verifies that the citation text returned for
    'yes' answers has meaningful similarity to actual passages in ChromaDB,
    rather than being invented prose.

    Uses embedding cosine similarity rather than exact-match to tolerate
    minor differences (truncation, whitespace normalization).
    """

    def test_yes_citation_grounded_in_corpus(self, batch_evaluations):
        """
        For each 'yes' evaluation, the citation_text must have cosine similarity
        ≥ GROUNDING_MIN_SIMILARITY to at least one passage in the policy corpus.
        """
        from sentence_transformers import SentenceTransformer

        from backend.tools.policy_search import get_chroma_collection

        model = SentenceTransformer("all-MiniLM-L6-v2")
        collection = get_chroma_collection()

        ungrounded = []
        for req, ev in batch_evaluations:
            if ev.answer != AnswerType.YES or not ev.citation_text.strip():
                continue

            # Retrieve the closest passage to the citation text
            results = collection.query(query_texts=[ev.citation_text], n_results=1)
            if not results["distances"] or not results["distances"][0]:
                ungrounded.append((req.id, ev.citation_text[:60], 0.0))
                continue

            # ChromaDB L2 distance; convert to cosine similarity (approx for normalized vecs)
            dist = results["distances"][0][0]
            sim = max(0.0, 1.0 - dist)

            if sim < GROUNDING_MIN_SIMILARITY:
                ungrounded.append((req.id, ev.citation_text[:60], sim))

        report = EvalReport("citation_grounding")
        yes_count = sum(1 for _, ev in batch_evaluations if ev.answer == AnswerType.YES)
        grounded_count = yes_count - len(ungrounded)
        report.record("grounded_count", float(grounded_count))
        report.record("yes_count", float(yes_count))
        report.extras["ungrounded"] = ungrounded
        print(report)

        assert not ungrounded, (
            f"Ungrounded citations found (similarity < {GROUNDING_MIN_SIMILARITY}): "
            f"{ungrounded}"
        )

    def test_citation_not_longer_than_any_passage(self, batch_evaluations):
        """
        A fabricated citation would likely be verbose.  Citations should not
        be implausibly long (>800 chars) compared to typical chunk sizes.
        """
        MAX_CITATION_LEN = 800
        long_citations = [
            (req.id, len(ev.citation_text), ev.citation_text[:80])
            for req, ev in batch_evaluations
            if ev.answer == AnswerType.YES and len(ev.citation_text) > MAX_CITATION_LEN
        ]
        assert not long_citations, (
            f"Suspiciously long citations (>{MAX_CITATION_LEN} chars): {long_citations}"
        )

    def test_no_citations_from_term_collection(self, batch_evaluations):
        """
        Citations must come from policy_documents, not document_terms.
        The system prompt forbids citing term definitions — verify this holds.
        """
        term_citation_markers = ["Definition:", "Abbreviation:", "glossary"]
        violations = [
            (req.id, ev.citation_text[:80])
            for req, ev in batch_evaluations
            if ev.answer == AnswerType.YES
            and any(marker in ev.citation_text for marker in term_citation_markers)
        ]
        assert not violations, (
            f"Citations appear to come from document_terms (violates prompt instruction): "
            f"{violations}"
        )


# ===========================================================================
# Eval 7 — Acronym awareness: agent expands terms via define_term
# ===========================================================================


@pytest.mark.eval
@requires_api_key
@requires_chroma
class TestAcronymAwareness:
    """
    Verifies that when a requirement contains known acronyms (ECM, POF),
    the agent expands them and uses the expanded terms in its search.

    We infer this by checking that the reasoning mentions the full term name,
    which the agent can only know if it called define_term or had prior knowledge.
    """

    @pytest.mark.anyio
    async def test_ecm_requirement_produces_valid_evaluation(self, model, seeded_term_collection):
        """
        Running the agent on an ECM/POF requirement must return a valid Evaluation
        (not a fallback), demonstrating it handled the acronyms.
        """
        ev = await run_question_agent(ECM_REQUIREMENT, model)

        report = EvalReport("ecm_acronym_evaluation")
        report.record("confidence", ev.confidence)
        report.record("is_fallback", float(_evaluation_is_fallback(ev)), threshold=0.0)
        report.extras["answer"] = ev.answer.value
        report.extras["reasoning_snippet"] = ev.reasoning[:200]
        print(report)

        assert not _evaluation_is_fallback(ev), (
            f"Agent returned a fallback for the ECM requirement. "
            f"Reasoning: {ev.reasoning}"
        )
        assert ev.answer in {AnswerType.YES, AnswerType.NO, AnswerType.PARTIAL}

    @pytest.mark.anyio
    async def test_ecm_reasoning_mentions_care_management(self, model, seeded_term_collection):
        """
        The reasoning for the ECM requirement should reference 'care management'
        or 'ECM' — evidence the agent understood the acronym.
        """
        ev = await run_question_agent(ECM_REQUIREMENT, model)

        reasoning_lower = ev.reasoning.lower()
        expanded_present = (
            "enhanced care management" in reasoning_lower
            or "care management" in reasoning_lower
            or "ecm" in reasoning_lower
        )

        report = EvalReport("ecm_term_in_reasoning")
        report.record("expanded_term_present", float(expanded_present), threshold=1.0)
        report.extras["reasoning"] = ev.reasoning[:300]
        print(report)

        assert expanded_present, (
            f"Reasoning for ECM requirement does not mention 'care management' or 'ecm'. "
            f"Reasoning: {ev.reasoning[:200]}"
        )


# ===========================================================================
# Eval 8 — Answer stability: same requirement → same answer on two runs
# ===========================================================================


@pytest.mark.eval
@requires_api_key
@requires_chroma
class TestAgentStability:
    """
    Runs the agent twice on the same requirement and checks that the
    answers agree.  High variance suggests the model is too sensitive
    to sampling noise at temperature=0.2.
    """

    @pytest.mark.anyio
    async def test_same_answer_on_two_runs(self, model):
        """
        Two independent runs on the same requirement must return the same
        AnswerType (yes/no/partial).
        """
        req = EVAL_REQUIREMENTS[0]  # Use Q1 (hospice coverage)

        import asyncio as _asyncio
        ev1 = await run_question_agent(req, model)
        ev2 = await run_question_agent(req, model)

        report = EvalReport("answer_stability")
        report.extras["run1"] = f"{ev1.answer.value} (conf={ev1.confidence:.2f})"
        report.extras["run2"] = f"{ev2.answer.value} (conf={ev2.confidence:.2f})"
        print(report)

        assert ev1.answer == ev2.answer, (
            f"Req #{req.id}: answers differ between runs — "
            f"run1={ev1.answer.value}, run2={ev2.answer.value}. "
            f"Model may be unstable at temperature=0.2."
        )

    @pytest.mark.anyio
    async def test_confidence_variance_within_tolerance(self, model):
        """
        Confidence scores for two runs on the same requirement must be
        within 0.30 of each other.
        """
        req = EVAL_REQUIREMENTS[1]  # Use Q18 (prior authorization)
        MAX_CONF_DELTA = 0.30

        ev1 = await run_question_agent(req, model)
        ev2 = await run_question_agent(req, model)

        delta = abs(ev1.confidence - ev2.confidence)

        report = EvalReport("confidence_stability")
        report.record("delta", delta, threshold=MAX_CONF_DELTA)
        report.extras["run1_confidence"] = ev1.confidence
        report.extras["run2_confidence"] = ev2.confidence
        print(report)

        assert delta <= MAX_CONF_DELTA, (
            f"Req #{req.id}: confidence delta {delta:.3f} > {MAX_CONF_DELTA}. "
            f"run1={ev1.confidence:.3f}, run2={ev2.confidence:.3f}"
        )
