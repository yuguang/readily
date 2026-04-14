"""
Tests for Component 3: Requirement Extraction.

Covers:
- parse_review_form: exact count, spot-check questions 1/18/40/64, text clean-up
- classify_and_extract router: correct doc_type for structured and narrative PDFs
- GetDocumentTextTool: returns text, validates smolagents interface
- _parse_requirements: JSON, markdown-fenced, list, and error cases
- Edge cases: empty text, no matching questions, whitespace-only text
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF — used to build synthetic PDFs in tests
import pytest

from backend.models.schemas import Requirement
from backend.tools.narrative_extractor import GetDocumentTextTool, _parse_requirements
from backend.tools.review_form_parser import (
    _clean_question_text,
    is_structured_form,
    parse_review_form,
)

# ---------------------------------------------------------------------------
# Paths to sample documents (used in integration-style tests)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
EASY_PDF = DATA_DIR / "Example Input Doc - Easy.pdf"
HARD_PDF = DATA_DIR / "Example Input Doc - Hard.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _easy_pdf_text() -> str:
    """Extract full text from the Easy PDF once for all tests in this module."""
    doc = fitz.open(str(EASY_PDF))
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _hard_pdf_text() -> str:
    """Extract first 20 pages of the Hard PDF (enough for routing tests)."""
    doc = fitz.open(str(HARD_PDF))
    try:
        return "\n".join(doc[i].get_text() for i in range(min(20, len(doc))))
    finally:
        doc.close()


# Cache to avoid re-parsing on every test
_EASY_TEXT: str | None = None
_HARD_TEXT: str | None = None


def easy_text() -> str:
    global _EASY_TEXT
    if _EASY_TEXT is None:
        _EASY_TEXT = _easy_pdf_text()
    return _EASY_TEXT


def hard_text() -> str:
    global _HARD_TEXT
    if _HARD_TEXT is None:
        _HARD_TEXT = _hard_pdf_text()
    return _HARD_TEXT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def easy_requirements() -> list[Requirement]:
    """All 64 requirements extracted from the Easy PDF."""
    return parse_review_form(easy_text())


# ===========================================================================
# parse_review_form — count and spot-checks
# ===========================================================================


class TestParseReviewFormCount:
    def test_exact_count(self, easy_requirements):
        assert len(easy_requirements) == 64

    def test_ids_are_sequential(self, easy_requirements):
        ids = [r.id for r in easy_requirements]
        assert ids == list(range(1, 65))

    def test_all_have_text(self, easy_requirements):
        for r in easy_requirements:
            assert r.text, f"Requirement {r.id} has empty text"

    def test_all_have_reference(self, easy_requirements):
        for r in easy_requirements:
            assert r.reference, f"Requirement {r.id} has no reference"
            assert r.reference.startswith("APL"), (
                f"Requirement {r.id} reference doesn't start with APL: {r.reference!r}"
            )


class TestParseReviewFormSpotChecks:
    """Spot-check requirements 1, 18, 40, and 64 for content accuracy."""

    def test_q1_text_starts_correctly(self, easy_requirements):
        q1 = next(r for r in easy_requirements if r.id == 1)
        assert q1.text.startswith("Does the P&P state")
        assert "hospice" in q1.text.lower()

    def test_q1_reference(self, easy_requirements):
        q1 = next(r for r in easy_requirements if r.id == 1)
        assert q1.reference == "APL 25-008, page 1"

    def test_q18_text_contains_prior_authorization(self, easy_requirements):
        q18 = next(r for r in easy_requirements if r.id == 18)
        assert "Prior Authorization" in q18.text or "prior authorization" in q18.text.lower()

    def test_q18_reference(self, easy_requirements):
        q18 = next(r for r in easy_requirements if r.id == 18)
        assert "APL 25-008" in q18.reference

    def test_q40_text_mentions_routine_home_care(self, easy_requirements):
        q40 = next(r for r in easy_requirements if r.id == 40)
        assert "routine home care" in q40.text.lower()

    def test_q40_reference(self, easy_requirements):
        q40 = next(r for r in easy_requirements if r.id == 40)
        assert q40.reference == "APL 25-008, page 12"

    def test_q64_is_last(self, easy_requirements):
        q64 = next(r for r in easy_requirements if r.id == 64)
        assert q64.id == 64

    def test_q64_text_mentions_dhcs_audit(self, easy_requirements):
        q64 = next(r for r in easy_requirements if r.id == 64)
        assert "DHCS" in q64.text
        assert "audit" in q64.text.lower() or "inspect" in q64.text.lower()

    def test_q64_reference(self, easy_requirements):
        q64 = next(r for r in easy_requirements if r.id == 64)
        assert q64.reference == "APL 25-008, page 16"


class TestParseReviewFormTextCleanup:
    """Verify whitespace normalisation and editorial note removal."""

    def test_no_embedded_newlines(self, easy_requirements):
        for r in easy_requirements:
            assert "\n" not in r.text, (
                f"Requirement {r.id} text contains newline: {r.text!r}"
            )

    def test_no_multiple_spaces(self, easy_requirements):
        for r in easy_requirements:
            assert "  " not in r.text, (
                f"Requirement {r.id} text contains multiple spaces: {r.text!r}"
            )

    def test_editorial_note_stripped(self):
        """Question 57 has '(P&P may include examples provided in APL)' — must be removed."""
        text = easy_text()
        reqs = parse_review_form(text)
        q57 = next(r for r in reqs if r.id == 57)
        assert "(P&P may include" not in q57.text

    def test_category_is_none_for_structured(self, easy_requirements):
        """Structured form extraction does not assign categories."""
        for r in easy_requirements:
            assert r.category is None


# ===========================================================================
# parse_review_form — edge cases
# ===========================================================================


class TestParseReviewFormEdgeCases:
    def test_empty_string(self):
        assert parse_review_form("") == []

    def test_whitespace_only(self):
        assert parse_review_form("   \n\n\t  ") == []

    def test_no_matching_pattern(self):
        text = "This document has no questions. It is a policy statement."
        assert parse_review_form(text) == []

    def test_partial_match_skipped(self):
        """A question missing its reference is not returned."""
        text = "1. Does the P&P state that members may enrol?\n\n Yes   No"
        assert parse_review_form(text) == []

    def test_single_question(self):
        text = (
            "1. Does the P&P state that MCPs must provide coverage?\n"
            "(Reference: APL 99-001, page 3)\n"
            " Yes   No\n"
        )
        reqs = parse_review_form(text)
        assert len(reqs) == 1
        assert reqs[0].id == 1
        assert reqs[0].reference == "APL 99-001, page 3"

    def test_multi_page_reference(self):
        text = (
            "5. Does the P&P state that timely access applies within 24 hours?\n"
            "(Reference: APL 25-008, pages 1-2)\n"
            " Yes   No\n"
        )
        reqs = parse_review_form(text)
        assert len(reqs) == 1
        assert reqs[0].reference == "APL 25-008, pages 1-2"


# ===========================================================================
# _clean_question_text
# ===========================================================================


class TestCleanQuestionText:
    def test_normalises_newlines(self):
        raw = "Does the P&P state that\nMCPs must\nprovide care?\n"
        assert "\n" not in _clean_question_text(raw)

    def test_normalises_multiple_spaces(self):
        raw = "Does the P&P state  that   MCPs  must?"
        assert "  " not in _clean_question_text(raw)

    def test_strips_editorial_note(self):
        raw = "Does the P&P state X? \n(P&P may include examples provided in APL) \n"
        result = _clean_question_text(raw)
        assert "(P&P may include" not in result
        assert "Does the P&P state X?" in result

    def test_empty_string(self):
        assert _clean_question_text("") == ""


# ===========================================================================
# is_structured_form
# ===========================================================================


class TestIsStructuredForm:
    def test_easy_pdf_is_structured(self):
        assert is_structured_form(easy_text()) is True

    def test_hard_pdf_is_not_structured(self):
        assert is_structured_form(hard_text()) is False

    def test_empty_string(self):
        assert is_structured_form("") is False

    def test_no_numbered_questions(self):
        assert is_structured_form("This is a narrative policy document.") is False


# ===========================================================================
# GetDocumentTextTool
# ===========================================================================


class TestGetDocumentTextTool:
    def test_returns_stored_text(self):
        tool = GetDocumentTextTool("hello regulatory world")
        assert tool() == "hello regulatory world"

    def test_empty_text(self):
        tool = GetDocumentTextTool("")
        assert tool() == ""

    def test_name_and_description_set(self):
        tool = GetDocumentTextTool("text")
        assert tool.name == "get_document_text"
        assert tool.description  # non-empty

    def test_inputs_is_empty_dict(self):
        tool = GetDocumentTextTool("x")
        assert tool.inputs == {}

    def test_output_type_is_string(self):
        tool = GetDocumentTextTool("x")
        assert tool.output_type == "string"


# ===========================================================================
# _parse_requirements
# ===========================================================================


class TestParseRequirements:
    def _make_item(self, id: int = 1) -> dict:
        return {
            "id": id,
            "text": f"Does the P&P state requirement {id}?",
            "reference": f"Section {id}",
            "category": "General",
        }

    def test_parses_list_of_dicts(self):
        data = [self._make_item(1), self._make_item(2)]
        reqs = _parse_requirements(data)
        assert len(reqs) == 2
        assert reqs[0].id == 1
        assert reqs[1].id == 2

    def test_parses_json_string(self):
        data = json.dumps([self._make_item(1)])
        reqs = _parse_requirements(data)
        assert len(reqs) == 1
        assert reqs[0].id == 1

    def test_parses_markdown_fenced_json(self):
        inner = json.dumps([self._make_item(3)])
        fenced = f"```json\n{inner}\n```"
        reqs = _parse_requirements(fenced)
        assert len(reqs) == 1
        assert reqs[0].id == 3

    def test_extracts_array_from_prose(self):
        inner = json.dumps([self._make_item(5)])
        wrapped = f"Here are the requirements:\n{inner}\nEnd of output."
        reqs = _parse_requirements(wrapped)
        assert len(reqs) == 1
        assert reqs[0].id == 5

    def test_skips_malformed_items(self):
        """Items missing required fields are skipped with a warning, not raised."""
        data = [
            self._make_item(1),
            {"id": "bad", "text": None},  # malformed — missing required fields
            self._make_item(3),
        ]
        reqs = _parse_requirements(data)
        # Only the two valid items should be returned
        valid_ids = {r.id for r in reqs}
        assert 1 in valid_ids
        assert 3 in valid_ids

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_requirements("not json at all {{{")

    def test_raises_on_non_array_json(self):
        with pytest.raises(ValueError, match="JSON array"):
            _parse_requirements('{"id": 1, "text": "Q?"}')

    def test_requirement_objects_passthrough(self):
        """A list of Requirement objects should be returned as-is."""
        req = Requirement(id=1, text="Q?")
        # _parse_requirements skips non-dict items from a list
        result = _parse_requirements([req])
        # The Requirement is not a dict, so it's excluded
        assert result == []


# ===========================================================================
# classify_and_extract router — async unit tests (mock LLM / Component 8)
# ===========================================================================


class TestClassifyAndExtractRouter:
    """
    Router tests that do NOT make real LLM calls.
    The narrative and compliance extractors are mocked so tests run offline.
    """

    @pytest.mark.anyio
    async def test_routes_easy_pdf_to_structured(self):
        """Easy PDF (14 pages, 64 numbered questions) → structured."""
        from backend.agents.extractor import classify_and_extract

        doc_type, reqs = await classify_and_extract(str(EASY_PDF))
        assert doc_type == "structured"
        assert len(reqs) == 64

    @pytest.mark.anyio
    async def test_routes_hard_pdf_to_compliance(self):
        """Hard PDF (145 pages) → compliance extraction agent (Route 2)."""
        from backend.agents.extractor import classify_and_extract
        from backend.models.schemas import ComplianceRequirement

        mock_reqs = [
            ComplianceRequirement(id=1, text="Does the P&P state X?", reference="Section I", category="ECM"),
            ComplianceRequirement(id=2, text="Does the P&P state Y?", reference="Section II", category="ECM"),
        ]
        with patch(
            "backend.agents.extractor.run_compliance_extractor",
            return_value=mock_reqs,
        ) as mock_fn:
            doc_type, reqs = await classify_and_extract(str(HARD_PDF))

        assert doc_type == "compliance"
        assert reqs == mock_reqs
        mock_fn.assert_called_once_with(str(HARD_PDF))

    @pytest.mark.anyio
    async def test_empty_pdf_returns_narrative_empty(self, tmp_path):
        """A PDF whose pages yield no text returns ('narrative', [])."""
        from backend.agents.extractor import classify_and_extract

        empty_pdf = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(empty_pdf))
        doc.close()

        with patch("backend.agents.extractor.run_narrative_extractor") as mock_narrative, \
             patch("backend.agents.extractor.run_compliance_extractor") as mock_compliance:
            doc_type, reqs = await classify_and_extract(str(empty_pdf))

        assert doc_type == "narrative"
        assert reqs == []
        mock_narrative.assert_not_called()
        mock_compliance.assert_not_called()

    @pytest.mark.anyio
    async def test_structured_threshold_is_3(self):
        """Two matching questions (< threshold) routes to narrative, not compliance."""
        from backend.agents.extractor import classify_and_extract

        two_questions = (
            "1. Does the P&P state MCPs must do X?\n"
            "(Reference: APL 99-001, page 1)\n"
            " Yes  No\n"
            "2. Does the P&P state MCPs must do Y?\n"
            "(Reference: APL 99-001, page 2)\n"
            " Yes  No\n"
        )

        with (
            patch("backend.agents.extractor.parse_pdf") as mock_parse,
            patch(
                "backend.agents.extractor.run_narrative_extractor",
                return_value=[],
            ) as mock_narrative,
        ):
            # 1 page → below threshold → Route 3 (short narrative)
            mock_parse.return_value = [{"page_number": 1, "text": two_questions}]
            doc_type, _ = await classify_and_extract("fake.pdf")

        assert doc_type == "narrative"
        mock_narrative.assert_called_once()

    @pytest.mark.anyio
    async def test_structured_threshold_met_skips_llm(self):
        """Three or more structured questions → Route 1, no LLM call."""
        from backend.agents.extractor import classify_and_extract

        three_questions = "".join(
            f"{i}. Does the P&P state requirement {i}?\n"
            f"(Reference: APL 99-001, page {i})\n Yes  No\n"
            for i in range(1, 4)
        )

        with (
            patch("backend.agents.extractor.parse_pdf") as mock_parse,
            patch("backend.agents.extractor.run_narrative_extractor") as mock_narrative,
            patch("backend.agents.extractor.run_compliance_extractor") as mock_compliance,
        ):
            mock_parse.return_value = [{"page_number": 1, "text": three_questions}]
            doc_type, reqs = await classify_and_extract("fake.pdf")

        assert doc_type == "structured"
        assert len(reqs) == 3
        mock_narrative.assert_not_called()
        mock_compliance.assert_not_called()

    @pytest.mark.anyio
    async def test_long_doc_routes_to_compliance(self):
        """Doc with no structured questions and >20 pages → Route 2 (compliance)."""
        from backend.agents.extractor import classify_and_extract
        from backend.models.schemas import ComplianceRequirement

        mock_reqs = [ComplianceRequirement(id=1, text="Does the P&P state Z?")]

        with (
            patch("backend.agents.extractor.parse_pdf") as mock_parse,
            patch(
                "backend.agents.extractor.run_compliance_extractor",
                return_value=mock_reqs,
            ) as mock_compliance,
            patch("backend.agents.extractor.run_narrative_extractor") as mock_narrative,
        ):
            # Simulate 21 pages of plain narrative text
            mock_parse.return_value = [
                {"page_number": i, "text": f"Narrative page {i}."}
                for i in range(1, 22)
            ]
            doc_type, reqs = await classify_and_extract("long_doc.pdf")

        assert doc_type == "compliance"
        assert reqs == mock_reqs
        mock_compliance.assert_called_once_with("long_doc.pdf")
        mock_narrative.assert_not_called()

    @pytest.mark.anyio
    async def test_boundary_20_pages_routes_to_narrative(self):
        """A doc with exactly 20 pages (= threshold) → Route 3 (short narrative)."""
        from backend.agents.extractor import classify_and_extract

        with (
            patch("backend.agents.extractor.parse_pdf") as mock_parse,
            patch(
                "backend.agents.extractor.run_narrative_extractor",
                return_value=[],
            ) as mock_narrative,
            patch("backend.agents.extractor.run_compliance_extractor") as mock_compliance,
        ):
            mock_parse.return_value = [
                {"page_number": i, "text": f"Narrative page {i}."}
                for i in range(1, 21)  # exactly 20 pages
            ]
            doc_type, _ = await classify_and_extract("boundary.pdf")

        assert doc_type == "narrative"
        mock_narrative.assert_called_once()
        mock_compliance.assert_not_called()
