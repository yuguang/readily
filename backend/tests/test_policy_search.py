"""
Unit tests for backend/tools/policy_search.py

Covers:
- define_term: exact abbreviation hit, case-insensitive variant, full-term hit, miss
- _format_definitions: output shape and content
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.tools.policy_search import _format_definitions, define_term

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ECM_META = {
    "term": "Enhanced Care Management",
    "abbreviation": "ECM",
    "definition": (
        "A whole-person, interdisciplinary approach to care that addresses the "
        "clinical and non-clinical needs of Members with the most complex medical "
        "and social needs."
    ),
    "source_file": "data/Example Input Doc - Hard.pdf",
    "page_number": 5,
    "section_heading": "Section II",
    "source": "glossary",
}

POF_META = {
    "term": "Population of Focus",
    "abbreviation": "POF",
    "definition": (
        "The groups of Medi-Cal Members eligible for ECM, such as adults and youth "
        "experiencing homelessness, justice-involved individuals, adults at risk for "
        "long-term care institutionalization, etc."
    ),
    "source_file": "data/Example Input Doc - Hard.pdf",
    "page_number": 10,
    "section_heading": "Section IV",
    "source": "glossary",
}


def _make_empty_query_result() -> dict:
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _make_query_result(metadatas: list[dict]) -> dict:
    return {
        "ids": [["id1"]],
        "documents": [[""]],
        "metadatas": [metadatas],
        "distances": [[0.1]],
    }


def _make_get_result(metadatas: list[dict]) -> dict:
    ids = [f"id{i}" for i in range(len(metadatas))]
    return {
        "ids": ids,
        "documents": [""] * len(metadatas),
        "metadatas": metadatas,
    }


def _make_empty_get_result() -> dict:
    return {"ids": [], "documents": [], "metadatas": []}


# ---------------------------------------------------------------------------
# _format_definitions
# ---------------------------------------------------------------------------


class TestFormatDefinitions:
    def test_single_entry_with_abbreviation(self):
        output = _format_definitions([ECM_META])
        assert "[Enhanced Care Management (ECM)]" in output
        assert "Definition: A whole-person" in output
        assert "Section II" in output
        assert "page 5" in output

    def test_entry_without_abbreviation(self):
        meta = dict(ECM_META)
        meta.pop("abbreviation")
        output = _format_definitions([meta])
        assert "[Enhanced Care Management]" in output
        assert "(ECM)" not in output

    def test_entry_without_section_heading(self):
        meta = dict(ECM_META)
        meta["section_heading"] = None
        output = _format_definitions([meta])
        assert "Section II" not in output
        assert "page 5" in output

    def test_multiple_entries_separated_by_dashes(self):
        output = _format_definitions([ECM_META, POF_META])
        assert "---" in output
        assert "Enhanced Care Management" in output
        assert "Population of Focus" in output


# ---------------------------------------------------------------------------
# define_term
# ---------------------------------------------------------------------------


class TestDefineTerm:
    def _patch_collection(self, get_result: dict, query_result: dict | None = None):
        """Patch _get_term_collection and return the mock collection."""
        mock_col = MagicMock()
        mock_col.get.return_value = get_result
        if query_result is not None:
            mock_col.query.return_value = query_result
        return mock_col

    def test_exact_abbreviation_hit(self):
        """define_term('ECM') finds the entry via exact metadata match."""
        mock_col = self._patch_collection(
            get_result=_make_get_result([ECM_META]),
        )
        with patch(
            "backend.tools.policy_search._get_term_collection",
            return_value=mock_col,
        ):
            result = define_term("ECM")

        assert "Enhanced Care Management" in result
        assert "ECM" in result
        assert "whole-person" in result
        # Should NOT fall through to embedding search
        mock_col.query.assert_not_called()

    def test_case_insensitive_abbreviation(self):
        """define_term('ecm') (lowercase) should produce the same result as 'ECM'."""
        mock_col = self._patch_collection(
            get_result=_make_get_result([ECM_META]),
        )
        with patch(
            "backend.tools.policy_search._get_term_collection",
            return_value=mock_col,
        ):
            result = define_term("ecm")

        assert "Enhanced Care Management" in result
        # The tool passes term.upper() as one of the where-clause values,
        # so the mock returns the same hit regardless of input case.
        mock_col.query.assert_not_called()

    def test_full_term_hit(self):
        """define_term('Population of Focus') matches on the 'term' metadata field."""
        mock_col = self._patch_collection(
            get_result=_make_get_result([POF_META]),
        )
        with patch(
            "backend.tools.policy_search._get_term_collection",
            return_value=mock_col,
        ):
            result = define_term("Population of Focus")

        assert "Population of Focus" in result
        assert "POF" in result
        mock_col.query.assert_not_called()

    def test_miss_returns_no_definition_string(self):
        """define_term('XYZZY') returns the canonical 'No definition found' string."""
        mock_col = self._patch_collection(
            get_result=_make_empty_get_result(),
            query_result=_make_empty_query_result(),
        )
        with patch(
            "backend.tools.policy_search._get_term_collection",
            return_value=mock_col,
        ):
            result = define_term("XYZZY")

        assert result == "No definition found for 'XYZZY'."

    def test_exact_miss_falls_back_to_embedding_search(self):
        """When exact metadata lookup returns nothing, embedding search is tried."""
        mock_col = self._patch_collection(
            get_result=_make_empty_get_result(),
            query_result=_make_query_result([ECM_META]),
        )
        with patch(
            "backend.tools.policy_search._get_term_collection",
            return_value=mock_col,
        ):
            result = define_term("Enhanced Care")

        mock_col.query.assert_called_once()
        assert "Enhanced Care Management" in result

    def test_returns_string(self):
        """define_term always returns a str (ToolCallingAgent requires string outputs)."""
        mock_col = self._patch_collection(
            get_result=_make_empty_get_result(),
            query_result=_make_empty_query_result(),
        )
        with patch(
            "backend.tools.policy_search._get_term_collection",
            return_value=mock_col,
        ):
            result = define_term("ECM")

        assert isinstance(result, str)
