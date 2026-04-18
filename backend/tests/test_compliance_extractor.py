"""
Tests for Component 8: Compliance Extraction Agent.

Covers:
- segment_document: heading detection, table accumulation, preamble grouping,
  empty-section merging, long-section sub-splitting
- filter_obligation_sections: obligation pattern matching, table pass-through
- deduplicate_requirements: identical texts collapse, distinct texts kept
- assemble_hierarchy: parent_id assignment for H3 requirements
- finalize_requirements: sequential IDs, parent_id conversion
- _parse_compliance_requirements: JSON/list/error cases
- extract_table_requirements: column mapping, LLM fallback trigger
- GetSectionTextTool: interface contract
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.compliance_extractor import (
    GetSectionTextTool,
    ResolveCrossReferenceTool,
    StripHeadersFootersTool,
    _parse_compliance_requirements,
    _run_section_with_timeout,
    assemble_hierarchy,
    deduplicate_requirements,
    extract_from_all_sections,
    extract_table_requirements,
    finalize_requirements,
)
from backend.models.schemas import ComplianceRequirement, TermDefinition
from backend.tools.document_segmenter import (
    DocumentSection,
    OBLIGATION_PATTERNS,
    filter_obligation_sections,
    segment_document,
)
from backend.tools.term_extractor import (
    _initials_match,
    _parse_pipe_table,
    extract_term_definitions,
    upsert_term_definitions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(
    heading: str = "Test Section",
    level: int = 2,
    text: str = "The MCP must maintain records.",
    tables: list[str] | None = None,
    page_start: int = 1,
    page_end: int = 1,
) -> DocumentSection:
    return DocumentSection(
        heading=heading,
        level=level,
        text=text,
        tables=tables or [],
        page_start=page_start,
        page_end=page_end,
    )


def _make_req(
    text: str = "Does the P&P state that MCPs must maintain records?",
    section_heading: str = "Test Section",
    level: int = 2,
    obligation_type: str | None = "mandatory",
    exact_quote: str | None = "The MCP must maintain records.",
    reference: str | None = "Test Section, page 1",
) -> ComplianceRequirement:
    return ComplianceRequirement(
        id=0,
        text=text,
        section_heading=section_heading,
        obligation_type=obligation_type,
        exact_quote=exact_quote,
        reference=reference,
    )


# ===========================================================================
# segment_document
# ===========================================================================


class TestSegmentDocument:
    def test_single_heading_section(self):
        text = "[PAGE 1]\n[HEADING 2] Provider Network Requirements\nThe MCP must maintain a network."
        sections = segment_document(text)
        # Preamble (empty) + the heading section
        content_sections = [s for s in sections if s.text.strip()]
        assert any(s.heading == "Provider Network Requirements" for s in content_sections)
        req_section = next(s for s in sections if s.heading == "Provider Network Requirements")
        assert "must maintain a network" in req_section.text

    def test_preamble_before_first_heading(self):
        text = "[PAGE 1]\nIntroduction text here.\n[HEADING 1] Section I\nMore text."
        sections = segment_document(text)
        preamble = next((s for s in sections if s.heading == "Preamble"), None)
        assert preamble is not None
        assert "Introduction text" in preamble.text

    def test_multiple_headings_create_multiple_sections(self):
        text = (
            "[PAGE 1]\n"
            "[HEADING 1] Chapter 1\nChapter 1 body.\n"
            "[PAGE 2]\n"
            "[HEADING 2] Section 1.1\nSection 1.1 body.\n"
            "[HEADING 3] Subsection 1.1.a\nSubsection body."
        )
        sections = segment_document(text)
        headings = [s.heading for s in sections]
        assert "Chapter 1" in headings
        assert "Section 1.1" in headings
        assert "Subsection 1.1.a" in headings

    def test_heading_levels_preserved(self):
        text = (
            "[HEADING 1] Top Level\nTop level body text.\n"
            "[HEADING 2] Second Level\nSecond level body text.\n"
            "[HEADING 3] Third Level\nThird level body text.\n"
        )
        sections = segment_document(text)
        by_heading = {s.heading: s.level for s in sections}
        assert by_heading.get("Top Level") == 1
        assert by_heading.get("Second Level") == 2
        assert by_heading.get("Third Level") == 3

    def test_page_numbers_tracked(self):
        text = (
            "[PAGE 1]\n[HEADING 1] Section A\nText A.\n"
            "[PAGE 5]\n[HEADING 2] Section B\nText B."
        )
        sections = segment_document(text)
        section_b = next(s for s in sections if s.heading == "Section B")
        assert section_b.page_start == 5

    def test_table_accumulated_in_section(self):
        text = (
            "[PAGE 1]\n"
            "[HEADING 2] Compliance Controls\n"
            "Narrative text.\n"
            "[TABLE]\n"
            "| Control | Owner | Frequency |\n"
            "| Training | CO | Annually |\n"
            "[/TABLE]"
        )
        sections = segment_document(text)
        section = next(s for s in sections if s.heading == "Compliance Controls")
        assert len(section.tables) == 1
        assert "Training" in section.tables[0]

    def test_empty_section_merged_into_next(self):
        text = (
            "[HEADING 1] Empty Section\n"
            "[HEADING 2] Content Section\nSome body text here."
        )
        sections = segment_document(text)
        # Empty Section has no text; should not appear as a standalone empty section
        # It should be merged: Content Section absorbs it
        empty_sec = next((s for s in sections if s.heading == "Empty Section"), None)
        # Either empty section is gone or content section absorbed it
        content_sec = next((s for s in sections if s.heading == "Content Section"), None)
        assert content_sec is not None
        assert content_sec.text.strip()

    def test_long_section_sub_split(self):
        # Create a section with text > SECTION_MAX_CHARS (5000)
        big_paragraph_a = "A " * 2600  # ~5200 chars
        big_paragraph_b = "B " * 2600
        text = f"[HEADING 1] Big Section\n{big_paragraph_a}\n\n{big_paragraph_b}"
        sections = segment_document(text)
        big_sections = [s for s in sections if "Big Section" in s.heading]
        assert len(big_sections) >= 2, "Long section should be sub-split"

    def test_empty_string_returns_preamble_only(self):
        sections = segment_document("")
        assert len(sections) == 1
        assert sections[0].heading == "Preamble"

    def test_no_headings_returns_single_preamble(self):
        text = "[PAGE 1]\nJust plain text with no headings."
        sections = segment_document(text)
        assert len(sections) == 1
        assert sections[0].heading == "Preamble"
        assert "plain text" in sections[0].text


# ===========================================================================
# filter_obligation_sections
# ===========================================================================


class TestFilterObligationSections:
    def test_section_with_must_passes(self):
        sec = _make_section(text="The MCP must report within 30 days.")
        kept, skipped = filter_obligation_sections([sec])
        assert sec in kept
        assert sec not in skipped

    def test_section_with_shall_passes(self):
        sec = _make_section(text="Providers shall submit claims monthly.")
        kept, skipped = filter_obligation_sections([sec])
        assert sec in kept

    def test_section_with_prohibited_passes(self):
        sec = _make_section(text="This action is prohibited under the contract.")
        kept, skipped = filter_obligation_sections([sec])
        assert sec in kept

    def test_section_with_timeframe_passes(self):
        sec = _make_section(text="Submit the report within 10 business days.")
        kept, skipped = filter_obligation_sections([sec])
        assert sec in kept

    def test_section_with_no_obligation_skipped(self):
        sec = _make_section(text="This document provides background information.")
        kept, skipped = filter_obligation_sections([sec])
        assert sec not in kept
        assert sec in skipped

    def test_table_always_passes(self):
        sec = _make_section(
            text="This section has no obligation language.",
            tables=["| Control | Owner |\n| Training | CO |"],
        )
        kept, skipped = filter_obligation_sections([sec])
        assert sec in kept

    def test_empty_section_skipped(self):
        sec = _make_section(text="", tables=[])
        kept, skipped = filter_obligation_sections([sec])
        assert sec in skipped

    def test_mixed_sections_split_correctly(self):
        ob_sec = _make_section(text="MCPs must comply with all requirements.")
        no_sec = _make_section(heading="Glossary", text="Definitions of terms used.")
        kept, skipped = filter_obligation_sections([ob_sec, no_sec])
        assert ob_sec in kept
        assert no_sec in skipped

    def test_obligation_patterns_case_insensitive(self):
        sec = _make_section(text="The organization SHALL NOT disclose PHI.")
        kept, skipped = filter_obligation_sections([sec])
        assert sec in kept

    def test_comply_with_pattern(self):
        sec = _make_section(text="The MCP shall comply with HIPAA regulations.")
        kept, skipped = filter_obligation_sections([sec])
        assert sec in kept


# ===========================================================================
# deduplicate_requirements
# ===========================================================================


class TestDeduplicateRequirements:
    def test_empty_list_returned_unchanged(self):
        assert deduplicate_requirements([]) == []

    def test_single_item_returned_unchanged(self):
        req = _make_req()
        result = deduplicate_requirements([req])
        assert result == [req]

    def test_identical_texts_merged_to_one(self):
        text = "Does the P&P state that MCPs must maintain credentialing records?"
        req1 = _make_req(text=text, reference="Section III, page 5")
        req2 = _make_req(text=text, reference="Appendix A, page 45")
        result = deduplicate_requirements([req1, req2], similarity_threshold=0.90)
        assert len(result) == 1

    def test_merged_requirement_combines_references(self):
        text = "Does the P&P state that MCPs must maintain credentialing records?"
        req1 = _make_req(text=text, reference="Section III, page 5")
        req2 = _make_req(text=text, reference="Appendix A, page 45")
        result = deduplicate_requirements([req1, req2], similarity_threshold=0.90)
        assert "Section III" in result[0].reference
        assert "Appendix A" in result[0].reference

    def test_distinct_texts_kept_separate(self):
        req1 = _make_req(text="Does the P&P state that MCPs must provide training annually?")
        req2 = _make_req(text="Does the P&P state that providers must submit claims within 30 days?")
        result = deduplicate_requirements([req1, req2], similarity_threshold=0.90)
        assert len(result) == 2

    def test_canonical_is_most_complete(self):
        """The requirement with more populated fields should be selected as canonical."""
        sparse = ComplianceRequirement(
            id=0,
            text="Does the P&P state that MCPs must maintain quality standards?",
            reference="Section V",
        )
        rich = ComplianceRequirement(
            id=0,
            text="Does the P&P state that MCPs must maintain quality standards?",
            reference="Section V, page 10",
            obligation_type="mandatory",
            actor="MCP",
            action_required="maintain quality standards",
            evidence_needed="audit logs",
            risk_area="Operations",
        )
        result = deduplicate_requirements([sparse, rich], similarity_threshold=0.90)
        assert len(result) == 1
        assert result[0].actor == "MCP"

    def test_threshold_zero_merges_all(self):
        """With threshold=0, everything merges into one cluster."""
        reqs = [_make_req(text=f"Question {i}?") for i in range(5)]
        result = deduplicate_requirements(reqs, similarity_threshold=0.0)
        assert len(result) == 1

    def test_threshold_one_keeps_all_distinct(self):
        """With threshold=1.0, only exact-identical texts merge."""
        req1 = _make_req(text="Does the P&P state that MCPs must provide training?")
        req2 = _make_req(text="Does the P&P state that providers must submit timely claims?")
        result = deduplicate_requirements([req1, req2], similarity_threshold=1.0)
        assert len(result) == 2


# ===========================================================================
# assemble_hierarchy
# ===========================================================================


class TestAssembleHierarchy:
    def test_h1_h2_requirements_have_no_parent(self):
        sections = [
            _make_section(heading="Chapter 1", level=1),
            _make_section(heading="Section 1.1", level=2),
        ]
        reqs = [
            _make_req(text="Q1?", section_heading="Chapter 1"),
            _make_req(text="Q2?", section_heading="Section 1.1"),
        ]
        result = assemble_hierarchy(reqs, sections)
        assert result[0].parent_id is None
        assert result[1].parent_id is None

    def test_h3_requirement_gets_parent_id(self):
        sections = [
            _make_section(heading="Section 1", level=2),
            _make_section(heading="Section 1.A", level=3),
        ]
        reqs = [
            _make_req(text="Parent Q?", section_heading="Section 1"),
            _make_req(text="Child Q?", section_heading="Section 1.A"),
        ]
        result = assemble_hierarchy(reqs, sections)
        assert result[0].parent_id is None
        # Child gets the 0-indexed parent position
        assert result[1].parent_id == 0

    def test_llm_parent_id_overwritten(self):
        """Any LLM-provided parent_id is cleared and recomputed."""
        sections = [_make_section(heading="Section A", level=2)]
        req = _make_req(section_heading="Section A")
        req.parent_id = 999  # bogus LLM value
        result = assemble_hierarchy([req], sections)
        assert result[0].parent_id is None  # cleared (H2 → no parent)

    def test_unknown_section_treated_as_top_level(self):
        """Requirements with unknown section_heading get level=0, treated as parent."""
        sections = [_make_section(heading="Known Section", level=3)]
        req = _make_req(section_heading="Unknown Section")
        result = assemble_hierarchy([req], sections)
        assert result[0].parent_id is None  # level 0 → no parent assigned


# ===========================================================================
# finalize_requirements
# ===========================================================================


class TestFinalizeRequirements:
    def test_ids_assigned_sequentially_from_1(self):
        reqs = [_make_req() for _ in range(5)]
        result = finalize_requirements(reqs)
        assert [r.id for r in result] == [1, 2, 3, 4, 5]

    def test_parent_id_converted_from_0indexed_to_1indexed(self):
        reqs = [
            _make_req(text="Parent?"),
            _make_req(text="Child?"),
        ]
        reqs[1].parent_id = 0  # 0-indexed → should become 1
        result = finalize_requirements(reqs)
        assert result[1].parent_id == 1  # 1-indexed

    def test_none_parent_id_unchanged(self):
        reqs = [_make_req()]
        reqs[0].parent_id = None
        result = finalize_requirements(reqs)
        assert result[0].parent_id is None

    def test_empty_list_returns_empty(self):
        assert finalize_requirements([]) == []


# ===========================================================================
# _parse_compliance_requirements
# ===========================================================================


class TestParseComplianceRequirements:
    def _section(self) -> DocumentSection:
        return _make_section()

    def _item(self, **overrides) -> dict:
        base = {
            "text": "Does the P&P state that MCPs must train staff?",
            "exact_quote": "MCPs must train all staff annually.",
            "reference": "Section IV, page 12",
            "category": "Training",
            "obligation_type": "mandatory",
            "obligation_level": "mandatory",
            "actor": "MCP",
            "action_required": "train staff",
            "condition": None,
            "timeframe": "annually",
            "evidence_needed": "training records",
            "risk_area": "Operations",
        }
        base.update(overrides)
        return base

    def test_parses_list_of_dicts(self):
        data = [self._item()]
        result = _parse_compliance_requirements(data, self._section())
        assert len(result) == 1
        assert result[0].text == "Does the P&P state that MCPs must train staff?"

    def test_parses_json_string(self):
        data = json.dumps([self._item()])
        result = _parse_compliance_requirements(data, self._section())
        assert len(result) == 1

    def test_parses_markdown_fenced_json(self):
        inner = json.dumps([self._item()])
        fenced = f"```json\n{inner}\n```"
        result = _parse_compliance_requirements(fenced, self._section())
        assert len(result) == 1

    def test_extracts_array_from_prose(self):
        inner = json.dumps([self._item()])
        wrapped = f"Here are the requirements:\n{inner}\nEnd."
        result = _parse_compliance_requirements(wrapped, self._section())
        assert len(result) == 1

    def test_injects_placeholder_id(self):
        """The parser should always set id=0 regardless of LLM output."""
        item = self._item()
        item["id"] = 99  # LLM might return an id
        result = _parse_compliance_requirements([item], self._section())
        assert result[0].id == 0

    def test_clears_llm_parent_id(self):
        item = self._item()
        item["parent_id"] = 5  # LLM-provided parent
        result = _parse_compliance_requirements([item], self._section())
        assert result[0].parent_id is None

    def test_sets_section_heading_from_section(self):
        item = self._item()
        section = _make_section(heading="Provider Network")
        result = _parse_compliance_requirements([item], section)
        assert result[0].section_heading == "Provider Network"

    def test_section_heading_from_item_not_overwritten_if_present(self):
        """section_heading from item is preserved if already set."""
        item = self._item()
        item["section_heading"] = "Custom Heading"
        result = _parse_compliance_requirements([item], self._section())
        assert result[0].section_heading == "Custom Heading"

    def test_invalid_json_returns_empty_list(self):
        result = _parse_compliance_requirements("{{{not json", self._section())
        assert result == []

    def test_non_array_json_returns_empty_list(self):
        result = _parse_compliance_requirements('{"text": "Q?"}', self._section())
        assert result == []

    def test_skips_malformed_items(self):
        """Items that fail validation are skipped; valid ones are returned."""
        items = [self._item(), {"bad": "no text field"}]
        result = _parse_compliance_requirements(items, self._section())
        assert len(result) == 1  # only the valid item

    def test_empty_list_returns_empty(self):
        result = _parse_compliance_requirements([], self._section())
        assert result == []

    def test_optional_fields_populated(self):
        item = self._item()
        result = _parse_compliance_requirements([item], self._section())
        r = result[0]
        assert r.obligation_type == "mandatory"
        assert r.actor == "MCP"
        assert r.timeframe == "annually"
        assert r.evidence_needed == "training records"
        assert r.risk_area == "Operations"


# ===========================================================================
# GetSectionTextTool
# ===========================================================================


class TestGetSectionTextTool:
    """GetSectionTextTool: no-input, returns raw section text + tables."""

    def test_interface_contract(self):
        tool = GetSectionTextTool(_make_section())
        assert tool.name == "get_section_text"
        assert tool.description
        assert tool.inputs == {}
        assert tool.output_type == "string"

    def test_returns_section_content(self):
        section = _make_section(
            heading="Provider Network Requirements",
            text="The MCP must maintain a network.",
            page_start=3,
            page_end=5,
        )
        output = GetSectionTextTool(section)()
        assert "Provider Network Requirements" in output
        assert "The MCP must maintain a network." in output
        assert "3" in output

    def test_tables_included_in_output(self):
        section = _make_section(
            text="Body text.",
            tables=["| Control | Owner |\n| Training | CO |"],
        )
        output = GetSectionTextTool(section)()
        assert "Training" in output
        assert "[TABLE 1]" in output
        assert "[/TABLE 1]" in output

    def test_no_referenced_sections_block(self):
        """GetSectionTextTool no longer auto-resolves cross-refs (separate tool)."""
        section = _make_section(text="See Section III.A for details.")
        output = GetSectionTextTool(section)()
        assert "REFERENCED SECTIONS" not in output

    def test_empty_section_text(self):
        output = GetSectionTextTool(_make_section(text="", tables=[]))()
        assert isinstance(output, str)


# ===========================================================================
# StripHeadersFootersTool
# ===========================================================================


class TestStripHeadersFootersTool:
    """StripHeadersFootersTool: text → text, drops header/footer lines."""

    def test_interface_contract(self):
        tool = StripHeadersFootersTool(frozenset())
        assert tool.name == "strip_headers_footers"
        assert "text" in tool.inputs
        assert tool.inputs["text"]["type"] == "string"
        assert tool.output_type == "string"

    def test_empty_signatures_is_passthrough(self):
        tool = StripHeadersFootersTool(frozenset())
        raw = "Body text line 1.\nBody text line 2."
        assert tool(text=raw) == raw

    def test_strips_known_header_line(self):
        sigs = frozenset({"dhcs apl #-#"})  # matches "DHCS APL 24-001"
        tool = StripHeadersFootersTool(sigs)
        raw = "DHCS APL 24-001\nThe MCP must maintain records.\nDHCS APL 24-001"
        result = tool(text=raw)
        assert "DHCS APL 24-001" not in result
        assert "The MCP must maintain records." in result

    def test_strips_page_number_line(self):
        sigs = frozenset({"page #"})  # matches "Page 3", "Page 17"
        tool = StripHeadersFootersTool(sigs)
        raw = "Page 3\nThe MCP must train staff annually."
        result = tool(text=raw)
        assert "Page 3" not in result
        assert "train staff" in result

    def test_preserves_body_lines_not_in_sigs(self):
        sigs = frozenset({"dhcs apl #-#"})
        tool = StripHeadersFootersTool(sigs)
        body = "MCPs shall maintain a credentialing program."
        result = tool(text=body)
        assert body in result

    def test_multiline_mixed_content(self):
        sigs = frozenset({"dhcs apl #-#", "page #"})
        tool = StripHeadersFootersTool(sigs)
        raw = (
            "DHCS APL 24-001\n"
            "Section III: Credentialing\n"
            "The MCP must maintain records.\n"
            "Page 12"
        )
        result = tool(text=raw)
        lines = result.splitlines()
        assert "DHCS APL 24-001" not in lines
        assert "Page 12" not in lines
        assert "Section III: Credentialing" in result
        assert "The MCP must maintain records." in result


# ===========================================================================
# ResolveCrossReferenceTool
# ===========================================================================


class TestResolveCrossReferenceTool:
    """ResolveCrossReferenceTool: section_id → referenced section text."""

    def _index_with_sections(self):
        from backend.tools.cross_reference_resolver import build_section_index
        sections = [
            _make_section(
                heading="III.A Initial Credentialing",
                text="Providers must be credentialed within 60 days.",
            ),
            _make_section(
                heading="Appendix B Definitions",
                text="Credentialing: the process by which providers are evaluated.",
            ),
        ]
        return build_section_index(sections)

    def test_interface_contract(self):
        from backend.tools.cross_reference_resolver import SectionIndex
        tool = ResolveCrossReferenceTool(SectionIndex())
        assert tool.name == "resolve_cross_reference"
        assert "section_id" in tool.inputs
        assert tool.inputs["section_id"]["type"] == "string"
        assert tool.output_type == "string"

    def test_resolves_known_section(self):
        tool = ResolveCrossReferenceTool(self._index_with_sections())
        result = tool(section_id="III.A")
        assert "60 days" in result
        assert "III.A" in result

    def test_resolves_appendix(self):
        tool = ResolveCrossReferenceTool(self._index_with_sections())
        result = tool(section_id="Appendix B")
        assert "Credentialing" in result

    def test_unknown_section_returns_not_found(self):
        tool = ResolveCrossReferenceTool(self._index_with_sections())
        result = tool(section_id="ZZZZ.999")
        assert "No section found" in result

    def test_empty_section_id_returns_error(self):
        from backend.tools.cross_reference_resolver import SectionIndex
        tool = ResolveCrossReferenceTool(SectionIndex())
        result = tool(section_id="")
        assert "Error" in result or "non-empty" in result

    def test_output_includes_heading_and_body(self):
        tool = ResolveCrossReferenceTool(self._index_with_sections())
        result = tool(section_id="III.A")
        assert "Section:" in result
        assert "credentialed" in result.lower()


# ===========================================================================
# extract_table_requirements
# ===========================================================================


class TestExtractTableRequirements:
    def test_known_column_headers_mapped(self):
        table = (
            "| Requirement | Responsible Party | Deadline |\n"
            "| Annual training | Compliance Officer | Q1 each year |"
        )
        result = extract_table_requirements(table, "Section III", 5)
        assert len(result) == 1
        req = result[0]
        assert "Annual training" in req.text
        assert req.actor == "Compliance Officer"
        assert req.timeframe == "Q1 each year"

    def test_multiple_rows_produce_multiple_requirements(self):
        table = (
            "| Requirement | Responsible Party |\n"
            "| Training completed | CO |\n"
            "| Records maintained | Records Manager |"
        )
        result = extract_table_requirements(table, "Section IV", 10)
        assert len(result) == 2

    def test_reference_set_correctly(self):
        table = (
            "| Requirement |\n"
            "| Submit reports |"
        )
        result = extract_table_requirements(table, "Appendix A", 99)
        assert len(result) == 1
        assert "Appendix A" in result[0].reference
        assert "99" in result[0].reference

    def test_empty_table_returns_empty(self):
        result = extract_table_requirements("", "Section I", 1)
        assert result == []

    def test_unknown_headers_trigger_llm_fallback(self):
        """When column headers are unrecognizable, fall back to run_section_extractor."""
        table = "| Alpha | Beta | Gamma |\n| x | y | z |"
        mock_reqs = [_make_req()]
        with patch(
            "backend.agents.compliance_extractor.run_section_extractor",
            return_value=mock_reqs,
        ) as mock_fn:
            result = extract_table_requirements(table, "Unknown Table", 1)
        mock_fn.assert_called_once()
        assert result == mock_reqs


# ===========================================================================
# _initials_match (term_extractor helper)
# ===========================================================================


class TestInitialsMatch:
    def test_ecm_matches(self):
        assert _initials_match("Enhanced Care Management", "ECM")

    def test_pof_matches_with_stop_word(self):
        assert _initials_match("Population of Focus", "POF")

    def test_mcp_matches(self):
        assert _initials_match("Managed Care Plan", "MCP")

    def test_case_insensitive_acronym(self):
        assert _initials_match("Enhanced Care Management", "ecm")

    def test_unrelated_phrase_does_not_match(self):
        assert not _initials_match("Random Unrelated Words", "ECM")

    def test_empty_acronym_returns_false(self):
        assert not _initials_match("Enhanced Care Management", "")

    def test_empty_phrase_returns_false(self):
        assert not _initials_match("", "ECM")


# ===========================================================================
# _parse_pipe_table (term_extractor helper)
# ===========================================================================


class TestParsePipeTable:
    def test_parses_headers_and_rows(self):
        table = "| Term | Definition |\n| ECM | Enhanced Care Management |"
        headers, rows = _parse_pipe_table(table)
        assert headers == ["term", "definition"]
        assert rows == [["ECM", "Enhanced Care Management"]]

    def test_empty_string_returns_empty(self):
        headers, rows = _parse_pipe_table("")
        assert headers == []
        assert rows == []

    def test_multi_row_table(self):
        table = (
            "| Criteria | Definition |\n"
            "| Mental Illness | A serious mental disorder... |\n"
            "| SUD | Substance Use Disorder |"
        )
        headers, rows = _parse_pipe_table(table)
        assert len(rows) == 2
        assert rows[0][0] == "Mental Illness"


# ===========================================================================
# extract_term_definitions
# ===========================================================================


class TestExtractTermDefinitions:
    _SOURCE = "data/test.pdf"

    def test_inline_acronym_detected(self):
        """A 'Term (ACRONYM)' pattern in the text produces an inline_acronym entry."""
        text = "[PAGE 1]\nEnhanced Care Management (ECM) is a whole-person approach."
        sections = [_make_section(heading="Preamble", text=text)]
        terms = extract_term_definitions(text, sections, self._SOURCE)
        ecm = next((t for t in terms if t.abbreviation == "ECM"), None)
        assert ecm is not None, "ECM not found in extracted terms"
        assert ecm.term == "Enhanced Care Management"
        assert ecm.source == "inline_acronym"

    def test_case_variant_acronym(self):
        """Lower-case input acronym string still produces the correct entry."""
        text = "[PAGE 1]\nPopulation of Focus (POF) members are identified by MCPs."
        sections = [_make_section(heading="Preamble", text=text)]
        terms = extract_term_definitions(text, sections, self._SOURCE)
        pof = next((t for t in terms if t.abbreviation == "POF"), None)
        assert pof is not None
        assert pof.term == "Population of Focus"

    def test_glossary_table_produces_entries(self):
        """A 'Criteria | Definition' two-column table in a Definitions section."""
        table = (
            "| Criteria | Definition |\n"
            "| Mental Illness | A serious mental health condition that substantially limits daily functioning. |"
        )
        section = _make_section(
            heading="Definitions",
            text="",
            tables=[table],
            page_start=10,
        )
        terms = extract_term_definitions("", [section], self._SOURCE)
        mi = next((t for t in terms if t.term == "Mental Illness"), None)
        assert mi is not None
        assert mi.source == "glossary"
        assert mi.page_number == 10

    def test_glossary_wins_over_inline_for_same_term(self):
        """
        When both a glossary entry and an inline acronym exist for the same term,
        the glossary entry should be kept and the inline acronym dropped.
        """
        # Glossary section with full definition of ECM
        table = "| Term | Definition |\n| Enhanced Care Management | A whole-person, interdisciplinary approach. |"
        glossary_section = _make_section(
            heading="Definitions",
            text="",
            tables=[table],
            page_start=5,
        )
        # Regular section with inline acronym
        body_text = "[PAGE 2]\nEnhanced Care Management (ECM) services are provided by MCPs."
        body_section = _make_section(heading="Section I", text=body_text, page_start=2)
        sections = [glossary_section, body_section]

        terms = extract_term_definitions(body_text, sections, self._SOURCE)
        ecm_terms = [t for t in terms if "enhanced care management" in t.term.lower()]
        assert len(ecm_terms) == 1, "Duplicate ECM entries should be deduplicated"
        assert ecm_terms[0].source == "glossary", "Glossary entry should take precedence"

    def test_miss_returns_empty(self):
        """A document with no acronyms or glossary produces an empty list."""
        text = "[PAGE 1]\nThis document contains only plain prose with no special terms."
        sections = [_make_section(heading="Preamble", text=text)]
        terms = extract_term_definitions(text, sections, self._SOURCE)
        assert terms == []

    def test_definition_table_in_regular_section(self):
        """A 'Term | Definition' table in a non-glossary section is captured."""
        table = (
            "| Term | Definition |\n"
            "| Credentialing | The process of verifying provider qualifications. |"
        )
        section = _make_section(
            heading="Section III Provider Requirements",
            text="",
            tables=[table],
            page_start=20,
        )
        terms = extract_term_definitions("", [section], self._SOURCE)
        cred = next((t for t in terms if t.term == "Credentialing"), None)
        assert cred is not None
        assert cred.source == "definition_table"

    def test_no_definitions_no_error(self):
        """A document with no definitions section and no acronyms returns empty without error."""
        sections = [
            _make_section(heading="Section I", text="The MCP must maintain records."),
        ]
        terms = extract_term_definitions("The MCP must maintain records.", sections, self._SOURCE)
        # Should return empty list (no acronyms with matching initials, no glossary)
        assert isinstance(terms, list)


# ===========================================================================
# upsert_term_definitions
# ===========================================================================


class TestUpsertTermDefinitions:
    def _make_term(
        self,
        term: str = "Enhanced Care Management",
        abbreviation: str | None = "ECM",
        definition: str = "A whole-person care approach.",
        source_file: str = "data/test.pdf",
        page_number: int = 5,
    ) -> TermDefinition:
        return TermDefinition(
            term=term,
            abbreviation=abbreviation,
            definition=definition,
            source_file=source_file,
            page_number=page_number,
            source="glossary",
        )

    def test_empty_list_is_noop(self):
        """Calling upsert with an empty list should not touch ChromaDB."""
        with patch("backend.tools.term_extractor._get_chroma_client") as mock_client:
            upsert_term_definitions([])
        mock_client.assert_not_called()

    def test_upsert_calls_collection_upsert(self):
        """Upsert should call collection.upsert with correct ids and documents."""
        terms = [self._make_term()]
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("backend.tools.term_extractor._get_chroma_client", return_value=mock_client):
            upsert_term_definitions(terms)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        ids = call_kwargs.kwargs.get("ids") or call_kwargs.args[0]
        assert ids == ["data/test.pdf::ECM"]
        documents = call_kwargs.kwargs.get("documents") or call_kwargs.args[1]
        assert "Enhanced Care Management (ECM)" in documents[0]

    def test_id_uses_abbreviation_when_present(self):
        """Document ID should use the abbreviation when available."""
        term = self._make_term(abbreviation="ECM")
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("backend.tools.term_extractor._get_chroma_client", return_value=mock_client):
            upsert_term_definitions([term])

        call_kwargs = mock_collection.upsert.call_args
        ids = call_kwargs.kwargs.get("ids") or call_kwargs.args[0]
        assert "ECM" in ids[0]

    def test_id_falls_back_to_term_when_no_abbreviation(self):
        """Document ID should use the term name when abbreviation is None."""
        term = self._make_term(abbreviation=None)
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("backend.tools.term_extractor._get_chroma_client", return_value=mock_client):
            upsert_term_definitions([term])

        call_kwargs = mock_collection.upsert.call_args
        ids = call_kwargs.kwargs.get("ids") or call_kwargs.args[0]
        assert "Enhanced Care Management" in ids[0]

    def test_metadata_has_no_none_values(self):
        """ChromaDB metadata must not contain None values."""
        term = self._make_term(abbreviation=None)  # section_heading is also None
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("backend.tools.term_extractor._get_chroma_client", return_value=mock_client):
            upsert_term_definitions([term])

        call_kwargs = mock_collection.upsert.call_args
        metadatas = call_kwargs.kwargs.get("metadatas") or call_kwargs.args[2]
        for meta in metadatas:
            assert None not in meta.values(), "Metadata must not contain None values"


# ===========================================================================
# _run_section_with_timeout / extract_from_all_sections
# ---------------------------------------------------------------------------
# Regression coverage for the "Step 5/8 stuck at 52/54 sections" bug.
#
# Prior to the fix the per-section worker was dispatched via
# `asyncio.to_thread(run_section_extractor, ...)` with no surrounding
# `asyncio.wait_for`, so a single stalled LLM call hung the entire pipeline:
# `asyncio.as_completed` could never advance past the stuck task.
#
# These tests patch `run_section_extractor` with a synthetic blocking
# implementation and assert that:
#   1. A single hanging section times out and returns [] instead of hanging.
#   2. `extract_from_all_sections` completes even when some sections block
#      forever, returning results from the healthy sections.
# ===========================================================================


class TestRunSectionWithTimeout:
    """Unit-level coverage of the per-section timeout helper."""

    def test_fast_section_returns_results(self):
        """A fast worker returns its results unchanged (no timeout path)."""
        section = _make_section(heading="Fast Section")
        expected = [_make_req(text="Does the P&P state that MCPs must train?")]

        with patch(
            "backend.agents.compliance_extractor.run_section_extractor",
            return_value=expected,
        ) as mock_extractor:
            result = asyncio.run(
                _run_section_with_timeout(section, None, None, timeout_seconds=5.0)
            )

        assert result == expected
        mock_extractor.assert_called_once()

    def test_stuck_section_times_out_and_returns_empty(self):
        """A worker that blocks past the timeout must yield [] instead of hanging."""
        section = _make_section(heading="Stuck Section")
        # A real smolagents/LLM hang manifests as a blocking call inside the
        # worker thread.  The release event lets us unblock the thread once
        # the coroutine has returned so that the asyncio event loop can tear
        # down without waiting the full safety-net duration on its default
        # executor.  We measure elapsed time INSIDE the coroutine because
        # `asyncio.run()` waits for the threadpool to drain during shutdown —
        # a concern that does not affect production where the event loop
        # keeps running and simply advances past the stuck task.
        release = threading.Event()

        def blocking_extractor(*args, **kwargs):
            release.wait(timeout=10.0)
            return []

        elapsed: dict[str, float] = {}

        async def run_and_measure():
            start = time.monotonic()
            result = await _run_section_with_timeout(
                section, None, None, timeout_seconds=0.1
            )
            elapsed["value"] = time.monotonic() - start
            # Release the blocked thread now that we've measured the
            # coroutine's return time; this keeps event-loop teardown fast.
            release.set()
            return result

        try:
            with patch(
                "backend.agents.compliance_extractor.run_section_extractor",
                side_effect=blocking_extractor,
            ):
                result = asyncio.run(run_and_measure())
        finally:
            release.set()

        assert result == []
        # The coroutine itself must return promptly after the timeout, not
        # block on the stuck thread.  Without `asyncio.wait_for` around
        # `asyncio.to_thread`, this would be >10s.
        assert elapsed["value"] < 1.0, (
            f"Timeout path took {elapsed['value']:.2f}s inside the coroutine; "
            "expected <1s. The asyncio.wait_for wrapper is likely missing."
        )


class TestExtractFromAllSectionsTimeout:
    """Regression test: one stuck section must not stall the whole pipeline."""

    def test_stuck_section_does_not_stall_pipeline(self):
        """
        Simulate the production symptom: 1 of 3 sections hangs indefinitely.

        With the fix, the async pipeline must still complete and yield the
        results from the 2 healthy sections.  Without the fix, this test
        hangs until the test runner's own timeout kicks in.

        Elapsed time is measured INSIDE the coroutine because
        ``asyncio.run`` waits for default-executor threads to drain at
        shutdown — a teardown artefact that does not represent production
        behaviour (the production event loop keeps running past stuck tasks).
        """
        sections = [
            _make_section(heading="Fast Section 1"),
            _make_section(heading="Stuck Section"),
            _make_section(heading="Fast Section 2"),
        ]
        release = threading.Event()

        def side_effect(section, section_index=None, hf_filter=None):
            if section.heading == "Stuck Section":
                # Simulate a hung LLM call: block until the test releases us.
                release.wait(timeout=10.0)
                return []
            return [
                _make_req(
                    text=f"Does the P&P state that {section.heading}?",
                    section_heading=section.heading,
                )
            ]

        elapsed: dict[str, float] = {}

        async def run_and_measure():
            start = time.monotonic()
            results = await extract_from_all_sections(sections)
            elapsed["value"] = time.monotonic() - start
            # Release the stuck thread so loop teardown is fast.
            release.set()
            return results

        try:
            with (
                patch(
                    "backend.agents.compliance_extractor.WORKER_TIMEOUT_SECONDS",
                    0.2,
                ),
                patch(
                    "backend.agents.compliance_extractor.run_section_extractor",
                    side_effect=side_effect,
                ),
            ):
                results = asyncio.run(
                    asyncio.wait_for(run_and_measure(), timeout=10.0)
                )
        finally:
            release.set()

        # The two healthy sections must have produced requirements; the stuck
        # one contributes nothing because it timed out.
        headings = {r.section_heading for r in results}
        assert headings == {"Fast Section 1", "Fast Section 2"}
        assert len(results) == 2
        # The pipeline coroutine itself must complete roughly on the
        # timeout boundary (0.2s), not wait for the 10s thread-release
        # safety net.  Prior to the fix this would be >10s.
        assert elapsed["value"] < 2.0, (
            f"Pipeline coroutine took {elapsed['value']:.2f}s; expected <2s "
            "with a 0.2s per-section timeout. Regression in timeout wrapper."
        )

    def test_progress_advances_past_stuck_section(self):
        """
        Streaming pipeline must emit a progress event for every section,
        including stuck ones (which contribute [] via the timeout path).

        This test exercises `run_compliance_extractor_with_progress` directly
        so the SSE frontend's "X / N sections" counter is proven to advance.
        """
        from backend.agents import compliance_extractor as mod

        sections = [
            _make_section(heading="Fast 1", text="MCPs must train staff."),
            _make_section(heading="Stuck", text="Providers must report issues."),
            _make_section(heading="Fast 2", text="The MCP shall maintain records."),
        ]
        release = threading.Event()

        def side_effect(section, section_index=None, hf_filter=None):
            if section.heading == "Stuck":
                release.wait(timeout=5.0)
                return []
            return [
                _make_req(
                    text=f"Does the P&P describe {section.heading}?",
                    section_heading=section.heading,
                )
            ]

        async def collect_events():
            events = []
            async for event in mod.run_compliance_extractor_with_progress(
                "/tmp/fake.pdf"
            ):
                events.append(event)
            return events

        async def collect_and_release():
            # Release the stuck thread once we've finished iterating so the
            # default executor can drain quickly at event-loop shutdown.
            try:
                return await collect_events()
            finally:
                release.set()

        try:
            with (
                patch.object(mod, "WORKER_TIMEOUT_SECONDS", 0.2),
                patch.object(
                    mod, "build_hf_filter_from_pdf", return_value=None
                ),
                patch.object(
                    mod,
                    "parse_pdf_with_structure",
                    return_value="[PAGE 1]\nBody text.",
                ),
                patch.object(mod, "segment_document", return_value=sections),
                patch.object(
                    mod, "filter_obligation_sections",
                    return_value=(sections, []),
                ),
                patch.object(
                    mod, "extract_term_definitions", return_value=[]
                ),
                patch.object(mod, "upsert_term_definitions"),
                patch.object(mod, "deduplicate_requirements", side_effect=lambda r: r),
                patch.object(
                    mod, "run_section_extractor", side_effect=side_effect
                ),
            ):
                events = asyncio.run(
                    asyncio.wait_for(collect_and_release(), timeout=10.0)
                )
        finally:
            release.set()

        # Collect the sub-progress events for Step 5.
        step5 = [
            e for e in events
            if e.get("type") == "progress" and e.get("step_number") == 5
        ]
        completions = [
            e["sections_completed"] for e in step5
            if "sections_completed" in e
        ]
        # Counter must reach the total (3/3) — this is what the frontend's
        # progress bar reads.  Prior to the fix it stopped at 2/3.
        assert completions and max(completions) == len(sections)

        # Pipeline must emit a final "complete" event.
        complete_events = [e for e in events if e.get("type") == "complete"]
        assert len(complete_events) == 1
        final = complete_events[0]
        # The two healthy sections produced one requirement each.
        assert final["total_requirements"] == 2
