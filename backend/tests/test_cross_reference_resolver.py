"""
Tests for cross_reference_resolver.py.
"""

from __future__ import annotations

from backend.tools.cross_reference_resolver import (
    CrossReference,
    ResolvedReference,
    SectionIndex,
    _extract_section_number,
    _normalize_key,
    build_section_index,
    detect_cross_references,
    format_resolved_context,
    resolve_references,
)
from backend.tools.document_segmenter import DocumentSection


def _sec(
    heading: str,
    text: str = "",
    level: int = 2,
    tables: list[str] | None = None,
) -> DocumentSection:
    return DocumentSection(
        heading=heading,
        level=level,
        text=text,
        tables=tables or [],
        page_start=1,
        page_end=1,
    )


class TestDetectCrossReferences:
    def test_detects_section_reference(self):
        text = "The MCP shall comply with the requirements in Section III.A."
        refs = detect_cross_references(text)
        assert any(r.target_key.upper() == "III.A" for r in refs)

    def test_detects_appendix_reference(self):
        text = "As defined in Appendix B, providers must maintain records."
        refs = detect_cross_references(text)
        assert any(r.target_key.upper() == "B" for r in refs)

    def test_detects_cfr_reference(self):
        text = "Per 42 CFR § 438.210, plans must provide coverage."
        refs = detect_cross_references(text)
        assert any(r.ref_type == "cfr" for r in refs)

    def test_deduplicates_same_target(self):
        text = "See Section III.A. The requirements in Section III.A are mandatory."
        refs = detect_cross_references(text)
        targets = [r.target_key.lower() for r in refs]
        assert targets.count("iii.a") == 1


class TestSectionIndex:
    def test_resolve_by_full_heading(self):
        index = build_section_index([_sec("Section III.A Credentialing", "Credentialing details.")])
        result = index.resolve("Section III.A Credentialing")
        assert result is not None
        heading, summary = result
        assert heading == "Section III.A Credentialing"

    def test_resolve_by_section_number(self):
        index = build_section_index([_sec("Section III.A Credentialing", "Credentialing details.")])
        result = index.resolve("III.A")
        assert result is not None
        heading, summary = result
        assert heading == "Section III.A Credentialing"

    def test_resolve_appendix_letter(self):
        index = build_section_index([_sec("Appendix B Definitions", "Definition text.")])
        result = index.resolve("B")
        assert result is not None
        heading, summary = result
        assert heading == "Appendix B Definitions"

    def test_resolve_prefix_match(self):
        index = build_section_index([_sec("III.A Credentialing", "Credentialing details.")])
        result = index.resolve("III")
        assert result is not None


class TestResolveReferences:
    def test_resolves_internal_section(self):
        sections = [
            _sec("Section III.A Credentialing", "Plans must credential providers."),
            _sec("Section IV Reporting", "See Section III.A for details."),
        ]
        index = build_section_index(sections)
        resolved = resolve_references(
            sections[1].text,
            index,
            current_heading=sections[1].heading,
        )
        assert len(resolved) == 1
        assert resolved[0].heading == "Section III.A Credentialing"

    def test_skips_cfr_reference(self):
        sections = [_sec("Section IV Reporting", "Per 42 CFR § 438.210, plans must report.")]
        index = build_section_index(sections)
        resolved = resolve_references(sections[0].text, index, current_heading=sections[0].heading)
        assert resolved == []

    def test_skips_self_reference(self):
        sec = _sec("Section III.A Credentialing", "See Section III.A for details.")
        index = build_section_index([sec])
        resolved = resolve_references(sec.text, index, current_heading=sec.heading)
        assert resolved == []

    def test_max_refs_respected(self):
        sections = [
            _sec("Section I", "Text 1"),
            _sec("Section II", "Text 2"),
            _sec("Section III", "Text 3"),
            _sec("Section IV", "See Section I, Section II, and Section III."),
        ]
        index = build_section_index(sections)
        resolved = resolve_references(sections[3].text, index, max_refs=2, current_heading=sections[3].heading)
        assert len(resolved) <= 2


class TestFormatResolvedContext:
    def test_empty_returns_empty_string(self):
        assert format_resolved_context([]) == ""

    def test_formats_resolved_references(self):
        resolved = resolve_references(
            "See Section III.A.",
            build_section_index([_sec("Section III.A Credentialing", "Plans must credential providers.")]),
        )
        output = format_resolved_context(resolved)
        assert "REFERENCED SECTIONS" in output
        assert "Section III.A Credentialing" in output
        assert "Plans must credential providers." in output

    def test_raw_reference_text_included(self):
        resolved = resolve_references(
            "See Section III.A.",
            build_section_index([_sec("Section III.A Credentialing", "Details.")]),
        )
        output = format_resolved_context(resolved)
        assert "Section III.A" in output


# ===========================================================================
# Helper function unit tests
# ===========================================================================


class TestNormalizeKey:
    def test_lowercase(self):
        assert _normalize_key("III.A") == "iii.a"

    def test_strips_trailing_dot(self):
        assert _normalize_key("IV.B.") == "iv.b"

    def test_collapses_whitespace(self):
        assert _normalize_key("  Section  IV  ") == "section iv"

    def test_empty_string(self):
        assert _normalize_key("") == ""


class TestExtractSectionNumber:
    def test_roman_numeral_prefix(self):
        assert _extract_section_number("III.A Credentialing") == "III.A"

    def test_numeric_prefix(self):
        assert _extract_section_number("4.2 Provider Responsibilities") == "4.2"

    def test_appendix_letter(self):
        assert _extract_section_number("Appendix B Definitions") == "B"

    def test_section_word_prefix(self):
        assert _extract_section_number("Section 3.1 Requirements") == "3.1"

    def test_no_match(self):
        assert _extract_section_number("Preamble") is None


class TestAdditionalDetectionPatterns:
    def test_pursuant_to_section(self):
        refs = detect_cross_references("Pursuant to Section III, the MCP shall maintain records.")
        assert any("III" in r.target_key for r in refs)

    def test_requirements_set_forth_in(self):
        refs = detect_cross_references("The requirements set forth in Section IV apply here.")
        assert any("IV" in r.target_key for r in refs)

    def test_in_accordance_with_appendix(self):
        refs = detect_cross_references("In accordance with Appendix A, reports must be filed.")
        assert any(r.ref_type in ("section", "appendix") for r in refs)

    def test_no_refs_in_clean_text(self):
        refs = detect_cross_references("The MCP shall maintain training records annually.")
        assert refs == []

    def test_empty_text(self):
        assert detect_cross_references("") == []


class TestSectionIndexTruncation:
    def test_long_text_is_truncated(self):
        long_text = "word " * 200
        idx = SectionIndex()
        idx.register("Long Section", long_text, [])
        result = idx.resolve("Long Section", max_chars=100)
        assert result is not None
        _, summary = result
        assert len(summary) <= 115  # small buffer for ellipsis
        assert summary.endswith("\u2026") or summary.endswith("...")

    def test_short_text_not_truncated(self):
        text = "Short body."
        idx = SectionIndex()
        idx.register("Short Section", text, [])
        result = idx.resolve("Short Section")
        assert result is not None
        _, summary = result
        assert summary == text


# ===========================================================================
# Integration: smolagents extraction tools work together
# ===========================================================================


class TestExtractionToolsIntegration:
    """
    Verify that the three smolagents tools (GetSectionTextTool,
    StripHeadersFootersTool, ResolveCrossReferenceTool) work correctly
    in the workflow the agent is instructed to follow:
    get → strip → resolve → final_answer.
    """

    def test_strip_tool_cleans_header_noise_from_get_text_output(self):
        """The strip tool removes known header/footer lines from raw section text."""
        from backend.agents.compliance_extractor import (
            GetSectionTextTool,
            StripHeadersFootersTool,
        )

        sec = _sec(
            heading="III Credentialing",
            text="DHCS APL 24-001\nMCPs must maintain records.\nPage 3",
        )
        raw = GetSectionTextTool(sec)()
        assert "DHCS APL 24-001" in raw  # present before stripping

        # Simulate what the agent does: strip with the doc's repeated sigs
        sigs = frozenset({"dhcs apl #-#", "page #"})
        cleaned = StripHeadersFootersTool(sigs)(text=raw)
        assert "DHCS APL 24-001" not in cleaned
        assert "MCPs must maintain records." in cleaned

    def test_resolve_tool_fetches_cross_referenced_content(self):
        """The resolve tool returns referenced section content for LLM context."""
        from backend.agents.compliance_extractor import ResolveCrossReferenceTool

        idx = build_section_index([
            _sec("III.A Initial Credentialing",
                 "Providers must be credentialed within 60 days of contract."),
        ])
        tool = ResolveCrossReferenceTool(idx)
        result = tool(section_id="III.A")
        assert "60 days" in result
        assert "III.A" in result

    def test_get_text_tool_no_longer_auto_resolves_refs(self):
        """GetSectionTextTool is now plain — no REFERENCED SECTIONS block."""
        from backend.agents.compliance_extractor import GetSectionTextTool

        sec = _sec(
            heading="IV Quality",
            text="Per Section III.A, MCPs must credential providers.",
        )
        output = GetSectionTextTool(sec)()
        assert "REFERENCED SECTIONS" not in output
        # The raw reference is still present — the agent calls resolve_cross_reference
        assert "Section III.A" in output
