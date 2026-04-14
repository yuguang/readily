"""
Deterministic document segmentation for structured PDF text.

Takes the output of ``parse_pdf_with_structure`` (text with ``[PAGE N]``,
``[HEADING N]``, and ``[TABLE]...[/TABLE]`` markers) and splits it into
:class:`DocumentSection` objects, one per logical heading.

Used exclusively by the compliance extraction agent (Component 8) for
long regulatory PDFs.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from backend.config import SECTION_MAX_CHARS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Obligation language patterns (rule-based pre-filter)
# ---------------------------------------------------------------------------

OBLIGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:must|shall|required to|is responsible for)\b", re.IGNORECASE),
    re.compile(r"\b(?:prohibited|may not|must not|shall not)\b", re.IGNORECASE),
    re.compile(r"\b(?:should|expected to|is expected)\b", re.IGNORECASE),
    re.compile(r"\b(?:within \d+ (?:days|business days|calendar days))\b", re.IGNORECASE),
    re.compile(r"\b(?:annually|quarterly|monthly|upon request|no later than)\b", re.IGNORECASE),
    re.compile(r"\b(?:comply with|in accordance with|pursuant to)\b", re.IGNORECASE),
    re.compile(r"\b(?:ensure that|maintain|establish|implement|develop)\b", re.IGNORECASE),
]

# Regex to match [HEADING N] and [PAGE N] markers
_HEADING_RE = re.compile(r"^\[HEADING (\d+)\] (.+)$")
_PAGE_RE = re.compile(r"^\[PAGE (\d+)\]$")
_TABLE_START = "[TABLE]"
_TABLE_END = "[/TABLE]"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DocumentSection:
    """One logical section extracted from a structured regulatory PDF."""

    heading: str            # Section heading text
    level: int              # Heading level (1, 2, 3)
    text: str               # Full section body text (excluding tables)
    tables: list[str] = field(default_factory=list)   # Tables within this section
    page_start: int = 1     # Starting page number
    page_end: int = 1       # Ending page number
    char_count: int = 0     # Length for chunking decisions

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _sub_split_section(section: DocumentSection) -> list[DocumentSection]:
    """
    Split a section that exceeds SECTION_MAX_CHARS on paragraph boundaries.

    Each sub-section inherits the parent heading and level but gets a
    disambiguating suffix (e.g. ``" (cont.)"``).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", section.text) if p.strip()]
    if not paragraphs:
        return [section]

    sub_sections: list[DocumentSection] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        sep = 2 if current_parts else 0
        if current_len + sep + len(para) > SECTION_MAX_CHARS and current_parts:
            sub_sections.append(
                DocumentSection(
                    heading=section.heading + (" (cont.)" if sub_sections else ""),
                    level=section.level,
                    text="\n\n".join(current_parts),
                    tables=section.tables if not sub_sections else [],
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += sep + len(para)

    if current_parts:
        sub_sections.append(
            DocumentSection(
                heading=section.heading + (" (cont.)" if sub_sections else ""),
                level=section.level,
                text="\n\n".join(current_parts),
                tables=section.tables if not sub_sections else [],
                page_start=section.page_start,
                page_end=section.page_end,
            )
        )

    return sub_sections if sub_sections else [section]


def segment_document(structured_text: str) -> list[DocumentSection]:
    """
    Split structured PDF text into sections based on heading markers.

    Handles:
    - Nested headings (H1 > H2 > H3)
    - Tables extracted as separate items within their parent section
    - Appendices and annexes as top-level sections
    - Text before the first heading grouped as a ``"Preamble"`` section
    - Empty sections (heading with no body) merged into the next section
    - Sections longer than ``SECTION_MAX_CHARS`` sub-split on paragraph boundaries

    Args:
        structured_text: Output of :func:`~backend.tools.pdf_parser.parse_pdf_with_structure`.

    Returns:
        List of :class:`DocumentSection` objects in document order.
    """
    lines = structured_text.splitlines()

    # Accumulator state
    current_heading = "Preamble"
    current_level = 1
    current_page = 1
    section_page_start = 1
    body_lines: list[str] = []
    current_tables: list[str] = []
    in_table = False
    table_lines: list[str] = []

    raw_sections: list[DocumentSection] = []

    def _flush() -> None:
        """Save the current section to raw_sections."""
        text = "\n".join(body_lines).strip()
        raw_sections.append(
            DocumentSection(
                heading=current_heading,
                level=current_level,
                text=text,
                tables=list(current_tables),
                page_start=section_page_start,
                page_end=current_page,
            )
        )

    for line in lines:
        # Track page boundaries
        page_match = _PAGE_RE.match(line)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Table accumulation
        if line.strip() == _TABLE_START:
            in_table = True
            table_lines = []
            continue
        if line.strip() == _TABLE_END:
            in_table = False
            current_tables.append("\n".join(table_lines))
            table_lines = []
            continue
        if in_table:
            table_lines.append(line)
            continue

        # Heading marker → start a new section
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            _flush()
            current_heading = heading_match.group(2).strip()
            current_level = int(heading_match.group(1))
            section_page_start = current_page
            body_lines = []
            current_tables = []
            continue

        # Ordinary body text
        body_lines.append(line)

    # Flush final section
    _flush()

    # --- Post-process: merge empty sections into the next one ---
    merged: list[DocumentSection] = []
    pending_empty: DocumentSection | None = None

    for sec in raw_sections:
        has_content = bool(sec.text.strip()) or bool(sec.tables)
        if not has_content:
            pending_empty = sec  # defer — will be merged forward
            continue
        if pending_empty is not None:
            # Prepend empty section's heading as context in the body
            sec = DocumentSection(
                heading=sec.heading,
                level=sec.level,
                text=sec.text,
                tables=pending_empty.tables + sec.tables,
                page_start=pending_empty.page_start,
                page_end=sec.page_end,
            )
            pending_empty = None
        merged.append(sec)

    # Flush trailing empty section (e.g. document ends with a bare heading)
    if pending_empty is not None:
        merged.append(pending_empty)

    # --- Post-process: sub-split sections that are too long ---
    final: list[DocumentSection] = []
    for sec in merged:
        if sec.char_count > SECTION_MAX_CHARS:
            final.extend(_sub_split_section(sec))
        else:
            final.append(sec)

    logger.debug("Segmented document into %d sections.", len(final))
    return final


# ---------------------------------------------------------------------------
# Obligation language filtering
# ---------------------------------------------------------------------------


def _section_has_obligation(section: DocumentSection) -> bool:
    """Return True if *section* contains at least one obligation pattern."""
    combined = section.text + "\n".join(section.tables)
    return any(pat.search(combined) for pat in OBLIGATION_PATTERNS)


def filter_obligation_sections(
    sections: list[DocumentSection],
) -> tuple[list[DocumentSection], list[DocumentSection]]:
    """
    Split sections into ``(obligation_sections, skipped_sections)``.

    A section passes if:
    - It contains at least one obligation pattern match, OR
    - It contains tables (requirements are often embedded in table rows).

    Cover pages, tables of contents, definition-only sections, and
    acknowledgement pages are typically skipped.

    Args:
        sections: All sections from :func:`segment_document`.

    Returns:
        A two-tuple ``(kept, skipped)``.
    """
    kept: list[DocumentSection] = []
    skipped: list[DocumentSection] = []

    for sec in sections:
        # Tables always pass — compliance requirements are frequently table-only
        if sec.tables or _section_has_obligation(sec):
            kept.append(sec)
        else:
            skipped.append(sec)

    return kept, skipped
