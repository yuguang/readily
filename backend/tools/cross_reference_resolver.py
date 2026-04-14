"""
Cross-reference detection and resolution for regulatory PDFs.

Compliance documents are full of internal references:

  "see Section III.A"
  "as defined in Appendix B"
  "per 42 CFR § 438.210"
  "the requirements set forth in Section IV below"

When each section is extracted in isolation the LLM misses the conditional
logic that ties these references together.  This module:

1. **Builds a :class:`SectionIndex`** from the already-segmented
   ``list[DocumentSection]`` in a first pass — no extra PDF read needed.

2. **Detects cross-references** in a section's body text with a set of
   regex patterns covering the most common forms found in DHCS/CMS regulatory
   documents.

3. **Resolves** each detected reference to the actual section text (truncated
   to a configurable limit so it doesn't overwhelm the context window).

4. **Formats** the resolved context as a clearly-labelled block that is
   appended to the extraction prompt, giving the LLM the relevant definitions
   and requirements it needs to interpret conditional obligations.

Public API
----------
- :func:`build_section_index` — build index from ``list[DocumentSection]``.
- :class:`SectionIndex` — lookup table; call :meth:`~SectionIndex.resolve`.
- :func:`detect_cross_references` — find refs in a text string.
- :func:`resolve_references` — detect + resolve in one call.
- :func:`format_resolved_context` — format for LLM injection.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.tools.document_segmenter import DocumentSection

logger = logging.getLogger(__name__)

# Maximum characters of referenced section text to inject per reference
_MAX_REF_CHARS: int = 450

# Maximum number of unique references to resolve per section (avoids prompt bloat)
_MAX_REFS_PER_SECTION: int = 6


# ---------------------------------------------------------------------------
# Cross-reference patterns
# ---------------------------------------------------------------------------

# Each pattern is a compiled regex; group 1 captures the target identifier
# (e.g. "III.A", "Appendix B", "42 CFR § 438.210").

_SECTION_IDENTIFIERS = (
    r"[A-Z0-9]"                     # starts with capital letter or digit
    r"[A-Z0-9]*"                    # more letters/digits
    r"(?:[.\-][A-Z0-9a-z]+)*"       # optional ".X" or "-X" sub-parts
    r"\.?"                          # optional trailing dot
)

# "see Section X" / "see Sections X" / "see Appendix X" / "see Annex X"
_P_SEE = re.compile(
    r"\bsee\s+"
    r"(?:Sections?|Appendix|Appendices|Annex(?:es)?|Article|Exhibit)\s+"
    r"(" + _SECTION_IDENTIFIERS + r")",
    re.IGNORECASE,
)

# "as defined/described/set forth/noted/required in Section X / Appendix X"
_P_AS_IN = re.compile(
    r"\bas\s+(?:defined|described|noted|required|specified|set\s+forth)\s+in\s+"
    r"(?:Section|Sections?|Appendix|Annex|Article|Exhibit)\s+"
    r"(" + _SECTION_IDENTIFIERS + r")",
    re.IGNORECASE,
)

# "per / pursuant to / under / in accordance with Section X / Appendix X"
_P_PER = re.compile(
    r"\b(?:per|pursuant\s+to|under|in\s+accordance\s+with)\s+"
    r"(?:Section|Appendix|Annex|Article)\s+"
    r"(" + _SECTION_IDENTIFIERS + r")",
    re.IGNORECASE,
)

# "the requirements of / requirements in / requirements set forth in Section X"
_P_REQUIREMENTS_OF = re.compile(
    r"\brequirements?\s+(?:of|in|set\s+forth\s+in)\s+"
    r"(?:Section|Appendix|Annex)?\s*"
    r"(" + _SECTION_IDENTIFIERS + r")",
    re.IGNORECASE,
)

# "42 CFR § 438.xxx" or "42 CFR §438.xxx" or "42 C.F.R. § 438.xxx"
_P_CFR = re.compile(
    r"\b(\d+\s+C\.?F\.?R\.?\s*§\s*[\d.]+(?:\([a-z0-9]+\))*)",
    re.IGNORECASE,
)

# "Section X" / "Appendix X" / "Annex X" when immediately after a comma,
# opening parenthesis, or at start of clause — catches standalone refs
_P_STANDALONE = re.compile(
    r"(?:^|[\s,(;])(?:Section|Appendix|Annex|Exhibit)\s+"
    r"(" + _SECTION_IDENTIFIERS + r")"
    r"(?=[\s,.);\n]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# All patterns in priority order (specific → general)
_ALL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("section", _P_SEE),
    ("section", _P_AS_IN),
    ("section", _P_PER),
    ("section", _P_REQUIREMENTS_OF),
    ("cfr",     _P_CFR),
    ("section", _P_STANDALONE),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossReference:
    """A single detected cross-reference within a section's body text."""

    raw_text: str    # The matched text as it appears in the document
    target_key: str  # Normalised lookup key (e.g. "III.A", "42 CFR § 438.210")
    ref_type: str    # "section" | "cfr" | "appendix"


@dataclass(frozen=True)
class ResolvedReference:
    """A cross-reference successfully resolved to section content."""

    cross_ref: CrossReference
    heading: str    # The matched section heading
    summary: str    # Truncated body text of the referenced section


# ---------------------------------------------------------------------------
# Section index
# ---------------------------------------------------------------------------


def _extract_section_number(heading: str) -> str | None:
    """
    Try to extract a bare section identifier from a heading string.

    Examples::

        "III.A Credentialing" → "III.A"
        "Section 4.2 Provider Responsibilities" → "4.2"
        "Appendix B – Definitions" → "B"
        "42 CFR § 438.210" → None (handled separately)
    """
    heading = heading.strip()

    # "Appendix X" / "Annex X" / "Exhibit X"
    m = re.match(r"^(?:Appendix|Annex|Exhibit)\s+([A-Z0-9]+)", heading, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # "Section X.Y" prefix
    m = re.match(r"^Section\s+(" + _SECTION_IDENTIFIERS + r")", heading, re.IGNORECASE)
    if m:
        return m.group(1).rstrip(".")

    # Roman numeral or alpha-numeric prefix like "III.A Credentialing"
    m = re.match(r"^([IVXLCDM]+(?:\.[A-Z0-9]+)*\.?)\s+\S", heading)
    if m:
        return m.group(1).rstrip(".")

    # Pure numeric prefix like "4.2 Provider Responsibilities"
    m = re.match(r"^(\d+(?:\.\d+)+)\.?\s+\S", heading)
    if m:
        return m.group(1)

    # Single capital letter heading like "A. Definitions"
    m = re.match(r"^([A-Z])\.\s+\S", heading)
    if m:
        return m.group(1)

    return None


def _normalize_key(key: str) -> str:
    """Lower-case, strip trailing punctuation and excess whitespace."""
    return re.sub(r"\s+", " ", key.strip().lower().rstrip(".,;"))


class SectionIndex:
    """
    In-memory lookup table mapping section identifiers to their content.

    Built from ``list[DocumentSection]`` by :func:`build_section_index`.

    Multiple key variants are registered per section so that references in
    varied forms all resolve to the same canonical entry:

    * Full heading (lowercased): ``"iii.a initial credentialing"``
    * Section number extracted from heading: ``"III.A"`` → key ``"iii.a"``
    * Appendix letter: ``"appendix b"`` → key ``"b"``
    """

    def __init__(self) -> None:
        # key (normalised) → (canonical_heading, body_text)
        self._index: dict[str, tuple[str, str]] = {}

    # ── Construction ──────────────────────────────────────────────────────

    def register(self, heading: str, text: str, tables: list[str]) -> None:
        """Register one section under all its key variants."""
        # Combine body text and table content for the summary
        body = text.strip()
        if tables:
            body = body + "\n" + "\n".join(tables)
        body = body.strip()

        def _add(key: str) -> None:
            nk = _normalize_key(key)
            if nk and nk not in self._index:
                self._index[nk] = (heading, body)

        # 1. Full heading
        _add(heading)

        # 2. Section number extracted from heading
        sec_num = _extract_section_number(heading)
        if sec_num:
            _add(sec_num)
            # Also register with "appendix"/"section" prefix stripped
            _add(re.sub(r"^(?:appendix|annex|exhibit)\s*", "", sec_num, flags=re.IGNORECASE))

        # 3. Heading without leading section number
        # "III.A Credentialing" → "Credentialing"
        no_num = re.sub(
            r"^[IVXLCDM0-9][A-Z0-9.\-]*\.?\s+",
            "",
            heading,
            flags=re.IGNORECASE,
        ).strip()
        if no_num and no_num != heading:
            _add(no_num)

    # ── Lookup ─────────────────────────────────────────────────────────────

    def resolve(
        self, target_key: str, max_chars: int = _MAX_REF_CHARS
    ) -> tuple[str, str] | None:
        """
        Look up *target_key* and return ``(heading, truncated_summary)``.

        Tries (in order):
        1. Exact normalised key match.
        2. Prefix match: ``target_key`` is a prefix of a stored key.
        3. Suffix match: a stored key is a prefix of ``target_key`` (handles
           partial references like "III" matching "III.A").

        Returns ``None`` if no match is found.
        """
        nk = _normalize_key(target_key)
        if not nk:
            return None

        # 1. Exact match
        if nk in self._index:
            heading, text = self._index[nk]
            return heading, _truncate(text, max_chars)

        # 2. Prefix match: stored key starts with the query
        for stored_key, (heading, text) in self._index.items():
            if stored_key.startswith(nk) and len(nk) >= 2:
                return heading, _truncate(text, max_chars)

        # 3. Suffix match: query starts with a stored key (e.g. "appendix b" in "b definitions")
        for stored_key, (heading, text) in self._index.items():
            if nk.startswith(stored_key) and len(stored_key) >= 2:
                return heading, _truncate(text, max_chars)

        return None

    def __len__(self) -> int:
        return len(self._index)

    def __bool__(self) -> bool:
        return bool(self._index)


def _truncate(text: str, max_chars: int) -> str:
    """Return text truncated to *max_chars*, appending '…' if trimmed."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # Try to cut at a sentence boundary
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    if last_period > max_chars // 2:
        cut = cut[: last_period + 1]
    return cut.strip() + " …"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_section_index(sections: list[DocumentSection]) -> SectionIndex:
    """
    Build a :class:`SectionIndex` from a list of document sections.

    Args:
        sections: Output of :func:`~backend.tools.document_segmenter.segment_document`.

    Returns:
        Populated :class:`SectionIndex` ready for cross-reference resolution.
    """
    index = SectionIndex()
    for s in sections:
        index.register(s.heading, s.text, s.tables)
    logger.debug("SectionIndex built with %d key variants from %d sections.", len(index), len(sections))
    return index


def detect_cross_references(text: str) -> list[CrossReference]:
    """
    Detect all cross-references in *text* using the built-in regex patterns.

    Returns a deduplicated list in order of first appearance.  CFR references
    are included even though they reference external regulations — the caller
    can decide whether to attempt resolution.

    Args:
        text: Body text of a document section.

    Returns:
        List of :class:`CrossReference` objects (may be empty).
    """
    seen_keys: set[str] = set()
    results: list[CrossReference] = []

    for ref_type, pattern in _ALL_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0).strip()
            target = m.group(1).strip().rstrip(".")

            # Normalise target key for deduplication
            norm_target = _normalize_key(target)
            if norm_target in seen_keys:
                continue
            seen_keys.add(norm_target)

            # Classify
            classified = ref_type
            if re.match(r"^[A-Z]$", target, re.IGNORECASE):
                classified = "appendix"
            elif ref_type == "cfr":
                classified = "cfr"

            results.append(
                CrossReference(raw_text=raw, target_key=target, ref_type=classified)
            )

    return results


def resolve_references(
    text: str,
    section_index: SectionIndex,
    max_chars_per_ref: int = _MAX_REF_CHARS,
    max_refs: int = _MAX_REFS_PER_SECTION,
    current_heading: str = "",
) -> list[ResolvedReference]:
    """
    Detect and resolve cross-references in *text* against *section_index*.

    Skips:
    - CFR references (external, not in the section index)
    - Self-references (target resolves to *current_heading*)

    Args:
        text:             Body text to scan for cross-references.
        section_index:    Index built by :func:`build_section_index`.
        max_chars_per_ref: Maximum body text chars to include per resolved ref.
        max_refs:          Maximum number of resolved refs to return.
        current_heading:  Heading of the section being processed (to skip
                          self-references).

    Returns:
        List of :class:`ResolvedReference` objects for successfully-resolved
        references, capped at *max_refs*.
    """
    if not section_index or not text:
        return []

    refs = detect_cross_references(text)
    resolved: list[ResolvedReference] = []

    for ref in refs:
        if len(resolved) >= max_refs:
            break

        # Skip CFR — they're external
        if ref.ref_type == "cfr":
            continue

        result = section_index.resolve(ref.target_key, max_chars=max_chars_per_ref)
        if result is None:
            continue

        heading, summary = result

        # Skip self-references
        if (
            current_heading
            and _normalize_key(heading) == _normalize_key(current_heading)
        ):
            continue

        if not summary:
            continue

        resolved.append(
            ResolvedReference(
                cross_ref=ref,
                heading=heading,
                summary=summary,
            )
        )

    return resolved


def format_resolved_context(resolved: list[ResolvedReference]) -> str:
    """
    Format resolved cross-references for injection into an LLM prompt.

    The output is clearly labelled so the model understands this is
    supplementary context, not the primary section text.

    Args:
        resolved: Output of :func:`resolve_references`.

    Returns:
        A multi-line string to append to the section text, or ``""`` if empty.
    """
    if not resolved:
        return ""

    lines: list[str] = [
        "",
        "--- REFERENCED SECTIONS (auto-resolved for context) ---",
    ]
    for r in resolved:
        lines.append(f"\n[Reference: \"{r.cross_ref.raw_text}\"]")
        lines.append(f"Section: {r.heading}")
        lines.append(r.summary)

    return "\n".join(lines)
