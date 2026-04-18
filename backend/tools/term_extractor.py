"""
Term definition extraction for the Compliance Extraction pipeline (Component 8).

Scans structured PDF text for program-specific terms, acronym definitions,
and glossary entries, then persists them to a dedicated ChromaDB collection
(``document_terms``) for cheap lookup by downstream agents (e.g. the
``define_term`` tool in the Question Agent, Component 4).

Three sources are consulted in priority order — later sources do **not**
overwrite earlier ones for the same ``(term, abbreviation)`` key:

1. Glossary / Definitions / Abbreviations sections
2. Inline first-use acronyms (e.g. ``Enhanced Care Management (ECM)``)
3. Definition tables within regular document sections

The extraction is purely rule-based (regex + table parsing) so it is cheap
enough to run on every ingested document without incurring LLM costs.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import TYPE_CHECKING

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from backend.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    TERM_ACRONYM_MAX_LEN,
    TERM_COLLECTION_NAME,
    TERM_DEFINITION_MIN_LEN,
)
from backend.models.schemas import TermDefinition

if TYPE_CHECKING:
    from backend.tools.document_segmenter import DocumentSection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Match headings that indicate a glossary / definitions / abbreviations section.
_GLOSSARY_HEADING_RE = re.compile(
    r"(?i)^(?:appendix\s+[a-z]:\s*)?"
    r"(definitions?|glossary|abbreviations?|acronyms?|key\s+terms?)\b"
)

# Match inline first-use acronym expansions: "Full Term Name (ACRONYM)".
_INLINE_ACRONYM_RE = re.compile(
    r"([A-Z][A-Za-z0-9&/\- ]{2,80}?)\s*\(([A-Z][A-Z0-9/&\-]{1,10})\)"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _initials_match(phrase: str, acronym: str) -> bool:
    """
    Return True when the initials of all words in *phrase* contain the
    characters of *acronym* as a subsequence (case-insensitive).

    All words — including stop-words like "of", "and", "the" — contribute
    their first letter so that "Population of Focus" correctly matches "POF".

    Examples::

        _initials_match("Enhanced Care Management", "ECM")       → True
        _initials_match("Population of Focus", "POF")            → True
        _initials_match("Managed Care Plan", "MCP")              → True
        _initials_match("Random Unrelated Words", "ECM")         → False
    """
    words = re.split(r"[\s/&\-]+", phrase.strip())
    initials = "".join(w[0].upper() for w in words if w)
    clean_acronym = re.sub(r"[^A-Z0-9]", "", acronym.upper())
    if not clean_acronym or not initials:
        return False
    # Every character in the acronym must appear in order within the initials.
    ia = 0
    for ch in clean_acronym:
        while ia < len(initials) and initials[ia] != ch:
            ia += 1
        if ia >= len(initials):
            return False
        ia += 1
    return True


def _parse_pipe_table(table_text: str) -> tuple[list[str], list[list[str]]]:
    """
    Parse a pipe-delimited table into ``(header_row, data_rows)``.

    Headers are returned lower-cased for keyword comparison.
    Returns ``([], [])`` if the table has no lines.
    """
    lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
    if not lines:
        return [], []
    headers = [c.strip().lower() for c in lines[0].split("|") if c.strip()]
    rows: list[list[str]] = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            rows.append(cells)
    return headers, rows


def _is_definition_table(headers: list[str]) -> bool:
    """Return True if *headers* suggest this table contains term definitions."""
    joined = " ".join(headers)
    return any(kw in joined for kw in ("definition", "criteria", "meaning"))


def _col_index(headers: list[str], keywords: list[str], fallback: int) -> int:
    """Return the first column index whose header contains one of *keywords*."""
    for i, h in enumerate(headers):
        if any(kw in h for kw in keywords):
            return i
    return fallback


def _term_metadata(t: TermDefinition) -> dict:
    """Return a ChromaDB-compatible metadata dict (no None values allowed)."""
    return {k: v for k, v in t.model_dump().items() if v is not None}


# ---------------------------------------------------------------------------
# Source 1: Glossary / Definitions sections
# ---------------------------------------------------------------------------


def _parse_glossary_section(
    section: DocumentSection,
    source_file: str,
) -> list[TermDefinition]:
    """
    Extract term definitions from a section identified as a glossary.

    Handles three formats commonly found in regulatory documents:

    - Two-column pipe-delimited tables (``Criteria | Definition``,
      ``Term | Definition``)
    - ``Term — definition`` or ``Term: definition`` line patterns
    - Bullet / dash lists: ``- Term: definition``
    """
    results: list[TermDefinition] = []
    seen_terms: set[str] = set()

    # 1a: Tables within the section
    for table_text in section.tables:
        headers, rows = _parse_pipe_table(table_text)
        if not headers or len(headers) < 2:
            continue
        term_col = _col_index(headers, ["term", "criteria", "name", "abbr"], 0)
        def_col = _col_index(headers, ["definition", "description", "meaning", "criteria"], 1)
        # Avoid pointing both columns at the same index
        if term_col == def_col:
            def_col = 1 if term_col == 0 else 0
        for row in rows:
            if len(row) <= max(term_col, def_col):
                continue
            term_text = row[term_col].strip()
            def_text = row[def_col].strip()
            if not term_text or len(def_text) < TERM_DEFINITION_MIN_LEN:
                continue
            key = term_text.lower()
            if key in seen_terms:
                continue
            seen_terms.add(key)
            results.append(
                TermDefinition(
                    term=term_text,
                    definition=def_text,
                    source_file=source_file,
                    page_number=section.page_start,
                    section_heading=section.heading,
                    source="glossary",
                )
            )

    # 1b: Line-based "Term: definition" or "Term — definition" patterns.
    # Captures lines like "Mental Illness — A condition characterized by..."
    # or "Enhanced Care Management (ECM): A whole-person approach..."
    line_re = re.compile(
        r"^[-\u2022*]?\s*"
        r"([A-Z][^\n:\u2014\-]{1,80}?)"   # term (starts with capital, ≤80 chars)
        r"\s*(?:[:—\u2014\-]\s*)"           # separator: colon, em-dash, or hyphen
        r"(.{10,}?)$",                       # definition body (≥10 chars)
        re.MULTILINE,
    )
    for match in line_re.finditer(section.text):
        term_text = match.group(1).strip().rstrip(":").strip()
        def_text = match.group(2).strip()
        if not term_text or len(def_text) < TERM_DEFINITION_MIN_LEN:
            continue
        # Skip lines that look like sentences rather than term definitions.
        if len(term_text.split()) > 8:
            continue
        key = term_text.lower()
        if key in seen_terms:
            continue
        seen_terms.add(key)
        results.append(
            TermDefinition(
                term=term_text,
                definition=def_text,
                source_file=source_file,
                page_number=section.page_start,
                section_heading=section.heading,
                source="glossary",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Source 2: Inline first-use acronyms
# ---------------------------------------------------------------------------


def _build_page_map(structured_text: str) -> list[tuple[int, int]]:
    """
    Return a sorted list of ``(char_position, page_number)`` built from
    ``[PAGE N]`` markers in *structured_text*.
    """
    positions: list[tuple[int, int]] = []
    for m in re.finditer(r"\[PAGE (\d+)\]", structured_text):
        positions.append((m.start(), int(m.group(1))))
    return positions


def _page_for_pos(pos: int, page_map: list[tuple[int, int]]) -> int:
    """Return the page number that contains character position *pos*."""
    page = 1
    for char_pos, page_num in page_map:
        if char_pos <= pos:
            page = page_num
        else:
            break
    return page


def _extract_inline_acronyms(
    structured_text: str,
    source_file: str,
) -> list[TermDefinition]:
    """
    Scan the full structured text for inline first-use acronym expansions.

    Only the **first** occurrence of each acronym is captured.
    Initials of the phrase must roughly match the acronym (stop-words excluded).
    """
    results: list[TermDefinition] = []
    seen_abbrs: set[str] = set()
    page_map = _build_page_map(structured_text)

    for match in _INLINE_ACRONYM_RE.finditer(structured_text):
        phrase = match.group(1).strip()
        acronym = match.group(2).strip()

        if len(acronym) > TERM_ACRONYM_MAX_LEN:
            continue
        if acronym.upper() in seen_abbrs:
            continue
        if not _initials_match(phrase, acronym):
            continue

        seen_abbrs.add(acronym.upper())
        page_num = _page_for_pos(match.start(), page_map)

        results.append(
            TermDefinition(
                term=phrase,
                abbreviation=acronym,
                # Minimal definition from inline context — may be enriched by
                # a glossary entry if one exists.
                definition=f"{phrase} ({acronym})",
                source_file=source_file,
                page_number=page_num,
                source="inline_acronym",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Source 3: Definition tables within regular sections
# ---------------------------------------------------------------------------


def _extract_definition_tables(
    sections: list[DocumentSection],
    source_file: str,
    seen_keys: set[str],
) -> list[TermDefinition]:
    """
    Extract term definitions from definition tables in non-glossary sections.

    A table qualifies if its header row contains ``definition``, ``criteria``,
    or ``meaning``.  Terms already present in *seen_keys* are skipped.
    """
    results: list[TermDefinition] = []

    for section in sections:
        # Glossary sections are handled by Source 1; skip here to avoid duplication.
        if _GLOSSARY_HEADING_RE.match(section.heading):
            continue
        for table_text in section.tables:
            headers, rows = _parse_pipe_table(table_text)
            if not headers or not _is_definition_table(headers):
                continue
            term_col = _col_index(headers, ["term", "criteria", "name"], 0)
            def_col = _col_index(headers, ["definition", "description", "meaning"], 1)
            if term_col == def_col:
                continue
            for row in rows:
                if len(row) <= max(term_col, def_col):
                    continue
                term_text = row[term_col].strip()
                def_text = row[def_col].strip()
                if not term_text or len(def_text) < TERM_DEFINITION_MIN_LEN:
                    continue
                key = term_text.lower()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                results.append(
                    TermDefinition(
                        term=term_text,
                        definition=def_text,
                        source_file=source_file,
                        page_number=section.page_start,
                        section_heading=section.heading,
                        source="definition_table",
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_term_definitions(
    structured_text: str,
    sections: list[DocumentSection],
    source_file: str,
) -> list[TermDefinition]:
    """
    Scan the structured PDF text and its segmented sections for term definitions.

    Looks in three places in priority order — later sources do **not** overwrite
    earlier ones for the same ``(term, abbreviation)`` key:

    1. Glossary / Definitions / Abbreviations sections
    2. Inline first-use acronyms (e.g. ``Enhanced Care Management (ECM)``)
    3. Definition tables within regular sections

    Args:
        structured_text: Output of ``parse_pdf_with_structure``.
        sections:        Segmented document sections from ``segment_document``.
        source_file:     Path to the source PDF (used for attribution).

    Returns:
        Deduplicated list of :class:`~backend.models.schemas.TermDefinition`
        objects.  Explicit glossary entries take precedence over inline acronyms
        for the same term.
    """
    # Source 1: Glossary sections (highest priority).
    glossary_terms: list[TermDefinition] = []
    seen_keys: set[str] = set()  # term.lower() keys already captured

    for section in sections:
        if _GLOSSARY_HEADING_RE.match(section.heading):
            for t in _parse_glossary_section(section, source_file):
                key = t.term.lower()
                if key not in seen_keys:
                    seen_keys.add(key)
                    glossary_terms.append(t)

    # Source 2: Inline acronyms — skip any already covered by glossary.
    glossary_abbrs = {t.abbreviation.upper() for t in glossary_terms if t.abbreviation}
    inline_terms: list[TermDefinition] = []
    seen_abbrs: set[str] = set(glossary_abbrs)

    for t in _extract_inline_acronyms(structured_text, source_file):
        abbr_key = (t.abbreviation or t.term).upper()
        term_key = t.term.lower()
        if abbr_key in seen_abbrs or term_key in seen_keys:
            continue
        seen_abbrs.add(abbr_key)
        seen_keys.add(term_key)
        inline_terms.append(t)

    # Source 3: Definition tables in regular sections.
    table_terms = _extract_definition_tables(sections, source_file, seen_keys)

    all_terms = glossary_terms + inline_terms + table_terms
    logger.debug(
        "Term extractor: %d glossary + %d inline + %d table = %d total for %s",
        len(glossary_terms),
        len(inline_terms),
        len(table_terms),
        len(all_terms),
        source_file,
    )
    return all_terms


def upsert_term_definitions(terms: list[TermDefinition]) -> None:
    """
    Upsert :class:`~backend.models.schemas.TermDefinition` objects into the
    ``document_terms`` ChromaDB collection.

    Idempotent: re-running on the same document overwrites existing entries
    because document IDs are derived from ``f"{source_file}::{abbreviation or term}"``.

    Args:
        terms: Term definitions to upsert.  No-op if the list is empty.
    """
    if not terms:
        return

    collection = _get_chroma_client().get_or_create_collection(
        name=TERM_COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        ),
    )
    ids = [f"{t.source_file}::{t.abbreviation or t.term}" for t in terms]
    documents = [
        f"{t.term} ({t.abbreviation}): {t.definition}"
        if t.abbreviation
        else f"{t.term}: {t.definition}"
        for t in terms
    ]
    metadatas = [_term_metadata(t) for t in terms]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.debug(
        "Upserted %d term definitions to '%s' collection.",
        len(terms),
        TERM_COLLECTION_NAME,
    )
