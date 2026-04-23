"""
Deterministic regex-based parser for structured DHCS Submission Review Forms.

These documents contain explicit numbered questions (e.g. "1. Does the P&P state...")
followed by an APL reference in the format "(Reference: APL XX-XXX, page(s) N)".
No LLM is needed for these forms.
"""

from __future__ import annotations

import re

from backend.models.schemas import Requirement

# ---------------------------------------------------------------------------
# Detection heuristic
# ---------------------------------------------------------------------------

# A structured review form contains numbered questions starting with
# "Does the P&P" and "Yes  No" checkbox markers.
_DETECTION_PATTERN = re.compile(
    r"^\d+\.\s+Does the P&P\b",
    re.MULTILINE | re.IGNORECASE,
)


def is_structured_form(text: str) -> bool:
    """Return True if the text looks like a structured DHCS review form."""
    return bool(_DETECTION_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Extraction regex
# ---------------------------------------------------------------------------

# Matches numbered questions of the form:
#   <num>. Does the P&P ... (Reference: APL XX-XXX, page(s) N[-M])
#
# Group 1: question number (str)
# Group 2: full question text including any sub-clauses (may contain newlines)
# Group 3: APL reference string
_QUESTION_PATTERN = re.compile(
    r"(\d+)\.\s+(Does the P&P\b.+?)"
    r"\(Reference:\s*(APL\s+[\d-]+,\s*(?:[^()]*|\([^)]*\))*)\)",
    re.DOTALL | re.IGNORECASE,
)

# Editorial notes that sometimes appear between the question text and the
# reference, e.g. "(P&P may include examples provided in APL)".
# We strip these so they don't pollute the requirement text.
_EDITORIAL_NOTE_PATTERN = re.compile(
    r"\([^)]*(?:P&P may|examples provided|list below)[^)]*\)\s*$",
    re.IGNORECASE,
)


def _clean_question_text(raw: str) -> str:
    """
    Normalise extracted question text:
    1. Strip trailing editorial parenthetical notes.
    2. Collapse all internal whitespace/newlines to single spaces.
    3. Strip leading/trailing whitespace.
    """
    # Remove editorial notes at the end of the question text
    cleaned = _EDITORIAL_NOTE_PATTERN.sub("", raw)
    # Normalise internal whitespace (newlines, multiple spaces, etc.)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_review_form(pdf_text: str) -> list[Requirement]:
    """
    Extract all numbered compliance questions from a structured DHCS review form.

    Uses a deterministic regex — no LLM call is made.  Questions that cannot
    be parsed into a valid :class:`~backend.models.schemas.Requirement` are
    silently skipped so that a partial but clean result is always returned.

    Args:
        pdf_text: The full plain-text content of the PDF (all pages joined).

    Returns:
        Ordered list of :class:`~backend.models.schemas.Requirement` objects,
        one per numbered question.  Returns an empty list when no questions
        are found (not a structured form, or text extraction failed).
    """
    matches = _QUESTION_PATTERN.findall(pdf_text)

    requirements: list[Requirement] = []
    for num_str, text_raw, reference_raw in matches:
        try:
            requirements.append(
                Requirement(
                    id=int(num_str),
                    text=_clean_question_text(text_raw),
                    reference=reference_raw.strip(),
                )
            )
        except (ValueError, TypeError):
            # Malformed question — skip gracefully
            continue

    return requirements
