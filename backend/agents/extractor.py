"""
Requirement Extraction Agent — Component 3.

Classifies an uploaded PDF as either a structured DHCS review form or a
narrative regulatory document, then routes to the appropriate extractor:

- **Structured** (e.g. DHCS Submission Review Form): deterministic regex
  via :func:`~backend.tools.review_form_parser.parse_review_form`.
- **Narrative** (e.g. DHCS Policy Guide): LLM-based extraction via
  :func:`~backend.tools.narrative_extractor.run_narrative_extractor`.

Entry point: :func:`classify_and_extract`.
"""

from __future__ import annotations

import logging

from backend.models.schemas import Requirement
from backend.tools.narrative_extractor import run_narrative_extractor
from backend.tools.pdf_parser import parse_pdf
from backend.tools.review_form_parser import parse_review_form

logger = logging.getLogger(__name__)

# Minimum number of structured questions required to treat a document as a
# structured review form.  Below this threshold we fall back to narrative
# extraction (avoids false positives from documents with a few incidental
# "Does the P&P" mentions).
_STRUCTURED_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_and_extract(pdf_path: str) -> tuple[str, list[Requirement]]:
    """
    Extract compliance requirements from a PDF, routing on document type.

    Algorithm
    ---------
    1. Parse the PDF with PyMuPDF to get per-page text.
    2. Join all pages into a single string and attempt structured extraction.
    3. If at least ``_STRUCTURED_THRESHOLD`` numbered questions are found,
       treat the document as structured and return those requirements.
    4. Otherwise fall back to the LLM-based narrative extractor.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        A ``(doc_type, requirements)`` tuple where *doc_type* is either
        ``"structured"`` or ``"narrative"``.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        ValueError: If the narrative extractor cannot parse the LLM output.
    """
    pages = parse_pdf(pdf_path)

    if not pages:
        logger.warning(
            "parse_pdf returned no pages for %s — no text extractable.", pdf_path
        )
        return "narrative", []

    full_text = "\n".join(p["text"] for p in pages)

    # --- Try structured extraction first ---
    requirements = parse_review_form(full_text)
    if len(requirements) >= _STRUCTURED_THRESHOLD:
        logger.info(
            "Classified %s as structured (%d questions found).",
            pdf_path,
            len(requirements),
        )
        return "structured", requirements

    # --- Fall back to narrative LLM extraction ---
    logger.info(
        "Classified %s as narrative (only %d structured questions found, "
        "threshold=%d). Running LLM extractor.",
        pdf_path,
        len(requirements),
        _STRUCTURED_THRESHOLD,
    )
    requirements = run_narrative_extractor(full_text)
    return "narrative", requirements
