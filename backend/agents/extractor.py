"""
Requirement Extraction Agent — Component 3.

Implements three-way routing based on document type and length:

- **Route 1 — Structured** (e.g. DHCS Submission Review Form): regex-based,
  no LLM needed.  Triggered when ≥3 numbered questions are detected.
  Returns ``("structured", list[Requirement])``.

- **Route 2 — Long narrative** (> ``LONG_DOC_PAGE_THRESHOLD`` pages): defers
  to the multi-step compliance extraction agent (Component 8) which segments
  the document and extracts enriched ``ComplianceRequirement`` objects.
  Returns ``("compliance", list[ComplianceRequirement])``.

- **Route 3 — Short narrative** (≤ ``LONG_DOC_PAGE_THRESHOLD`` pages): single-pass
  LLM extraction via a smolagents ``ToolCallingAgent``.
  Returns ``("narrative", list[Requirement])``.

Entry point: :func:`classify_and_extract` (async).
"""

from __future__ import annotations

import logging

from backend.agents.compliance_extractor import run_compliance_extractor
from backend.config import LONG_DOC_PAGE_THRESHOLD
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


async def classify_and_extract(pdf_path: str) -> tuple[str, list[Requirement]]:
    """
    Extract compliance requirements from a PDF using three-way routing.

    Algorithm
    ---------
    1. Parse the PDF with PyMuPDF; count pages and join text.
    2. **Route 1**: If ≥ ``_STRUCTURED_THRESHOLD`` numbered questions are found,
       return structured requirements immediately (no LLM call).
    3. **Route 2**: If page count > ``LONG_DOC_PAGE_THRESHOLD``, delegate to
       the compliance extraction agent (Component 8) for section-by-section
       processing and return enriched ``ComplianceRequirement`` objects.
    4. **Route 3**: Otherwise run the single-pass narrative extractor.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        ``(doc_type, requirements)`` where *doc_type* is one of
        ``"structured"``, ``"compliance"``, or ``"narrative"``.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        NotImplementedError: If Route 2 is taken and Component 8 is not built.
        ValueError: If the narrative extractor cannot parse the LLM output.
    """
    pages = parse_pdf(pdf_path)
    page_count = len(pages)

    if not pages:
        logger.warning(
            "parse_pdf returned no pages for %s — no text extractable.", pdf_path
        )
        return "narrative", []

    full_text = "\n".join(p["text"] for p in pages)

    # Route 1: structured extraction (regex, no LLM)
    requirements = parse_review_form(full_text)
    if len(requirements) >= _STRUCTURED_THRESHOLD:
        logger.info(
            "Route 1 — structured: %s (%d questions found).",
            pdf_path,
            len(requirements),
        )
        return "structured", requirements

    # Route 2: long narrative → compliance extraction agent (Component 8)
    if page_count > LONG_DOC_PAGE_THRESHOLD:
        logger.info(
            "Route 2 — compliance: %s (%d pages > threshold %d).",
            pdf_path,
            page_count,
            LONG_DOC_PAGE_THRESHOLD,
        )
        requirements = await run_compliance_extractor(pdf_path)
        return "compliance", requirements

    # Route 3: short narrative → single-pass LLM extraction
    logger.info(
        "Route 3 — short narrative: %s (%d pages ≤ threshold %d).",
        pdf_path,
        page_count,
        LONG_DOC_PAGE_THRESHOLD,
    )
    requirements = run_narrative_extractor(full_text)
    return "narrative", requirements
