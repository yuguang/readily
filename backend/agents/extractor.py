"""
Requirement Extraction Agent — Component 3.

Implements two-way routing based on document length:

- **Route 1 — Long narrative** (> ``LONG_DOC_PAGE_THRESHOLD`` pages): defers
  to the multi-step compliance extraction agent (Component 8) which segments
  the document and extracts enriched ``ComplianceRequirement`` objects.
  Returns ``("compliance", list[ComplianceRequirement])``.
- **Route 2 — Short narrative** (≤ ``LONG_DOC_PAGE_THRESHOLD`` pages): single-pass
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def classify_and_extract(pdf_path: str) -> tuple[str, list[Requirement]]:
    """
    Extract compliance requirements from a PDF using two-way routing.

    Algorithm
    ---------
    1. Parse the PDF with PyMuPDF; count pages and join text.
    2. **Route 1**: If page count > ``LONG_DOC_PAGE_THRESHOLD``, delegate to
       the compliance extraction agent (Component 8) for section-by-section
       processing and return enriched ``ComplianceRequirement`` objects.
    3. **Route 2**: Otherwise run the single-pass narrative extractor.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        ``(doc_type, requirements)`` where *doc_type* is one of
        ``"compliance"`` or ``"narrative"``.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        NotImplementedError: If Route 1 is taken and Component 8 is not built.
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

    # Route 1: long narrative → compliance extraction agent (Component 8)
    if page_count > LONG_DOC_PAGE_THRESHOLD:
        logger.info(
            "Route 1 — compliance: %s (%d pages > threshold %d).",
            pdf_path,
            page_count,
            LONG_DOC_PAGE_THRESHOLD,
        )
        requirements = await run_compliance_extractor(pdf_path)
        return "compliance", requirements
    # Route 2: short narrative → single-pass LLM extraction
    logger.info(
        "Route 2 — short narrative: %s (%d pages ≤ threshold %d).",
        pdf_path,
        page_count,
        LONG_DOC_PAGE_THRESHOLD,
    )
    requirements = run_narrative_extractor(full_text)
    return "narrative", requirements
