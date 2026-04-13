"""PDF text extraction using PyMuPDF (fitz)."""

import logging
from typing import List, Dict

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of a PDF.

    Returns a list of { page_number: int, text: str } for each page that
    contains extractable text.  Pages with no text layer (scanned / image-only)
    are skipped with a warning so the caller can decide whether to OCR them.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts with keys ``page_number`` (1-indexed) and ``text``.
        Returns an empty list on any fatal error (e.g. file not found,
        corrupted PDF).
    """
    pages: List[Dict] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        return pages

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed

            try:
                text = page.get_text()
            except Exception as exc:
                logger.warning(
                    "Could not extract text from page %d of %s: %s",
                    page_number,
                    pdf_path,
                    exc,
                )
                continue

            if not text or not text.strip():
                logger.warning(
                    "Page %d of %s has no extractable text "
                    "(possibly scanned / image-only). Skipping.",
                    page_number,
                    pdf_path,
                )
                continue

            pages.append({"page_number": page_number, "text": text})
    finally:
        doc.close()

    return pages
