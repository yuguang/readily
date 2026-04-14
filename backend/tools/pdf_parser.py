"""PDF text extraction using PyMuPDF (fitz)."""

import logging
from typing import List, Dict

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for structured parsing
# ---------------------------------------------------------------------------


def _collect_font_sizes(doc: fitz.Document) -> dict[float, int]:
    """Return {rounded_font_size: total_char_count} across all text in *doc*."""
    counts: dict[float, int] = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:  # 0 = text block
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    counts[size] = counts.get(size, 0) + len(span["text"])
    return counts


def _build_heading_size_map(font_size_counts: dict[float, int]) -> dict[float, int]:
    """
    Return {font_size: heading_level} for the (up to 3) largest font sizes
    above the body font size.

    The body font size is whichever size has the most characters.
    """
    if not font_size_counts:
        return {}
    body_size = max(font_size_counts, key=lambda s: font_size_counts[s])
    heading_sizes = sorted(
        [s for s in font_size_counts if s > body_size + 0.5],
        reverse=True,
    )[:3]  # max 3 heading levels
    return {hs: level for level, hs in enumerate(heading_sizes, start=1)}


def _heading_level_for_span(size: float, heading_map: dict[float, int]) -> int | None:
    """Return the heading level for *size*, or None if it is body text."""
    for hs, level in heading_map.items():
        if abs(size - hs) <= 0.5:
            return level
    return None


def _extract_table_text(tab) -> str:
    """Render a PyMuPDF Table as a pipe-delimited string."""
    rows = tab.extract()  # list[list[str | None]]
    lines = []
    for row in rows:
        cells = [str(c).strip() if c is not None else "" for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rect_intersects(bbox_a, bbox_b) -> bool:
    """Return True if two bounding boxes (x0,y0,x1,y1) overlap."""
    r1 = fitz.Rect(bbox_a)
    r2 = fitz.Rect(bbox_b)
    return not r1.intersect(r2).is_empty


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


def parse_pdf_with_structure(pdf_path: str) -> str:
    """
    Parse a PDF preserving headings, tables, and page boundaries.

    Uses PyMuPDF block-level extraction to detect heading levels from font
    sizes and ``find_tables()`` to extract tables as pipe-delimited text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Structured text with ``[PAGE N]``, ``[HEADING N] text``, and
        ``[TABLE] ... [/TABLE]`` markers.  Returns an empty string on
        any fatal error.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path, exc)
        return ""

    try:
        # --- Pass 1: determine heading font sizes across the whole document ---
        font_size_counts = _collect_font_sizes(doc)
        heading_map = _build_heading_size_map(font_size_counts)

        parts: list[str] = []

        # --- Pass 2: build structured text page by page ---
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1
            parts.append(f"[PAGE {page_number}]")

            # Detect tables on this page
            tables: list = []
            table_texts: list[str] = []
            try:
                tab_finder = page.find_tables()
                for tab in tab_finder.tables:
                    tables.append(tab)
                    table_texts.append(_extract_table_text(tab))
            except Exception:
                pass  # find_tables unavailable or failed — continue without tables

            # Build a list of (y0, item_type, payload) sorted top-to-bottom
            page_items: list[tuple[float, str, object]] = []

            for tab, tab_text in zip(tables, table_texts):
                y0 = tab.bbox[1]
                page_items.append((y0, "table", tab_text))

            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:  # skip image blocks
                    continue
                # Skip text blocks that lie inside a table bounding box
                if any(_rect_intersects(block["bbox"], tab.bbox) for tab in tables):
                    continue

                # Determine dominant font size and full block text
                block_text_parts: list[str] = []
                max_size = 0.0
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            block_text_parts.append(span["text"])
                            max_size = max(max_size, span["size"])

                block_text = "".join(block_text_parts).strip()
                if not block_text:
                    continue

                level = _heading_level_for_span(max_size, heading_map)
                y0 = block["bbox"][1]
                page_items.append((y0, "block", (block_text, level)))

            # Emit items in top-to-bottom order
            page_items.sort(key=lambda x: x[0])

            for _, item_type, payload in page_items:
                if item_type == "table":
                    parts.append("[TABLE]")
                    parts.append(payload)  # type: ignore[arg-type]
                    parts.append("[/TABLE]")
                else:
                    block_text, level = payload  # type: ignore[misc]
                    if level is not None:
                        parts.append(f"[HEADING {level}] {block_text}")
                    else:
                        parts.append(block_text)

        return "\n".join(parts)
    finally:
        doc.close()
