"""
Header/footer detection and stripping for regulatory PDFs.

Regulatory PDFs repeat running headers, footers, and page numbers on every
page.  These pollute the structured text and confuse the segmenter with
fragments like "DHCS APL 24-001" or "Page 3 of 47" appearing mid-section.

Detection uses two independent signals:

1. **Repeated text signature** — block text that appears (after normalization)
   on ≥ ``min_page_fraction`` of pages.  Digit sequences are replaced with
   ``#`` so "Page 1" and "Page 2" both normalize to "page #" and are treated
   as the same repeating pattern.

2. **High-frequency Y-zone** — blocks in the top ``header_zone_ratio`` or
   bottom ``footer_zone_ratio`` fraction of page height that appear on
   ≥ ``min_page_fraction`` of pages.  This strips pure page-number lines and
   other zone-consistent blocks that normalization might not fully coalesce.

Public API
----------
- :func:`build_header_footer_filter` — analyse an open ``fitz.Document``
  and return a :class:`HeaderFooterFilter`.
- :func:`build_hf_filter_from_pdf` — convenience wrapper: open *pdf_path*,
  build and return the filter, close the file.
- :class:`HeaderFooterFilter` — returned filter; call
  :meth:`~HeaderFooterFilter.is_header_footer` on each block while iterating
  pages.
"""

from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (overridable via kwargs to build_header_footer_filter)
# ---------------------------------------------------------------------------

_HEADER_ZONE_RATIO: float = 0.12   # Top 12 % of page height
_FOOTER_ZONE_RATIO: float = 0.12   # Bottom 12 % of page height
_MIN_PAGE_FRACTION: float = 0.55   # Must appear on ≥ 55 % of pages
_MIN_PAGES_ABSOLUTE: int  = 3      # Must appear on at least 3 pages

# Y-position bucket granularity (as fraction of page height)
_Y_BUCKET_SIZE: float = 0.025


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _get_block_text(block: dict) -> str:
    """Extract and join all span text from a text block dict."""
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            t = span.get("text", "")
            if t.strip():
                parts.append(t)
    return "".join(parts).strip()


def _normalize_signature(text: str) -> str:
    """
    Normalize block text into a comparison signature.

    - Lowercase
    - Replace every run of digits with ``#`` (so "Page 3" == "Page 17")
    - Collapse whitespace to single spaces
    - Strip leading/trailing whitespace

    Returns an empty string if the result is empty or purely ``#`` /
    whitespace (avoids false-positive matches on digit-only blocks like
    bare page numbers).
    """
    text = text.strip().lower()
    # Replace digit sequences with '#'
    text = re.sub(r"\d+", "#", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Reject degenerate signatures
    if not text or set(text) <= {"#", " ", ".", ",", "|", "-", "/", "\\"}:
        return ""
    return text


def _y_bucket(y_norm: float) -> float:
    """Round a normalised y coordinate to the nearest bucket boundary."""
    return round(round(y_norm / _Y_BUCKET_SIZE) * _Y_BUCKET_SIZE, 4)


# ---------------------------------------------------------------------------
# Filter data class
# ---------------------------------------------------------------------------


@dataclass
class HeaderFooterFilter:
    """
    Encapsulates the detected header/footer patterns for one document.

    Created by :func:`build_header_footer_filter`.  Thread-safe for
    read-only use: pass the same instance to all workers.
    """

    # Normalised text signatures that appear on many pages → strip these anywhere
    repeated_signatures: frozenset[str] = field(default_factory=frozenset)

    # Normalised Y-position buckets (top of block) confirmed as header zones
    # Blocks whose y0_bucket appears in this set are stripped
    header_y_buckets: frozenset[float] = field(default_factory=frozenset)

    # Same for footer zones
    footer_y_buckets: frozenset[float] = field(default_factory=frozenset)

    # ── Interrogation ──────────────────────────────────────────────────────

    def is_header_footer(self, block: dict, page_height: float) -> bool:
        """
        Return ``True`` if *block* should be stripped as a header or footer.

        Args:
            block:       A PyMuPDF text block dict (``block["type"] == 0``).
            page_height: Height of the current page in points (``page.rect.height``).

        Returns:
            ``True`` if the block is classified as a header/footer.
        """
        if page_height <= 0:
            return False

        # ── Signal 1: repeated text signature ─────────────────────────────
        text = _get_block_text(block)
        if text:
            sig = _normalize_signature(text)
            if sig and sig in self.repeated_signatures:
                return True

        # ── Signal 2: high-frequency Y-zone ───────────────────────────────
        y0_norm = block["bbox"][1] / page_height
        bucket = _y_bucket(y0_norm)
        if bucket in self.header_y_buckets or bucket in self.footer_y_buckets:
            return True

        return False

    def __bool__(self) -> bool:
        """``False`` when the filter detected nothing (e.g. single-page doc)."""
        return bool(
            self.repeated_signatures
            or self.header_y_buckets
            or self.footer_y_buckets
        )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def build_header_footer_filter(
    doc: fitz.Document,
    header_zone_ratio: float = _HEADER_ZONE_RATIO,
    footer_zone_ratio: float = _FOOTER_ZONE_RATIO,
    min_page_fraction: float = _MIN_PAGE_FRACTION,
    min_pages_absolute: int  = _MIN_PAGES_ABSOLUTE,
) -> HeaderFooterFilter:
    """
    Analyse *doc* and return a :class:`HeaderFooterFilter`.

    Two passes over the document:

    * **Pass A** — collect block text signatures and their page sets; collect
      zone-bucket → page sets for top/bottom zone blocks.
    * **Decision** — signatures and buckets that exceed the frequency
      threshold become filter rules.

    Args:
        doc:                Open :class:`fitz.Document` instance.
        header_zone_ratio:  Fraction of page height considered the header zone.
        footer_zone_ratio:  Fraction of page height considered the footer zone.
        min_page_fraction:  Fraction of pages a signature/bucket must appear on.
        min_pages_absolute: Absolute minimum page count regardless of fraction.

    Returns:
        A :class:`HeaderFooterFilter` ready for use in parsing.
    """
    n_pages = len(doc)
    if n_pages < 2:
        logger.debug("Document has <2 pages; skipping header/footer detection.")
        return HeaderFooterFilter()

    min_pages = max(min_pages_absolute, math.ceil(n_pages * min_page_fraction))

    # sig -> set of page indices where it was seen
    sig_page_sets: dict[str, set[int]] = {}

    # (zone, y_bucket) -> set of page indices  (zone ∈ {"header", "footer"})
    zone_bucket_pages: dict[tuple[str, float], set[int]] = {}

    for page_idx in range(n_pages):
        page = doc[page_idx]
        page_height = page.rect.height
        if page_height <= 0:
            continue

        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:  # skip non-text blocks
                continue

            text = _get_block_text(block)
            if not text:
                continue

            y0_norm = block["bbox"][1] / page_height
            y1_norm = block["bbox"][3] / page_height
            bucket  = _y_bucket(y0_norm)
            in_header_zone = y0_norm < header_zone_ratio
            in_footer_zone = y1_norm > (1.0 - footer_zone_ratio)
            in_zone = in_header_zone or in_footer_zone

            # ── Signal 1: text signature (zone-restricted) ─────────────────
            # Only count occurrences that are inside a header/footer zone so
            # repeated body text (e.g. a recurring section title) is not
            # falsely flagged.
            if in_zone:
                sig = _normalize_signature(text)
                if sig:
                    sig_page_sets.setdefault(sig, set()).add(page_idx)

            # ── Signal 2: zone + bucket ────────────────────────────────────
            if in_header_zone:
                zone_bucket_pages.setdefault(("header", bucket), set()).add(page_idx)
            elif in_footer_zone:
                zone_bucket_pages.setdefault(("footer", bucket), set()).add(page_idx)

    # ── Build filter ───────────────────────────────────────────────────────
    repeated = frozenset(
        sig
        for sig, pages in sig_page_sets.items()
        if len(pages) >= min_pages
    )

    header_buckets = frozenset(
        bucket
        for (zone, bucket), pages in zone_bucket_pages.items()
        if zone == "header" and len(pages) >= min_pages
    )
    footer_buckets = frozenset(
        bucket
        for (zone, bucket), pages in zone_bucket_pages.items()
        if zone == "footer" and len(pages) >= min_pages
    )

    filt = HeaderFooterFilter(
        repeated_signatures=repeated,
        header_y_buckets=header_buckets,
        footer_y_buckets=footer_buckets,
    )
    logger.debug(
        "Header/footer detection: %d repeated signatures, %d header buckets, "
        "%d footer buckets (min_pages=%d / %d total).",
        len(repeated),
        len(header_buckets),
        len(footer_buckets),
        min_pages,
        n_pages,
    )
    return filt


def build_hf_filter_from_pdf(pdf_path: str) -> HeaderFooterFilter:
    """
    Convenience wrapper: open *pdf_path*, detect header/footer patterns,
    and return the resulting :class:`HeaderFooterFilter`.

    Returns an empty (pass-through) filter on any error so that a failed
    detection never breaks the extraction pipeline.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A :class:`HeaderFooterFilter` populated with the document's
        header/footer patterns, or an empty filter on failure.
    """
    try:
        doc = fitz.open(pdf_path)
        try:
            return build_header_footer_filter(doc)
        finally:
            doc.close()
    except Exception as exc:
        logger.warning(
            "Could not build header/footer filter for %s: %s", pdf_path, exc
        )
        return HeaderFooterFilter()
