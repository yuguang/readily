"""
Tests for header_footer_stripper.py.

Focuses on pure detection logic with synthetic block/page inputs so tests stay
fast and deterministic.
"""

from __future__ import annotations

import fitz

from backend.tools.header_footer_stripper import (
    HeaderFooterFilter,
    _normalize_signature,
    build_header_footer_filter,
)


def _make_pdf_with_repeating_header_footer() -> fitz.Document:
    """Create a small synthetic PDF in memory with repeating header/footer."""
    doc = fitz.open()
    for i in range(1, 5):  # 4 pages
        page = doc.new_page()
        # Header
        page.insert_text((72, 30), "DHCS APL 24-001", fontsize=10)
        # Body
        page.insert_text((72, 120), f"Section body text page {i}.", fontsize=12)
        # Footer / page number
        page.insert_text((72, 780), f"Page {i}", fontsize=10)
    return doc


class TestNormalizeSignature:
    def test_digits_collapsed(self):
        assert _normalize_signature("Page 3 of 47") == "page # of #"

    def test_whitespace_collapsed(self):
        assert _normalize_signature("  DHCS   APL   24-001  ") == "dhcs apl #-#"

    def test_empty_result_for_digits_only(self):
        assert _normalize_signature("12345") == ""


class TestBuildHeaderFooterFilter:
    def test_repeated_signature_detected(self):
        doc = _make_pdf_with_repeating_header_footer()
        try:
            filt = build_header_footer_filter(doc)
        finally:
            doc.close()
        assert isinstance(filt, HeaderFooterFilter)
        assert "dhcs apl #-#" in filt.repeated_signatures

    def test_header_y_bucket_detected(self):
        doc = _make_pdf_with_repeating_header_footer()
        try:
            filt = build_header_footer_filter(doc)
        finally:
            doc.close()
        assert filt.header_y_buckets, "Expected at least one header y-bucket"

    def test_footer_y_bucket_detected(self):
        doc = _make_pdf_with_repeating_header_footer()
        try:
            filt = build_header_footer_filter(doc)
        finally:
            doc.close()
        assert filt.footer_y_buckets, "Expected at least one footer y-bucket"

    def test_single_page_doc_returns_empty_filter(self):
        doc = fitz.open()
        doc.new_page()
        try:
            filt = build_header_footer_filter(doc)
        finally:
            doc.close()
        assert not filt


class TestIsHeaderFooter:
    def test_header_block_flagged(self):
        doc = _make_pdf_with_repeating_header_footer()
        try:
            filt = build_header_footer_filter(doc)
            page = doc[0]
            blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
            header_block = min(blocks, key=lambda b: b["bbox"][1])  # top-most
            assert filt.is_header_footer(header_block, page.rect.height) is True
        finally:
            doc.close()

    def test_footer_block_flagged(self):
        doc = _make_pdf_with_repeating_header_footer()
        try:
            filt = build_header_footer_filter(doc)
            page = doc[0]
            blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
            footer_block = max(blocks, key=lambda b: b["bbox"][1])  # bottom-most
            assert filt.is_header_footer(footer_block, page.rect.height) is True
        finally:
            doc.close()

    def test_body_block_not_flagged(self):
        doc = _make_pdf_with_repeating_header_footer()
        try:
            filt = build_header_footer_filter(doc)
            page = doc[0]
            blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
            body_block = sorted(blocks, key=lambda b: b["bbox"][1])[1]  # middle block
            assert filt.is_header_footer(body_block, page.rect.height) is False
        finally:
            doc.close()
