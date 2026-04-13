"""
Tests for the ingestion pipeline.

Run from the project root:
    pytest backend/tests/test_ingestion.py -v
"""

import shutil
from pathlib import Path

import pytest

from backend.ingestion.chunker import chunk_document
from backend.tools.pdf_parser import parse_pdf

# ---------------------------------------------------------------------------
# Paths to real data files used in tests
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
_POLICIES_DIR = _REPO_ROOT / "data" / "Public Policies"
_GA_DIR = _POLICIES_DIR / "GA"
# Pick one small PDF that we know exists
_SAMPLE_PDF = _GA_DIR / "GA.4010_CEO20241220_v20241201.pdf"


# ===========================================================================
# parse_pdf — unit tests
# ===========================================================================

class TestParsePdf:
    def test_known_pdf_returns_pages(self):
        pages = parse_pdf(str(_SAMPLE_PDF))
        assert len(pages) >= 1, "Expected at least one page with text"

    def test_page_structure(self):
        pages = parse_pdf(str(_SAMPLE_PDF))
        for page in pages:
            assert "page_number" in page
            assert "text" in page
            assert isinstance(page["page_number"], int)
            assert isinstance(page["text"], str)
            assert page["text"].strip(), "text should be non-empty after stripping"

    def test_page_numbers_are_1_indexed(self):
        pages = parse_pdf(str(_SAMPLE_PDF))
        if pages:
            assert pages[0]["page_number"] == 1

    def test_page_numbers_are_sequential(self):
        pages = parse_pdf(str(_SAMPLE_PDF))
        nums = [p["page_number"] for p in pages]
        # Page numbers must be increasing (some may be skipped if image-only)
        assert nums == sorted(nums)
        assert len(nums) == len(set(nums)), "Page numbers must be unique"

    def test_nonexistent_pdf_returns_empty(self):
        pages = parse_pdf("/nonexistent/path/to/missing.pdf")
        assert pages == []

    def test_non_pdf_file_returns_empty(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_text("this is not a pdf")
        pages = parse_pdf(str(bad))
        assert pages == []


# ===========================================================================
# chunk_document — unit tests
# ===========================================================================

class TestChunkDocument:
    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_pages(text: str, page_number: int = 1):
        return [{"page_number": page_number, "text": text}]

    # ── basic structure ───────────────────────────────────────────────────────

    def test_returns_list(self):
        pages = self._make_pages("Hello world.")
        result = chunk_document(pages, source_file="X/doc.pdf")
        assert isinstance(result, list)

    def test_required_keys_present(self):
        pages = self._make_pages("Hello world.")
        chunks = chunk_document(pages, source_file="GG/doc.pdf")
        assert chunks
        for c in chunks:
            for key in ("text", "source_file", "folder", "page_number", "chunk_index"):
                assert key in c, f"Missing key: {key}"

    def test_metadata_values(self):
        pages = [{"page_number": 3, "text": "Policy paragraph."}]
        chunks = chunk_document(pages, source_file="GG/GG.1508.pdf")
        assert len(chunks) == 1
        assert chunks[0]["source_file"] == "GG/GG.1508.pdf"
        assert chunks[0]["folder"] == "GG"
        assert chunks[0]["page_number"] == 3
        assert chunks[0]["chunk_index"] == 0

    def test_folder_derived_from_source_file(self):
        pages = self._make_pages("text")
        chunks = chunk_document(pages, source_file="MA/MA.1001.pdf")
        assert chunks[0]["folder"] == "MA"

    def test_no_source_file_gives_empty_folder(self):
        pages = self._make_pages("text")
        chunks = chunk_document(pages)
        assert chunks[0]["folder"] == ""

    # ── empty / whitespace input ──────────────────────────────────────────────

    def test_empty_pages_list(self):
        assert chunk_document([]) == []

    def test_whitespace_only_text(self):
        pages = self._make_pages("   \n\n   ")
        assert chunk_document(pages) == []

    # ── chunk sizing ──────────────────────────────────────────────────────────

    def test_two_paras_fit_in_one_chunk(self):
        # 999 + '\n\n' + 999 = 2000 chars — exactly fits
        text = "A" * 999 + "\n\n" + "B" * 999
        pages = self._make_pages(text)
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=2000, overlap=0)
        assert len(chunks) == 1
        assert "A" * 5 in chunks[0]["text"]
        assert "B" * 5 in chunks[0]["text"]

    def test_overflow_creates_second_chunk(self):
        # 999 + 2 + 999 = 2000, then 999 more chars → 2nd chunk
        text = "A" * 999 + "\n\n" + "B" * 999 + "\n\n" + "C" * 999
        pages = self._make_pages(text)
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=2000, overlap=0)
        assert len(chunks) == 2

    # ── sequential chunk indices ──────────────────────────────────────────────

    def test_chunk_indices_are_sequential(self):
        paras = ["Para " + str(i) + " " + "x" * 200 for i in range(10)]
        pages = self._make_pages("\n\n".join(paras))
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=500, overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    # ── overlap ───────────────────────────────────────────────────────────────

    def test_overlap_prefix_appears_in_next_chunk(self):
        # chunk_size=150 forces a split after the first paragraph (100 chars)
        text = "A" * 100 + "\n\n" + "B" * 100
        pages = self._make_pages(text)
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=150, overlap=20)
        assert len(chunks) == 2
        tail = chunks[0]["text"][-20:]
        assert chunks[1]["text"].startswith(tail), (
            f"Expected chunk[1] to start with last 20 chars of chunk[0]\n"
            f"  tail  = {tail!r}\n"
            f"  start = {chunks[1]['text'][:30]!r}"
        )

    def test_no_overlap_when_zero(self):
        text = "A" * 100 + "\n\n" + "B" * 100
        pages = self._make_pages(text)
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=150, overlap=0)
        if len(chunks) == 2:
            # Second chunk should NOT start with content from first
            assert not chunks[1]["text"].startswith("A")

    # ── oversized paragraph (sentence splitting) ──────────────────────────────

    def test_oversized_paragraph_is_split(self):
        # Build a paragraph much larger than chunk_size
        sentences = ["This is sentence number %d." % i for i in range(50)]
        para = " ".join(sentences)
        pages = self._make_pages(para)
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=200, overlap=0)
        assert len(chunks) > 1, "Oversized paragraph should produce multiple chunks"

    def test_all_text_is_preserved_across_chunks(self):
        # Verify no text is silently dropped (modulo overlap)
        text = "\n\n".join(["Word%d." % i for i in range(100)])
        pages = self._make_pages(text)
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=200, overlap=0)
        combined = " ".join(c["text"] for c in chunks)
        for i in range(100):
            assert "Word%d." % i in combined

    # ── multi-page ────────────────────────────────────────────────────────────

    def test_page_number_reflects_first_page_of_chunk(self):
        pages = [
            {"page_number": 1, "text": "Page one paragraph."},
            {"page_number": 2, "text": "Page two paragraph."},
        ]
        chunks = chunk_document(pages, source_file="x/y.pdf", chunk_size=50, overlap=0)
        page_nums = {c["page_number"] for c in chunks}
        # Both pages should be represented
        assert 1 in page_nums
        assert 2 in page_nums


# ===========================================================================
# ingest_all_policies — integration tests
# ===========================================================================

@pytest.mark.integration
class TestIngestAllPolicies:
    """These tests spin up a real ChromaDB instance and process real PDFs."""

    def _ga_policies_dir(self, tmp_path: Path) -> Path:
        """Copy only the GA folder into a temp directory and return its parent."""
        temp_policies = tmp_path / "policies"
        temp_policies.mkdir()
        shutil.copytree(_GA_DIR, temp_policies / "GA")
        return temp_policies

    def test_ingest_ga_folder(self, tmp_path):
        import chromadb
        from backend.ingestion.ingest import ingest_all_policies

        policies_dir = self._ga_policies_dir(tmp_path)
        chroma_dir = str(tmp_path / "chroma_db")

        stats = ingest_all_policies(str(policies_dir), chroma_dir)

        assert stats["total_files"] > 0, "Expected at least one PDF to be processed"
        assert stats["total_chunks"] > 0, "Expected at least one chunk"

        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection("policy_documents")
        assert col.count() == stats["total_chunks"]

    def test_ingest_is_idempotent(self, tmp_path):
        import chromadb
        from backend.ingestion.ingest import ingest_all_policies

        policies_dir = self._ga_policies_dir(tmp_path)
        chroma_dir = str(tmp_path / "chroma_db")

        stats1 = ingest_all_policies(str(policies_dir), chroma_dir)
        stats2 = ingest_all_policies(str(policies_dir), chroma_dir)

        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection("policy_documents")

        assert col.count() == stats1["total_chunks"], "Re-run should not duplicate chunks"
        assert stats1["total_chunks"] == stats2["total_chunks"]

    def test_ingest_stats_structure(self, tmp_path):
        from backend.ingestion.ingest import ingest_all_policies

        policies_dir = self._ga_policies_dir(tmp_path)
        chroma_dir = str(tmp_path / "chroma_db")

        stats = ingest_all_policies(str(policies_dir), chroma_dir)

        assert "total_files" in stats
        assert "total_chunks" in stats
        assert "errors" in stats
        assert isinstance(stats["total_files"], int)
        assert isinstance(stats["total_chunks"], int)
        assert isinstance(stats["errors"], list)

    def test_chunk_metadata_stored_in_chroma(self, tmp_path):
        import chromadb
        from backend.ingestion.ingest import ingest_all_policies

        policies_dir = self._ga_policies_dir(tmp_path)
        chroma_dir = str(tmp_path / "chroma_db")

        ingest_all_policies(str(policies_dir), chroma_dir)

        client = chromadb.PersistentClient(path=chroma_dir)
        col = client.get_collection("policy_documents")
        results = col.get(limit=1, include=["metadatas"])

        meta = results["metadatas"][0]
        assert "source_file" in meta
        assert "folder" in meta
        assert "page_number" in meta
        assert "chunk_index" in meta
        assert meta["folder"] == "GA"
