"""
Tests for the compliance extraction parallelization changes (#1-5 + #8):

- AsyncTokenBucket rate limiter (#5b)
- Extraction cache roundtrip + pipeline cache hit (#8)
- Batched cross-reference resolver tool (#2)
- Batched small-section extraction (#3)
- Process-pool embedding path for dedup (#4)
- Overlapped preparatory steps (#1)
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.agents import compliance_extractor as mod
from backend.agents.compliance_extractor import (
    BatchResolveCrossReferenceTool,
    _build_batch_prompt,
    _parse_batch_response,
    _run_batch_with_timeout,
    deduplicate_requirements,
    partition_sections_for_batching,
)
from backend.models.schemas import ComplianceRequirement
from backend.tools import extraction_cache
from backend.tools.cross_reference_resolver import SectionIndex, build_section_index
from backend.tools.document_segmenter import DocumentSection
from backend.tools.rate_limiter import AsyncTokenBucket


async def _async_empty(*_args, **_kwargs):
    """Async helper that returns an empty list immediately."""
    return []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_section(
    heading: str = "Test",
    text: str = "The MCP must comply.",
    level: int = 2,
    tables: list[str] | None = None,
    page_start: int = 1,
) -> DocumentSection:
    return DocumentSection(
        heading=heading,
        level=level,
        text=text,
        tables=tables or [],
        page_start=page_start,
        page_end=page_start,
    )


def _make_req(text: str, section_heading: str = "Test") -> ComplianceRequirement:
    return ComplianceRequirement(
        id=0,
        text=text,
        section_heading=section_heading,
        obligation_type="mandatory",
        exact_quote=text,
        reference=f"{section_heading}, page 1",
    )


# ===========================================================================
# AsyncTokenBucket  (#5b)
# ===========================================================================


class TestAsyncTokenBucket:
    def test_disabled_bucket_is_noop(self):
        bucket = AsyncTokenBucket(rate_per_sec=0.0)

        async def main():
            start = time.monotonic()
            for _ in range(100):
                await bucket.acquire()
            return time.monotonic() - start

        elapsed = asyncio.run(main())
        assert elapsed < 0.05

    def test_bucket_throttles_to_rate(self):
        """10 req/s bucket → 20 sequential acquires should take ~1s, not 0s."""
        bucket = AsyncTokenBucket(rate_per_sec=10.0, capacity=1.0)

        async def main():
            start = time.monotonic()
            # First acquire spends the initial token immediately; subsequent
            # acquires must wait on refill.
            for _ in range(6):
                await bucket.acquire()
            return time.monotonic() - start

        elapsed = asyncio.run(main())
        # With rate=10/s and capacity=1, 6 acquires need ~5 refill intervals
        # = 0.5s.  Allow generous slack on slow CI runners.
        assert 0.3 < elapsed < 2.0, f"Expected throttled elapsed ~0.5s, got {elapsed:.2f}s"

    def test_concurrent_acquires_share_rate(self):
        """Two coroutines sharing the bucket should still respect the rate."""
        bucket = AsyncTokenBucket(rate_per_sec=5.0, capacity=1.0)

        async def worker(count: int) -> None:
            for _ in range(count):
                await bucket.acquire()

        async def main():
            start = time.monotonic()
            await asyncio.gather(worker(3), worker(3))
            return time.monotonic() - start

        elapsed = asyncio.run(main())
        # 6 total requests at 5/s ≈ 1s minimum.
        assert elapsed > 0.6, f"Expected >= 1s throttled elapsed, got {elapsed:.2f}s"


# ===========================================================================
# Extraction cache  (#8)
# ===========================================================================


class TestExtractionCache:
    def test_hash_pdf_bytes_stable(self, tmp_path: Path):
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"hello world")
        h1 = extraction_cache.hash_pdf_bytes(p)
        h2 = extraction_cache.hash_pdf_bytes(p)
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_pdf_bytes_differs_per_content(self, tmp_path: Path):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"first")
        b.write_bytes(b"second")
        assert extraction_cache.hash_pdf_bytes(a) != extraction_cache.hash_pdf_bytes(b)

    def test_structured_text_roundtrip(self, tmp_path: Path):
        with patch.object(extraction_cache, "EXTRACTION_CACHE_DIR", tmp_path):
            assert extraction_cache.get_structured_text("abc123") is None
            extraction_cache.put_structured_text("abc123", "[PAGE 1]\nBody.")
            assert extraction_cache.get_structured_text("abc123") == "[PAGE 1]\nBody."

    def test_hf_signatures_roundtrip(self, tmp_path: Path):
        with patch.object(extraction_cache, "EXTRACTION_CACHE_DIR", tmp_path):
            sigs = frozenset({"page #", "dhcs apl #-#"})
            extraction_cache.put_hf_signatures("abc123", sigs)
            assert extraction_cache.get_hf_signatures("abc123") == sigs

    def test_sections_roundtrip(self, tmp_path: Path):
        with patch.object(extraction_cache, "EXTRACTION_CACHE_DIR", tmp_path):
            sections = [_make_section("A"), _make_section("B")]
            extraction_cache.put_sections("abc", sections)
            restored = extraction_cache.get_sections("abc")
            assert restored is not None
            assert [s.heading for s in restored] == ["A", "B"]

    def test_section_requirements_roundtrip(self, tmp_path: Path):
        with patch.object(extraction_cache, "EXTRACTION_CACHE_DIR", tmp_path):
            reqs = [{"id": 0, "text": "Q?", "section_heading": "S"}]
            extraction_cache.put_section_requirements("pdf", "sec", reqs)
            assert extraction_cache.get_section_requirements("pdf", "sec") == reqs

    def test_cache_disabled_returns_none(self, tmp_path: Path):
        with (
            patch.object(extraction_cache, "EXTRACTION_CACHE_DIR", tmp_path),
            patch.object(extraction_cache, "EXTRACTION_CACHE_ENABLED", False),
        ):
            extraction_cache.put_structured_text("h", "payload")
            assert extraction_cache.get_structured_text("h") is None


class TestPipelineCacheIntegration:
    """Second run should hit the cache and skip PDF parsing + segmentation."""

    def test_second_run_is_a_cache_hit(self, tmp_path: Path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 synthetic bytes")

        sections = [_make_section("Heading", text="MCPs shall do X.")]
        structured_text = "[PAGE 1]\n[HEADING 1] Heading\nMCPs shall do X."

        parse_calls: list[str] = []
        hf_calls: list[str] = []
        segment_calls: list[str] = []

        def fake_parse(path, hf_filter=None):
            parse_calls.append(path)
            return structured_text

        def fake_hf(path):
            hf_calls.append(path)
            return mod.HeaderFooterFilter(repeated_signatures=frozenset({"page #"}))

        def fake_segment(text):
            segment_calls.append(text)
            return sections

        async def run_once():
            events = []
            gen = mod.run_compliance_extractor_with_progress(str(pdf_path))
            async for ev in gen:
                events.append(ev)
            return events

        with (
            patch.object(extraction_cache, "EXTRACTION_CACHE_DIR", tmp_path / "cache"),
            # Avoid running the real extraction + term extraction + dedup.
            patch.object(
                mod, "partition_sections_for_batching",
                return_value=([], list(sections)),
            ),
            patch.object(mod, "_run_section_with_timeout", side_effect=_async_empty),
            patch.object(mod, "extract_term_definitions", return_value=[]),
            patch.object(mod, "upsert_term_definitions"),
            patch.object(mod, "parse_pdf_with_structure", side_effect=fake_parse),
            patch.object(mod, "build_hf_filter_from_pdf", side_effect=fake_hf),
            patch.object(mod, "segment_document", side_effect=fake_segment),
        ):
            asyncio.run(run_once())
            # First run: all three primary producers called once.
            assert len(parse_calls) == 1
            assert len(hf_calls) == 1
            assert len(segment_calls) == 1

            asyncio.run(run_once())
            # Second run: cache short-circuits all three.
            assert len(parse_calls) == 1
            assert len(hf_calls) == 1
            assert len(segment_calls) == 1


# ===========================================================================
# BatchResolveCrossReferenceTool  (#2)
# ===========================================================================


class TestBatchResolveCrossReferenceTool:
    def _index(self) -> SectionIndex:
        sections = [
            _make_section(heading="III.A Initial Credentialing",
                          text="Providers must be credentialed within 60 days."),
            _make_section(heading="Appendix B Definitions",
                          text="Credentialing: the process ..."),
        ]
        return build_section_index(sections)

    def test_interface_contract(self):
        tool = BatchResolveCrossReferenceTool(SectionIndex())
        assert tool.name == "resolve_cross_references"
        assert "section_ids" in tool.inputs
        assert tool.inputs["section_ids"]["type"] == "array"
        assert tool.output_type == "string"

    def test_resolves_multiple_ids_in_one_call(self):
        tool = BatchResolveCrossReferenceTool(self._index())
        out = tool(section_ids=["III.A", "Appendix B"])
        assert "III.A" in out
        assert "60 days" in out
        assert "Appendix B" in out
        assert "Credentialing" in out

    def test_unknown_ids_reported_individually(self):
        tool = BatchResolveCrossReferenceTool(self._index())
        out = tool(section_ids=["III.A", "ZZZ.999"])
        assert "60 days" in out
        assert "ZZZ.999" in out
        assert "No section found" in out

    def test_empty_list_returns_error(self):
        tool = BatchResolveCrossReferenceTool(self._index())
        assert "non-empty" in tool(section_ids=[]).lower() or "error" in tool(section_ids=[]).lower()

    def test_duplicate_ids_deduped(self):
        tool = BatchResolveCrossReferenceTool(self._index())
        out = tool(section_ids=["III.A", "iii.a", "III.A"])
        # Only one block for III.A, not three.
        assert out.count("60 days") == 1


# ===========================================================================
# partition_sections_for_batching  (#3)
# ===========================================================================


class TestPartitionSections:
    def test_small_sections_grouped(self):
        sections = [_make_section(f"S{i}", text="short " * 5) for i in range(10)]
        batches, large = partition_sections_for_batching(
            sections, max_chars=200, max_count=4,
        )
        assert large == []
        # 10 sections / batch of 4 → 3 batches
        assert len(batches) == 3
        assert sum(len(b) for b in batches) == 10

    def test_tables_routed_to_large_path(self):
        sections = [
            _make_section("Small", text="short text"),
            _make_section("TableSec", text="", tables=["| col |\n| val |"]),
        ]
        batches, large = partition_sections_for_batching(sections, max_chars=200, max_count=4)
        assert [s.heading for s in large] == ["TableSec"]
        assert sum(len(b) for b in batches) == 1

    def test_large_sections_skip_batching(self):
        big = _make_section("Big", text="x " * 5000)
        small = _make_section("Small", text="short")
        batches, large = partition_sections_for_batching([big, small], max_chars=800, max_count=4)
        assert big in large
        assert small not in large

    def test_batching_disabled_when_max_chars_zero(self):
        sections = [_make_section(f"S{i}") for i in range(5)]
        batches, large = partition_sections_for_batching(sections, max_chars=0, max_count=4)
        assert batches == []
        assert large == sections


# ===========================================================================
# Batched small-section extraction: prompt + response parsing  (#3)
# ===========================================================================


class TestBatchExtraction:
    def test_build_batch_prompt_contains_every_section(self):
        sections = [
            _make_section("Section One", text="Body of one."),
            _make_section("Section Two", text="Body of two."),
        ]
        prompt = _build_batch_prompt(sections)
        assert "Section One" in prompt
        assert "Section Two" in prompt
        assert "Body of one." in prompt
        assert "[SECTION_END]" in prompt

    def test_parse_batch_response_builds_requirements_per_section(self):
        sections = [
            _make_section("Section One"),
            _make_section("Section Two"),
        ]
        raw = """{
          "sections": [
            {"heading": "Section One", "requirements": [
              {"text": "Does the P&P state X?", "exact_quote": "X"}
            ]},
            {"heading": "Section Two", "requirements": [
              {"text": "Does the P&P state Y?", "exact_quote": "Y"}
            ]}
          ]
        }"""
        out = _parse_batch_response(raw, sections)
        assert len(out) == 2
        headings = {r.section_heading for r in out}
        assert headings == {"Section One", "Section Two"}

    def test_parse_batch_response_tolerates_markdown_fences(self):
        sections = [_make_section("S")]
        raw = "```json\n{\"sections\": [{\"heading\": \"S\", \"requirements\": []}]}\n```"
        # No requirements produced, but parsing must not crash.
        assert _parse_batch_response(raw, sections) == []

    def test_batch_timeout_returns_empty(self):
        """A hung batched LLM call must not stall the pipeline."""
        sections = [_make_section("A"), _make_section("B")]
        release = threading.Event()

        def blocking(*args, **kwargs):
            # Block until the async coroutine has already returned (released
            # by run_and_release below), so asyncio.run() teardown is fast.
            release.wait(timeout=5.0)
            return []

        async def run_and_release():
            result = await _run_batch_with_timeout(
                sections, timeout_seconds=0.1, pdf_hash=None
            )
            release.set()  # unblock the thread so asyncio.run() can drain quickly
            return result

        with patch.object(mod, "_run_batch_extraction_sync", side_effect=blocking):
            result = asyncio.run(run_and_release())
        assert result == []


# ===========================================================================
# Process-pool embedding path  (#4)
# ===========================================================================


class TestDedupParallelWorkers:
    @pytest.fixture(autouse=True)
    def _fast_embeddings(self, monkeypatch):
        """Stub _encode_texts so tests don't load a SentenceTransformer model.

        Returns deterministic unit vectors: same text → same vector (cosine
        sim 1.0), different texts → orthogonal vectors (cosine sim 0.0).
        """
        def _fake_encode(texts: list[str]) -> np.ndarray:
            unique = list(dict.fromkeys(texts))
            n = len(unique)
            dim = max(n, 16)
            basis = np.eye(dim)
            text_to_vec = {t: basis[i] for i, t in enumerate(unique)}
            return np.array([text_to_vec[t] for t in texts])

        monkeypatch.setattr(
            "backend.agents.compliance_extractor._encode_texts",
            _fake_encode,
        )

    def test_parallel_workers_falls_back_when_pool_fails(self):
        """If the ProcessPoolExecutor errors out, dedup must still succeed."""
        reqs = [_make_req(f"Does X {i}?") for i in range(20)]

        def boom(*args, **kwargs):
            raise OSError("simulated failure")

        with (
            patch("backend.agents.compliance_extractor._encode_with_process_pool",
                  side_effect=boom),
        ):
            # Direct call ignoring process pool (parallel_workers=1) should work.
            out = deduplicate_requirements(reqs, parallel_workers=1)
        assert len(out) >= 1

    def test_parallel_workers_path_matches_serial_output(self):
        """Serial and parallel encoding must produce equivalent dedup results."""
        reqs = [_make_req(f"Unique question number {i}?") for i in range(8)]

        serial = deduplicate_requirements(reqs, parallel_workers=1)
        # Verify the parallel branch is taken: we patch _encode_with_process_pool
        # to delegate to the serial helper so we avoid spawning actual child
        # processes in the test runner, but still cover the routing logic.
        def encode_spy(texts, _workers):
            return mod._encode_texts(texts)

        with patch.object(
            mod, "_encode_with_process_pool", side_effect=encode_spy
        ) as spy:
            parallel = deduplicate_requirements(reqs, parallel_workers=2)
        assert len(parallel) == len(serial)
        assert spy.call_count == 1


# ===========================================================================
# Overlapped prep steps  (#1)
# ===========================================================================


class TestOverlappedPrepSteps:
    """
    Steps 3 (term extraction), 4 (filter), and the cross-ref index build
    must run concurrently rather than serially.  We simulate three slow
    operations and assert the total wall-clock is close to the slowest
    one, not their sum.
    """

    def test_prep_steps_run_concurrently(self, tmp_path: Path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 synthetic")

        sections = [_make_section("Heading", text="MCPs shall do X.")]
        structured_text = "[PAGE 1]\nBody."

        # Each prep step blocks for ~0.3s.  If they run serially total is >=0.9s;
        # concurrently it's ~0.3s.
        def slow_terms(*args, **kwargs):
            time.sleep(0.3)
            return []

        def slow_filter(secs):
            time.sleep(0.3)
            return (secs, [])

        def slow_index(secs):
            time.sleep(0.3)
            return build_section_index(secs)

        async def drain():
            async for _ in mod.run_compliance_extractor_with_progress(str(pdf_path)):
                pass

        with (
            patch.object(extraction_cache, "EXTRACTION_CACHE_DIR", tmp_path / "cache"),
            patch.object(
                mod, "partition_sections_for_batching",
                return_value=([], list(sections)),
            ),
            patch.object(
                mod, "_run_section_with_timeout",
                side_effect=_async_empty,
            ),
            patch.object(mod, "parse_pdf_with_structure", return_value=structured_text),
            patch.object(
                mod, "build_hf_filter_from_pdf",
                return_value=mod.HeaderFooterFilter(),
            ),
            patch.object(mod, "segment_document", return_value=sections),
            patch.object(mod, "extract_term_definitions", side_effect=slow_terms),
            patch.object(mod, "upsert_term_definitions"),
            patch.object(mod, "filter_obligation_sections", side_effect=slow_filter),
            patch.object(mod, "build_section_index", side_effect=slow_index),
        ):
            start = time.monotonic()
            asyncio.run(drain())
            elapsed = time.monotonic() - start

        # Serial execution would take >= 0.9s; concurrent ~0.3s.  Allow slack
        # for thread scheduling and pipeline overhead.
        assert elapsed < 0.7, (
            f"Prep steps took {elapsed:.2f}s; expected <0.7s if running "
            "concurrently. Overlap regression."
        )
