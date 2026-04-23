"""
Evaluation suite for Component 3: Requirement Extraction Workflow.

Measures output quality of the narrative and compliance extractors using
real LLM calls. Unlike unit tests, these evals assert on continuous quality
metrics (recall, format adherence, field completeness) against defined
thresholds.

Usage
-----
    # Run evals only (skipped in CI unless GEMINI_API_KEY is set):
    pytest -m eval backend/evals/eval_extraction.py -v

    # Run with detailed score report:
    pytest -m eval backend/evals/eval_extraction.py -v -s

    # Run a single eval:
    pytest -m eval backend/evals/eval_extraction.py::TestNarrativeExtractorQuality -v -s

Prerequisites
-------------
- GEMINI_API_KEY environment variable must be set.
- Run from the project root (so DATA_DIR resolves correctly).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from backend.models.schemas import ComplianceRequirement, Requirement
from backend.tools.narrative_extractor import run_narrative_extractor
from backend.tools.review_form_parser import parse_review_form

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
EASY_PDF = DATA_DIR / "Example Input Doc - Easy.pdf"
HARD_PDF = DATA_DIR / "Example Input Doc - Hard.pdf"

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

# Narrative extractor
FORMAT_ADHERENCE_MIN = 0.90       # ≥90 % of reqs must start with "Does the P&P state"
REFERENCE_COMPLETENESS_MIN = 0.75  # ≥75 % of reqs must have a non-empty reference
CATEGORY_COMPLETENESS_MIN = 0.75   # ≥75 % of reqs must have a non-empty category
COUNT_LOWER_FACTOR = 0.50          # count ≥ 50 % of ground-truth count
COUNT_UPPER_FACTOR = 1.50          # count ≤ 150 % of ground-truth count
SEMANTIC_RECALL_MIN = 0.60         # ≥60 % of ground-truth reqs semantically covered
SEMANTIC_SIMILARITY_CUTOFF = 0.65  # cosine similarity threshold for a "match"
ID_UNIQUENESS_MIN = 1.0            # 100 % of IDs must be unique

# Compliance extractor
COMPLIANCE_COUNT_MIN = 20          # at least 20 requirements from the 145-page doc
OBLIGATION_TYPE_MIN = 0.40         # ≥40 % of ComplianceRequirements have obligation_type
ACTOR_MIN = 0.40                   # ≥40 % have actor
EVIDENCE_MIN = 0.30                # ≥30 % have evidence_needed
TIMEFRAME_MIN = 0.20               # ≥20 % have timeframe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Accumulates metric scores and formats a human-readable report."""

    name: str
    scores: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)

    def record(self, metric: str, value: float, threshold: float | None = None) -> None:
        self.scores[metric] = value
        if threshold is not None:
            self.thresholds[metric] = threshold

    def passed(self) -> bool:
        return all(
            self.scores[m] >= t
            for m, t in self.thresholds.items()
            if m in self.scores
        )

    def __str__(self) -> str:
        lines = [f"\n{'='*60}", f"  Eval: {self.name}", "=" * 60]
        for metric, score in self.scores.items():
            threshold = self.thresholds.get(metric)
            if threshold is not None:
                status = "PASS" if score >= threshold else "FAIL"
                lines.append(
                    f"  [{status}] {metric:<40} {score:.3f}  (min {threshold:.3f})"
                )
            else:
                lines.append(f"  [INFO] {metric:<40} {score:.3f}")
        for k, v in self.extras.items():
            lines.append(f"  [INFO] {k:<40} {v!r}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _read_pdf_text(path: Path) -> str:
    doc = fitz.open(str(path))
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _semantic_recall(
    ground_truth: list[str],
    predictions: list[str],
    model: SentenceTransformer,
    threshold: float = SEMANTIC_SIMILARITY_CUTOFF,
) -> tuple[float, list[float]]:
    """
    Compute the fraction of ground-truth texts that are semantically
    covered by at least one prediction.

    Returns:
        (recall, per_gt_max_similarities) — recall is in [0, 1];
        per_gt_max_similarities[i] is the highest cosine similarity
        between ground_truth[i] and any prediction.
    """
    if not predictions:
        return 0.0, [0.0] * len(ground_truth)

    gt_emb = _embed(ground_truth, model)    # (N, D)
    pred_emb = _embed(predictions, model)   # (M, D)
    sims = gt_emb @ pred_emb.T              # (N, M)
    max_sims: list[float] = sims.max(axis=1).tolist()
    covered = sum(1 for s in max_sims if s >= threshold)
    return covered / len(ground_truth), max_sims


# ---------------------------------------------------------------------------
# Skip guard — all evals require a real API key
# ---------------------------------------------------------------------------

requires_api_key = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping LLM eval",
)


# ---------------------------------------------------------------------------
# Module-level fixtures (cached across tests in the same session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def easy_text() -> str:
    """Full text of the Easy PDF (structured review form, 14 pages)."""
    assert EASY_PDF.exists(), f"Easy PDF not found: {EASY_PDF}"
    return _read_pdf_text(EASY_PDF)


@pytest.fixture(scope="module")
def ground_truth_requirements(easy_text: str) -> list[Requirement]:
    """64 deterministic requirements from the Easy PDF regex parser."""
    reqs = parse_review_form(easy_text)
    assert len(reqs) == 64, f"Expected 64 ground-truth requirements, got {len(reqs)}"
    return reqs


@pytest.fixture(scope="module")
def embedding_model() -> SentenceTransformer:
    """Lightweight embedding model for semantic similarity (already a project dep)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def narrative_extraction_result(easy_text: str) -> list[Requirement]:
    """
    Run the narrative LLM extractor once and cache for all tests in this module.
    Fixture-level caching avoids redundant API calls.

    A 240-second thread timeout guards against the OpenAI SDK's 600 s default,
    which would otherwise cause a silent hang during fixture setup.
    """
    import concurrent.futures

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set — skipping LLM eval")

    print("\n[fixture] running narrative extractor on Easy PDF — may take 30–120s …", flush=True)
    t0 = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_narrative_extractor, easy_text)
        try:
            reqs = future.result(timeout=240)
        except concurrent.futures.TimeoutError:
            pytest.fail("run_narrative_extractor timed out after 240s — check API connectivity")

    elapsed = time.perf_counter() - t0
    print(f"[fixture] done — {len(reqs)} requirements in {elapsed:.1f}s", flush=True)
    return reqs


@pytest.fixture(scope="module")
def compliance_extraction_result() -> list[ComplianceRequirement]:
    """
    Run the compliance extractor on the Hard PDF once for all compliance evals.

    A 600-second thread timeout guards against silent hangs; the 145-page doc
    legitimately takes 3–8 minutes with parallel LLM calls.
    """
    import asyncio
    import concurrent.futures

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set — skipping LLM eval")

    from backend.agents.compliance_extractor import run_compliance_extractor

    assert HARD_PDF.exists(), f"Hard PDF not found: {HARD_PDF}"

    print("\n[fixture] running compliance extractor on Hard PDF — may take 3–8 min …", flush=True)
    t0 = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, run_compliance_extractor(str(HARD_PDF)))
        try:
            reqs = future.result(timeout=600)
        except concurrent.futures.TimeoutError:
            pytest.fail("run_compliance_extractor timed out after 600s — check API connectivity")

    elapsed = time.perf_counter() - t0
    print(f"[fixture] done — {len(reqs)} requirements in {elapsed:.1f}s", flush=True)
    return reqs


# ===========================================================================
# Eval 1 — Narrative Extractor: Output Format and Field Completeness
# ===========================================================================


@pytest.mark.eval
@requires_api_key
class TestNarrativeExtractorQuality:
    """
    Quality checks on the narrative LLM extractor run against the Easy PDF.

    The Easy PDF is a structured 14-page review form whose 64 requirements
    are our ground truth (deterministically extracted by the regex parser).
    Running the narrative extractor on the same text lets us compare LLM
    extraction quality against a known-correct baseline.
    """

    def test_returns_nonempty_list(self, narrative_extraction_result):
        """Extractor must return at least one requirement."""
        report = EvalReport("returns_nonempty")
        report.record("count", len(narrative_extraction_result), threshold=1.0)
        print(report)
        assert len(narrative_extraction_result) > 0, "Narrative extractor returned no requirements"

    def test_count_within_tolerance(
        self,
        narrative_extraction_result,
        ground_truth_requirements,
    ):
        """
        Extracted count must be within [50%, 150%] of ground-truth count.
        Ground truth: 64 requirements.  Acceptable range: 32–96.
        """
        gt_count = len(ground_truth_requirements)
        extracted_count = len(narrative_extraction_result)
        lower = gt_count * COUNT_LOWER_FACTOR
        upper = gt_count * COUNT_UPPER_FACTOR

        report = EvalReport("count_accuracy")
        report.record("ground_truth_count", float(gt_count))
        report.record("extracted_count", float(extracted_count))
        report.record("lower_bound", lower)
        report.record("upper_bound", upper)
        print(report)

        assert lower <= extracted_count <= upper, (
            f"Extracted {extracted_count} requirements; expected between "
            f"{int(lower)} and {int(upper)} (ground truth: {gt_count})"
        )

    def test_format_adherence(self, narrative_extraction_result):
        """
        At least FORMAT_ADHERENCE_MIN of extracted requirements must begin
        with "Does the P&P state" (the canonical question format).
        """
        reqs = narrative_extraction_result
        n = len(reqs)
        prefix = "does the p&p state"
        conforming = sum(1 for r in reqs if r.text.lower().startswith(prefix))
        rate = conforming / n if n else 0.0

        report = EvalReport("format_adherence")
        report.record("adherence_rate", rate, threshold=FORMAT_ADHERENCE_MIN)
        report.extras["non_conforming"] = [
            r.text[:80] for r in reqs if not r.text.lower().startswith(prefix)
        ][:5]
        print(report)

        assert rate >= FORMAT_ADHERENCE_MIN, (
            f"Format adherence {rate:.1%} < {FORMAT_ADHERENCE_MIN:.0%} threshold. "
            f"Non-conforming examples: "
            f"{[r.text[:60] for r in reqs if not r.text.lower().startswith(prefix)][:3]}"
        )

    def test_reference_completeness(self, narrative_extraction_result):
        """
        At least REFERENCE_COMPLETENESS_MIN of extracted requirements must
        have a non-empty reference field.
        """
        reqs = narrative_extraction_result
        n = len(reqs)
        with_ref = sum(1 for r in reqs if r.reference and r.reference.strip())
        rate = with_ref / n if n else 0.0

        report = EvalReport("reference_completeness")
        report.record("rate", rate, threshold=REFERENCE_COMPLETENESS_MIN)
        report.extras["missing_reference_count"] = n - with_ref
        print(report)

        assert rate >= REFERENCE_COMPLETENESS_MIN, (
            f"Reference completeness {rate:.1%} < {REFERENCE_COMPLETENESS_MIN:.0%} threshold. "
            f"{n - with_ref}/{n} requirements have no reference."
        )

    def test_category_completeness(self, narrative_extraction_result):
        """
        At least CATEGORY_COMPLETENESS_MIN of extracted requirements must
        have a non-empty category field.
        """
        reqs = narrative_extraction_result
        n = len(reqs)
        with_cat = sum(1 for r in reqs if r.category and r.category.strip())
        rate = with_cat / n if n else 0.0

        report = EvalReport("category_completeness")
        report.record("rate", rate, threshold=CATEGORY_COMPLETENESS_MIN)
        report.extras["unique_categories"] = sorted(
            {r.category for r in reqs if r.category}
        )
        print(report)

        assert rate >= CATEGORY_COMPLETENESS_MIN, (
            f"Category completeness {rate:.1%} < {CATEGORY_COMPLETENESS_MIN:.0%} threshold. "
            f"{n - with_cat}/{n} requirements have no category."
        )

    def test_unique_ids(self, narrative_extraction_result):
        """All extracted requirement IDs must be unique (no duplicates)."""
        ids = [r.id for r in narrative_extraction_result]
        unique = len(set(ids))
        total = len(ids)
        rate = unique / total if total else 1.0

        report = EvalReport("id_uniqueness")
        report.record("uniqueness_rate", rate, threshold=ID_UNIQUENESS_MIN)
        duplicate_ids = sorted({i for i in ids if ids.count(i) > 1})
        report.extras["duplicate_ids"] = duplicate_ids
        print(report)

        assert rate >= ID_UNIQUENESS_MIN, (
            f"ID uniqueness {rate:.1%} < 100%. Duplicate IDs: {duplicate_ids}"
        )

    def test_ids_are_positive_integers(self, narrative_extraction_result):
        """All extracted IDs must be positive integers."""
        bad = [r.id for r in narrative_extraction_result if not isinstance(r.id, int) or r.id < 1]
        assert not bad, f"Non-positive integer IDs found: {bad[:10]}"

    def test_no_empty_text(self, narrative_extraction_result):
        """No requirement may have an empty or whitespace-only text."""
        empty = [r.id for r in narrative_extraction_result if not r.text or not r.text.strip()]
        assert not empty, f"Requirements with empty text: {empty}"


# ===========================================================================
# Eval 2 — Narrative Extractor: Semantic Recall Against Ground Truth
# ===========================================================================


@pytest.mark.eval
@requires_api_key
class TestNarrativeExtractorRecall:
    """
    Measures how well the LLM covers the 64 ground-truth requirements.

    Uses cosine similarity of sentence embeddings (all-MiniLM-L6-v2) rather
    than exact-match, so equivalent paraphrases still count as covered.
    """

    def test_semantic_recall(
        self,
        narrative_extraction_result,
        ground_truth_requirements,
        embedding_model,
    ):
        """
        At least SEMANTIC_RECALL_MIN of ground-truth requirements must be
        semantically matched (cosine similarity ≥ SEMANTIC_SIMILARITY_CUTOFF)
        by at least one extracted requirement.
        """
        gt_texts = [r.text for r in ground_truth_requirements]
        pred_texts = [r.text for r in narrative_extraction_result]

        recall, max_sims = _semantic_recall(gt_texts, pred_texts, embedding_model)

        # Build a ranked list of the hardest-to-match ground-truth requirements
        # for diagnostic purposes
        indexed = sorted(enumerate(max_sims), key=lambda x: x[1])
        missed = [
            (ground_truth_requirements[i].id, ground_truth_requirements[i].text[:80], s)
            for i, s in indexed
            if s < SEMANTIC_SIMILARITY_CUTOFF
        ][:10]

        report = EvalReport("semantic_recall")
        report.record("recall", recall, threshold=SEMANTIC_RECALL_MIN)
        report.record("mean_similarity", float(np.mean(max_sims)))
        report.record("median_similarity", float(np.median(max_sims)))
        report.extras["ground_truth_count"] = len(gt_texts)
        report.extras["prediction_count"] = len(pred_texts)
        report.extras["missed_examples"] = missed
        print(report)

        assert recall >= SEMANTIC_RECALL_MIN, (
            f"Semantic recall {recall:.1%} < {SEMANTIC_RECALL_MIN:.0%} threshold. "
            f"Top missed GT reqs: {[m[1] for m in missed[:3]]}"
        )

    def test_no_hallucinated_format(
        self,
        narrative_extraction_result,
        embedding_model,
    ):
        """
        Every extracted requirement must have non-trivial similarity (≥ 0.30)
        to at least one ground-truth requirement.  Very low similarity indicates
        the model invented requirements not present in the document.
        """
        gt_texts = [r.text for r in parse_review_form(_read_pdf_text(EASY_PDF))]
        pred_texts = [r.text for r in narrative_extraction_result]

        pred_emb = _embed(pred_texts, embedding_model)
        gt_emb = _embed(gt_texts, embedding_model)
        sims = pred_emb @ gt_emb.T           # (M, N)
        max_sims: list[float] = sims.max(axis=1).tolist()

        HALLUCINATION_CUTOFF = 0.30
        suspicious = [
            (narrative_extraction_result[i].text[:80], s)
            for i, s in enumerate(max_sims)
            if s < HALLUCINATION_CUTOFF
        ]

        report = EvalReport("hallucination_check")
        report.record("min_similarity", float(min(max_sims)))
        report.record("mean_similarity", float(np.mean(max_sims)))
        report.extras["suspicious_requirements"] = suspicious[:5]
        print(report)

        hallucination_rate = len(suspicious) / len(pred_texts) if pred_texts else 0.0
        MAX_HALLUCINATION_RATE = 0.10
        assert hallucination_rate <= MAX_HALLUCINATION_RATE, (
            f"Hallucination rate {hallucination_rate:.1%} > {MAX_HALLUCINATION_RATE:.0%}. "
            f"Suspicious: {[s[0] for s in suspicious[:3]]}"
        )


# ===========================================================================
# Eval 3 — _parse_requirements: Robustness to Realistic LLM Output Formats
# ===========================================================================


@pytest.mark.eval
class TestParseRequirementsRobustness:
    """
    Tests the output parser against LLM output variations that don't
    require a live API call.  These verify that real-world LLM quirks
    (markdown fences, prose preamble, mixed types) are handled cleanly.
    """

    # Import once at class level for reference in helpers
    from backend.tools.narrative_extractor import _parse_requirements as _parse

    def _item(self, id: int = 1, **overrides) -> dict:
        base = {
            "id": id,
            "text": f"Does the P&P state that requirement {id} is met?",
            "reference": f"Section {id}",
            "category": "General",
        }
        return {**base, **overrides}

    def test_plain_json_array(self):
        import json
        from backend.tools.narrative_extractor import _parse_requirements
        data = json.dumps([self._item(1), self._item(2)])
        reqs = _parse_requirements(data)
        assert len(reqs) == 2
        assert reqs[0].id == 1

    def test_markdown_fenced_json(self):
        import json
        from backend.tools.narrative_extractor import _parse_requirements
        inner = json.dumps([self._item(1)])
        fenced = f"```json\n{inner}\n```"
        reqs = _parse_requirements(fenced)
        assert len(reqs) == 1

    def test_fenced_without_language_tag(self):
        import json
        from backend.tools.narrative_extractor import _parse_requirements
        inner = json.dumps([self._item(1)])
        fenced = f"```\n{inner}\n```"
        reqs = _parse_requirements(fenced)
        assert len(reqs) == 1

    def test_array_embedded_in_prose(self):
        import json
        from backend.tools.narrative_extractor import _parse_requirements
        inner = json.dumps([self._item(1), self._item(2)])
        wrapped = f"Here are the compliance requirements I found:\n{inner}\nEnd."
        reqs = _parse_requirements(wrapped)
        assert len(reqs) == 2

    def test_malformed_items_skipped(self):
        """Items that fail Pydantic validation are dropped, not raised."""
        from backend.tools.narrative_extractor import _parse_requirements
        data = [
            self._item(1),
            {"id": "not-an-int", "text": None},      # bad id type
            {"id": 3},                                 # missing required text
            self._item(4),
        ]
        reqs = _parse_requirements(data)
        ids = {r.id for r in reqs}
        assert 1 in ids
        assert 4 in ids
        assert len(reqs) == 2

    def test_extra_fields_ignored(self):
        """LLM output may include extra fields — they should be silently ignored."""
        import json
        from backend.tools.narrative_extractor import _parse_requirements
        item = {**self._item(1), "extra_field": "should be ignored", "another": 99}
        reqs = _parse_requirements(json.dumps([item]))
        assert len(reqs) == 1
        assert reqs[0].id == 1
        assert not hasattr(reqs[0], "extra_field")

    def test_large_output_parsed_correctly(self):
        """Parser handles output with 100+ items without truncation."""
        import json
        from backend.tools.narrative_extractor import _parse_requirements
        items = [self._item(i) for i in range(1, 101)]
        reqs = _parse_requirements(json.dumps(items))
        assert len(reqs) == 100
        assert reqs[0].id == 1
        assert reqs[-1].id == 100

    def test_raises_on_invalid_json(self):
        from backend.tools.narrative_extractor import _parse_requirements
        with pytest.raises(ValueError, match="non-JSON"):
            _parse_requirements("{this is not json")

    def test_raises_on_non_array_json(self):
        from backend.tools.narrative_extractor import _parse_requirements
        with pytest.raises(ValueError, match="JSON array"):
            _parse_requirements('{"id": 1, "text": "a question?"}')

    def test_empty_array(self):
        """An empty JSON array is valid — returns empty list without raising."""
        from backend.tools.narrative_extractor import _parse_requirements
        reqs = _parse_requirements("[]")
        assert reqs == []

    def test_unicode_in_text(self):
        """Requirements with non-ASCII characters round-trip correctly."""
        import json
        from backend.tools.narrative_extractor import _parse_requirements
        item = self._item(1)
        item["text"] = "Does the P&P state that MCPs must comply with §1396 requirements?"
        reqs = _parse_requirements(json.dumps([item]))
        assert "§1396" in reqs[0].text

    def test_whitespace_only_fenced_block(self):
        """A fenced block containing only whitespace raises ValueError, not crashes."""
        from backend.tools.narrative_extractor import _parse_requirements
        with pytest.raises(ValueError):
            _parse_requirements("```json\n   \n```")


# ===========================================================================
# Eval 4 — Router: End-to-End Routing Without Mocks
# ===========================================================================


@pytest.mark.eval
@requires_api_key
class TestRouterEndToEnd:
    """
    Integration checks that run the actual extractor router end-to-end.
    These tests do NOT mock the extractors, so they validate the full
    pipeline from PDF file to doc_type + requirements.
    """

    @pytest.mark.anyio
    async def test_easy_pdf_routes_to_narrative(self):
        """Easy PDF (14 pages ≤ 20) must route to 'narrative'."""
        from backend.agents.extractor import classify_and_extract

        doc_type, reqs = await classify_and_extract(str(EASY_PDF))

        report = EvalReport("easy_pdf_routing")
        report.record("requirement_count", float(len(reqs)))
        report.extras["doc_type"] = doc_type
        print(report)

        assert doc_type == "narrative", f"Expected 'narrative', got {doc_type!r}"
        assert len(reqs) > 0, "No requirements extracted from Easy PDF"

    @pytest.mark.anyio
    async def test_hard_pdf_routes_to_compliance(self):
        """Hard PDF (145 pages > 20) must route to 'compliance'."""
        from backend.agents.extractor import classify_and_extract

        doc_type, reqs = await classify_and_extract(str(HARD_PDF))

        report = EvalReport("hard_pdf_routing")
        report.record("requirement_count", float(len(reqs)))
        report.extras["doc_type"] = doc_type
        print(report)

        assert doc_type == "compliance", f"Expected 'compliance', got {doc_type!r}"
        assert len(reqs) >= COMPLIANCE_COUNT_MIN, (
            f"Compliance extractor returned only {len(reqs)} requirements "
            f"(expected ≥ {COMPLIANCE_COUNT_MIN})"
        )


# ===========================================================================
# Eval 5 — Compliance Extractor: Enriched Field Richness
# ===========================================================================


@pytest.mark.eval
@requires_api_key
class TestComplianceExtractorFieldRichness:
    """
    Verifies that the compliance extraction agent populates the
    enriched fields on ComplianceRequirement (not just the base
    Requirement fields).

    A low field population rate means the LLM is not following the
    structured output schema.
    """

    def _field_rate(
        self,
        reqs: list[ComplianceRequirement],
        field_name: str,
    ) -> float:
        if not reqs:
            return 0.0
        populated = sum(
            1 for r in reqs if getattr(r, field_name, None) is not None
            and str(getattr(r, field_name, "")).strip()
        )
        return populated / len(reqs)

    def test_base_fields_complete(self, compliance_extraction_result):
        """All ComplianceRequirements must have non-empty id, text, and reference."""
        reqs = compliance_extraction_result
        missing_text = [r.id for r in reqs if not r.text or not r.text.strip()]
        assert not missing_text, f"Requirements with empty text: {missing_text[:5]}"

    def test_requirement_count(self, compliance_extraction_result):
        """Compliance extractor must return at least COMPLIANCE_COUNT_MIN requirements."""
        count = len(compliance_extraction_result)
        assert count >= COMPLIANCE_COUNT_MIN, (
            f"Compliance extractor returned {count} requirements; "
            f"expected ≥ {COMPLIANCE_COUNT_MIN}"
        )

    def test_obligation_type_population(self, compliance_extraction_result):
        """At least OBLIGATION_TYPE_MIN of requirements have an obligation_type."""
        rate = self._field_rate(compliance_extraction_result, "obligation_type")
        report = EvalReport("obligation_type_population")
        report.record("rate", rate, threshold=OBLIGATION_TYPE_MIN)
        print(report)
        assert rate >= OBLIGATION_TYPE_MIN, (
            f"obligation_type populated in {rate:.1%} of requirements "
            f"(min {OBLIGATION_TYPE_MIN:.0%})"
        )

    def test_actor_population(self, compliance_extraction_result):
        """At least ACTOR_MIN of requirements have an actor field."""
        rate = self._field_rate(compliance_extraction_result, "actor")
        report = EvalReport("actor_population")
        report.record("rate", rate, threshold=ACTOR_MIN)
        print(report)
        assert rate >= ACTOR_MIN, (
            f"actor populated in {rate:.1%} of requirements (min {ACTOR_MIN:.0%})"
        )

    def test_evidence_needed_population(self, compliance_extraction_result):
        """At least EVIDENCE_MIN of requirements have evidence_needed."""
        rate = self._field_rate(compliance_extraction_result, "evidence_needed")
        report = EvalReport("evidence_needed_population")
        report.record("rate", rate, threshold=EVIDENCE_MIN)
        print(report)
        assert rate >= EVIDENCE_MIN, (
            f"evidence_needed populated in {rate:.1%} of requirements "
            f"(min {EVIDENCE_MIN:.0%})"
        )

    def test_timeframe_population(self, compliance_extraction_result):
        """At least TIMEFRAME_MIN of requirements have a timeframe."""
        rate = self._field_rate(compliance_extraction_result, "timeframe")
        report = EvalReport("timeframe_population")
        report.record("rate", rate, threshold=TIMEFRAME_MIN)
        print(report)
        assert rate >= TIMEFRAME_MIN, (
            f"timeframe populated in {rate:.1%} of requirements "
            f"(min {TIMEFRAME_MIN:.0%})"
        )

    def test_obligation_type_values_are_valid(self, compliance_extraction_result):
        """
        obligation_type values must be drawn from the expected vocabulary.
        Unknown values indicate schema drift between the prompt and the model.
        """
        valid_types = {"mandatory", "prohibition", "conditional", "recommended", "informational"}
        invalid = [
            (r.id, r.obligation_type)
            for r in compliance_extraction_result
            if r.obligation_type and r.obligation_type.lower() not in valid_types
        ]
        # Allow up to 10% unknown values (LLMs may paraphrase)
        rate_invalid = len(invalid) / len(compliance_extraction_result) if compliance_extraction_result else 0
        report = EvalReport("obligation_type_vocabulary")
        report.record("invalid_rate", rate_invalid, threshold=0.0)
        report.extras["invalid_examples"] = invalid[:5]
        print(report)
        assert rate_invalid <= 0.10, (
            f"obligation_type vocabulary violation rate {rate_invalid:.1%} > 10%. "
            f"Examples: {invalid[:3]}"
        )

    def test_parent_child_ids_valid(self, compliance_extraction_result):
        """
        Sub-requirements referencing a parent_id must point to an existing ID.
        Dangling parent_ids indicate the hierarchical assembly is broken.
        """
        all_ids = {r.id for r in compliance_extraction_result}
        dangling = [
            (r.id, r.parent_id)
            for r in compliance_extraction_result
            if r.parent_id is not None and r.parent_id not in all_ids
        ]
        report = EvalReport("parent_child_consistency")
        report.record("dangling_parent_count", float(len(dangling)))
        report.extras["dangling_examples"] = dangling[:5]
        print(report)
        assert len(dangling) == 0, (
            f"Requirements with dangling parent_id: {dangling[:5]}"
        )

    def test_unique_ids(self, compliance_extraction_result):
        """All compliance requirement IDs must be unique."""
        ids = [r.id for r in compliance_extraction_result]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        assert not duplicates, f"Duplicate compliance requirement IDs: {duplicates[:10]}"


# ===========================================================================
# Eval 6 — Regression Stability: Re-run Narrative Extractor, Compare Counts
# ===========================================================================


@pytest.mark.eval
@requires_api_key
class TestNarrativeExtractorStability:
    """
    Checks that two independent extraction runs on the same document
    produce counts within 20% of each other.  High variance between
    runs suggests the model is unstable at temperature=0.2.
    """

    def test_count_stability_across_two_runs(self, easy_text):
        """
        Run the narrative extractor twice on the Easy PDF and verify the
        requirement counts are within 20% of each other.
        """
        reqs1 = run_narrative_extractor(easy_text)
        reqs2 = run_narrative_extractor(easy_text)

        count1 = len(reqs1)
        count2 = len(reqs2)

        if count1 == 0 or count2 == 0:
            pytest.fail(f"One run returned 0 requirements: run1={count1}, run2={count2}")

        relative_diff = abs(count1 - count2) / max(count1, count2)
        MAX_VARIANCE = 0.20

        report = EvalReport("stability")
        report.record("run1_count", float(count1))
        report.record("run2_count", float(count2))
        report.record("relative_diff", relative_diff, threshold=MAX_VARIANCE)
        print(report)

        assert relative_diff <= MAX_VARIANCE, (
            f"Count variance {relative_diff:.1%} > {MAX_VARIANCE:.0%} between runs "
            f"(run1={count1}, run2={count2}). Model may be too unstable at temperature=0.2."
        )
