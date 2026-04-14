"""
Compliance Extraction Agent — Component 8.

A specialized extraction pipeline for long regulatory PDFs (>20 pages).

Pipeline:
  1. Structured PDF parse with heading/table markers
  2. Section segmentation
  3. Obligation-language filtering
  4. Per-section LLM extraction (parallelized)
  5. Embedding-based deduplication
  6. Hierarchy assembly
  7. ID assignment and final output

Entry point: :func:`run_compliance_extractor`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import numpy as np
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from smolagents import OpenAIModel, Tool, ToolCallingAgent

from backend.config import (
    DEDUP_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_API_BASE,
    LLM_MODEL_ID,
    MAX_CONCURRENT_WORKERS,
)
from backend.models.schemas import ComplianceRequirement, Requirement
from backend.tools.document_segmenter import (
    DocumentSection,
    filter_obligation_sections,
    segment_document,
)
from backend.tools.pdf_parser import parse_pdf_with_structure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section text tool (one instance per section, closure pattern)
# ---------------------------------------------------------------------------


class GetSectionTextTool(Tool):
    """
    A no-input smolagents tool that returns the text of the document section
    currently being processed.  Instantiated per extraction call so the
    closure over ``section`` is safe for parallel use.
    """

    name = "get_section_text"
    description = (
        "Returns the full text of the document section to analyze, including "
        "any tables within that section. Call this first before extracting requirements."
    )
    inputs: dict = {}
    output_type = "string"

    def __init__(self, section: DocumentSection) -> None:
        super().__init__()
        self._section = section

    def forward(self) -> str:  # type: ignore[override]
        s = self._section
        parts: list[str] = [
            f"Section heading: {s.heading}",
            f"Pages: {s.page_start}\u2013{s.page_end}",
            "",
            s.text,
        ]
        for i, table in enumerate(s.tables, start=1):
            parts.append(f"\n[TABLE {i}]")
            parts.append(table)
            parts.append(f"[/TABLE {i}]")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Section extraction prompt
# ---------------------------------------------------------------------------

_SECTION_EXTRACTION_INSTRUCTIONS = """\
You are a compliance requirement extractor. You will receive ONE section of a
regulatory document. Extract every compliance obligation from this section.

WHAT TO EXTRACT:
- Statements using: must, shall, required to, is responsible for, prohibited,
  may not, should (if normative), within X days, annually, quarterly, upon request
- Requirements embedded in tables (each row may be a separate requirement)
- Conditional requirements ("if X, then must Y")

WHAT TO SKIP:
- Definitions (unless they contain embedded obligations)
- Background/context paragraphs with no actionable language
- Examples and illustrations (unless they clarify a requirement)
- Section headings and page numbers

For each requirement, output a JSON object:
{
  "text": "Does the P&P state that [rephrased as yes/no question]?",
  "exact_quote": "verbatim text from the document",
  "reference": "Section heading, page N",
  "category": "topic label",
  "obligation_type": "mandatory | prohibition | conditional | recommended",
  "obligation_level": "mandatory | conditional_mandatory | recommended | informational",
  "actor": "who must act",
  "action_required": "what must be done",
  "condition": "trigger condition, or null",
  "timeframe": "deadline/frequency, or null",
  "evidence_needed": "what proves compliance, or null",
  "risk_area": "Privacy | Security | Financial | Operations | Clinical | Administrative"
}

IMPORTANT:
- Preserve the EXACT obligation language — do not summarize or paraphrase the
  exact_quote field.
- Capture conditional logic: "if X, then Y" → condition="if X", action="Y".
- Tables: treat each row as a potential separate requirement.
- If a statement is ambiguous between mandatory and recommended, label it
  "conditional_mandatory" and note the ambiguity in the condition field.

Return a JSON array of requirement objects via final_answer.
"""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_compliance_requirements(
    raw: Any,
    section: DocumentSection,
) -> list[ComplianceRequirement]:
    """
    Convert the agent's ``final_answer`` into a list of
    :class:`ComplianceRequirement` objects.

    Assigns ``id=0`` as a placeholder (replaced in :func:`finalize_requirements`).
    Sets ``section_heading`` from the source section.
    Clears any ``parent_id`` from LLM output (set correctly in
    :func:`assemble_hierarchy`).
    """
    # Already a list
    if isinstance(raw, list):
        data: list[Any] = [item for item in raw if isinstance(item, dict)]
    else:
        text = str(raw).strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:]
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            text = "\n".join(inner).strip()
        # Extract outermost JSON array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Section %r: non-JSON output from agent: %s. Raw (200): %s",
                section.heading,
                exc,
                str(raw)[:200],
            )
            return []
        if not isinstance(data, list):
            logger.warning(
                "Section %r: expected JSON array, got %s.",
                section.heading,
                type(data).__name__,
            )
            return []

    requirements: list[ComplianceRequirement] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Inject required id placeholder and section heading
        item["id"] = 0
        item.setdefault("section_heading", section.heading)
        item["parent_id"] = None  # will be set in assemble_hierarchy
        try:
            req = ComplianceRequirement(**item)
            requirements.append(req)
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning(
                "Section %r: skipping malformed requirement %r: %s",
                section.heading,
                item,
                exc,
            )

    return requirements


# ---------------------------------------------------------------------------
# Per-section LLM extraction
# ---------------------------------------------------------------------------


def _make_model() -> OpenAIModel:
    return OpenAIModel(
        model_id=LLM_MODEL_ID,
        api_key=GEMINI_API_KEY,
        api_base=GEMINI_API_BASE,
        temperature=0.2,
    )


def run_section_extractor(section: DocumentSection) -> list[ComplianceRequirement]:
    """
    Run the per-section ToolCallingAgent for one :class:`DocumentSection`.

    Returns an empty list (with a logged warning) if the agent fails or
    returns malformed JSON — never raises.
    """
    tool = GetSectionTextTool(section)
    agent = ToolCallingAgent(
        tools=[tool],
        model=_make_model(),
        max_steps=5,
        name="section_requirement_extractor",
        description="Extracts compliance requirements from one document section.",
        instructions=_SECTION_EXTRACTION_INSTRUCTIONS,
        verbosity_level=0,
    )
    try:
        result = agent.run(
            f"Call get_section_text to read the section '{section.heading}', "
            "then extract all compliance requirements and return them as a "
            "JSON array via final_answer."
        )
        return _parse_compliance_requirements(result, section)
    except Exception as exc:
        logger.warning(
            "Section %r: extraction failed: %s. Marking for human review.",
            section.heading,
            exc,
        )
        return []


# ---------------------------------------------------------------------------
# Parallelized extraction across all sections
# ---------------------------------------------------------------------------


async def extract_from_all_sections(
    sections: list[DocumentSection],
) -> list[ComplianceRequirement]:
    """
    Extract compliance requirements from all sections concurrently.

    Uses a semaphore to cap parallelism at ``MAX_CONCURRENT_WORKERS``.
    Each section runs in a thread via :func:`asyncio.to_thread`.

    Args:
        sections: Obligation-bearing sections from the filtering step.

    Returns:
        Flat list of all extracted :class:`ComplianceRequirement` objects.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

    async def extract_one(section: DocumentSection) -> list[ComplianceRequirement]:
        async with semaphore:
            return await asyncio.to_thread(run_section_extractor, section)

    tasks = [asyncio.create_task(extract_one(s)) for s in sections]
    all_reqs: list[ComplianceRequirement] = []
    for coro in asyncio.as_completed(tasks):
        section_reqs = await coro
        all_reqs.extend(section_reqs)

    return all_reqs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _completeness_score(req: ComplianceRequirement) -> int:
    """Score a requirement by how many optional fields are populated."""
    optional_fields = [
        req.exact_quote,
        req.obligation_type,
        req.obligation_level,
        req.actor,
        req.action_required,
        req.condition,
        req.timeframe,
        req.evidence_needed,
        req.risk_area,
    ]
    return sum(1 for f in optional_fields if f is not None) + len(req.text or "")


def deduplicate_requirements(
    requirements: list[ComplianceRequirement],
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> list[ComplianceRequirement]:
    """
    Merge near-duplicate requirements using embedding cosine similarity.

    For each cluster of similar requirements (similarity > threshold):
    1. Pick the most complete one as canonical (most fields populated).
    2. Merge references from duplicates into the canonical's reference field.
    3. Drop the duplicates.

    Args:
        requirements: Raw requirements from the extraction step.
        similarity_threshold: Cosine similarity cutoff (default 0.90).

    Returns:
        Deduplicated list of :class:`ComplianceRequirement` objects.
    """
    if len(requirements) <= 1:
        return requirements

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [r.text for r in requirements]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    # Since embeddings are L2-normalised, dot product == cosine similarity
    sim_matrix: np.ndarray = np.dot(embeddings, embeddings.T)

    n = len(requirements)
    merged = [False] * n
    result: list[ComplianceRequirement] = []

    for i in range(n):
        if merged[i]:
            continue
        cluster = [i]
        for j in range(i + 1, n):
            if merged[j]:
                continue
            if float(sim_matrix[i, j]) >= similarity_threshold:
                cluster.append(j)
                merged[j] = True

        # Select the most complete requirement as canonical
        canonical_idx = max(cluster, key=lambda k: _completeness_score(requirements[k]))
        canonical = requirements[canonical_idx].model_copy(deep=True)

        # Merge references from all cluster members
        if len(cluster) > 1:
            refs = list(
                dict.fromkeys(
                    requirements[k].reference
                    for k in cluster
                    if requirements[k].reference
                )
            )
            if refs:
                canonical = canonical.model_copy(update={"reference": "; ".join(refs)})

        result.append(canonical)

    logger.debug(
        "Deduplication: %d \u2192 %d requirements (threshold=%.2f).",
        n,
        len(result),
        similarity_threshold,
    )
    return result


# ---------------------------------------------------------------------------
# Hierarchy assembly
# ---------------------------------------------------------------------------


def assemble_hierarchy(
    requirements: list[ComplianceRequirement],
    sections: list[DocumentSection],
) -> list[ComplianceRequirement]:
    """
    Set ``parent_id`` on sub-requirements based on document structure.

    Rules:
    - Requirements from H3+ sections get ``parent_id`` set to the list index
      of the nearest H1/H2 requirement above them (0-indexed; resolved to
      1-indexed IDs by :func:`finalize_requirements`).
    - Requirements from H1/H2 sections have ``parent_id = None``.

    The ``section_heading`` field on each requirement is used to look up the
    heading level from *sections*.

    Args:
        requirements: Deduplicated requirements (``id`` not yet assigned).
        sections:     All segments from :func:`segment_document`, used to
                      look up heading levels.

    Returns:
        The same list with ``parent_id`` fields populated.
    """
    # Build heading → level lookup
    section_levels: dict[str, int] = {s.heading: s.level for s in sections}

    # Clear any LLM-provided parent_id values
    for req in requirements:
        req.parent_id = None

    # Track the most recent top-level parent index (H1 or H2)
    last_parent_index: int | None = None

    for i, req in enumerate(requirements):
        heading = req.section_heading or ""
        level = section_levels.get(heading, 0)

        if level in (0, 1, 2):
            # Top-level or unknown: this is a potential parent
            last_parent_index = i
        else:
            # H3+: link to the last H1/H2 parent
            if last_parent_index is not None:
                req.parent_id = last_parent_index  # 0-indexed; resolved in finalize

    return requirements


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def finalize_requirements(
    requirements: list[ComplianceRequirement],
) -> list[Requirement]:
    """
    Assign sequential 1-indexed IDs and resolve ``parent_id`` references.

    ``parent_id`` is stored as a 0-indexed list position by
    :func:`assemble_hierarchy`; this converts it to the 1-indexed final ID.

    Args:
        requirements: Requirements with temporary 0-indexed ``parent_id``.

    Returns:
        The same objects with final ``id`` and ``parent_id`` values.
    """
    for i, req in enumerate(requirements, start=1):
        req.id = i

    for req in requirements:
        if req.parent_id is not None:
            req.parent_id = req.parent_id + 1  # 0-indexed position → 1-indexed ID

    return requirements  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Table-specific extraction helper
# ---------------------------------------------------------------------------


def extract_table_requirements(
    table_text: str,
    section_heading: str,
    page_number: int,
) -> list[ComplianceRequirement]:
    """
    Extract requirements from a single table.

    Attempts rule-based column mapping first; falls back to LLM extraction
    on the raw table text if column headers cannot be identified.

    Args:
        table_text:      Pipe-delimited table text (``| col | val |`` format).
        section_heading: Heading of the enclosing section (for citation).
        page_number:     Page number for citation.

    Returns:
        List of :class:`ComplianceRequirement` objects, one per table row.
    """
    lines = [line.strip() for line in table_text.strip().splitlines() if line.strip()]
    if not lines:
        return []

    # Parse header row
    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    header_lower = [h.lower() for h in header_cells]

    # Try to map columns to known fields
    col_map: dict[str, int] = {}
    for i, h in enumerate(header_lower):
        if any(kw in h for kw in ("requirement", "description", "obligation", "control")):
            col_map.setdefault("text", i)
        if any(kw in h for kw in ("owner", "responsible", "party", "actor", "role")):
            col_map.setdefault("actor", i)
        if any(kw in h for kw in ("deadline", "frequency", "timeframe", "period")):
            col_map.setdefault("timeframe", i)
        if any(kw in h for kw in ("evidence", "documentation", "artifact")):
            col_map.setdefault("evidence_needed", i)

    # If we cannot identify the requirement text column, fall back to LLM
    if "text" not in col_map:
        logger.debug(
            "Table in %r: could not identify requirement column — using LLM fallback.",
            section_heading,
        )
        dummy_section = DocumentSection(
            heading=section_heading,
            level=2,
            text="",
            tables=[table_text],
            page_start=page_number,
            page_end=page_number,
        )
        return run_section_extractor(dummy_section)

    requirements: list[ComplianceRequirement] = []
    for row_line in lines[1:]:
        cells = [c.strip() for c in row_line.split("|") if c.strip()]
        if len(cells) <= col_map["text"]:
            continue
        req_text = cells[col_map["text"]]
        if not req_text:
            continue

        kwargs: dict[str, Any] = {
            "id": 0,
            "text": f"Does the P&P state that {req_text}?",
            "exact_quote": req_text,
            "reference": f"{section_heading}, page {page_number}",
            "section_heading": section_heading,
        }
        if "actor" in col_map and len(cells) > col_map["actor"]:
            kwargs["actor"] = cells[col_map["actor"]]
        if "timeframe" in col_map and len(cells) > col_map["timeframe"]:
            kwargs["timeframe"] = cells[col_map["timeframe"]]
        if "evidence_needed" in col_map and len(cells) > col_map["evidence_needed"]:
            kwargs["evidence_needed"] = cells[col_map["evidence_needed"]]

        try:
            requirements.append(ComplianceRequirement(**kwargs))
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning("Skipping malformed table row requirement: %s", exc)

    return requirements


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------


async def run_compliance_extractor(pdf_path: str) -> list[Requirement]:
    """
    Full extraction pipeline for long regulatory documents.

    Steps:
    1. Parse PDF with structure preservation
    2. Segment into sections
    3. Filter to obligation-bearing sections
    4. Extract requirements per section (parallelized)
    5. Deduplicate
    6. Assemble hierarchy
    7. Assign IDs and return

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        List of :class:`~backend.models.schemas.Requirement` (actually
        :class:`~backend.models.schemas.ComplianceRequirement`) objects.
    """
    # Step 1: Structured parse
    logger.info("Compliance extractor: parsing %s with structure preservation.", pdf_path)
    structured_text = parse_pdf_with_structure(pdf_path)
    if not structured_text.strip():
        logger.warning("Compliance extractor: no text extracted from %s.", pdf_path)
        return []

    # Step 2: Segment
    sections = segment_document(structured_text)
    logger.info("Compliance extractor: %d sections found.", len(sections))

    # Step 3: Filter
    obligation_sections, skipped = filter_obligation_sections(sections)
    logger.info(
        "Compliance extractor: kept %d/%d sections with obligation language (skipped %d).",
        len(obligation_sections),
        len(sections),
        len(skipped),
    )

    if not obligation_sections:
        logger.warning("Compliance extractor: no obligation sections found in %s.", pdf_path)
        return []

    # Step 4: Parallel per-section extraction
    raw_requirements = await extract_from_all_sections(obligation_sections)
    logger.info("Compliance extractor: extracted %d raw requirements.", len(raw_requirements))

    if not raw_requirements:
        logger.warning(
            "Compliance extractor: zero requirements extracted from %s.", pdf_path
        )
        return []

    # Step 5: Deduplicate
    deduped = deduplicate_requirements(raw_requirements)
    logger.info(
        "Compliance extractor: deduplicated %d \u2192 %d requirements.",
        len(raw_requirements),
        len(deduped),
    )

    # Step 6: Hierarchy
    with_hierarchy = assemble_hierarchy(deduped, sections)

    # Step 7: Finalize
    return finalize_requirements(with_hierarchy)
