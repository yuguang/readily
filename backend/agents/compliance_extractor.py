"""
Compliance Extraction Agent — Component 8.

A specialized extraction pipeline for long regulatory PDFs (>20 pages).

Pipeline:
  1. Structured PDF parse with heading/table markers
  2. Section segmentation
  3. Term definition extraction (rule-based; persists to ``document_terms``)
  4. Obligation-language filtering
  5. Per-section LLM extraction (parallelized)
  6. Embedding-based deduplication
  7. Hierarchy assembly
  8. ID assignment and final output

Entry point: :func:`run_compliance_extractor`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

import fitz  # PyMuPDF

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    RetryError,
)

import numpy as np
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from smolagents import OpenAIModel, Tool, ToolCallingAgent

from backend.config import (
    COMPLIANCE_EMBED_WORKERS,
    COMPLIANCE_HTTP_TIMEOUT_SECONDS,
    COMPLIANCE_LLM_API_BASE,
    COMPLIANCE_LLM_MODEL_ID,
    COMPLIANCE_LLM_RPM,
    DEDUP_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    GEMINI_API_BASE,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    COMPLIANCE_MAX_CONCURRENT_WORKERS,
    SECTION_BATCH_MAX_CHARS,
    SECTION_BATCH_MAX_COUNT,
    WORKER_TIMEOUT_SECONDS,
)
from backend.models.schemas import ComplianceRequirement, Requirement
from backend.tools import extraction_cache
from backend.tools.nested_list_parser import ParseNestedListTool
from backend.tools.cross_reference_resolver import (
    SectionIndex,
    build_section_index,
)
from backend.tools.document_segmenter import (
    DocumentSection,
    filter_obligation_sections,
    segment_document,
)
from backend.tools.rate_limiter import (
    acquire_global,
    configure_global_bucket,
    get_global_bucket,
)
from backend.tools.term_extractor import extract_term_definitions, upsert_term_definitions
from backend.tools.header_footer_stripper import (
    HeaderFooterFilter,
    _normalize_signature,
    build_header_footer_filter,
    build_hf_filter_from_pdf,
)

# Prompt-version tag baked into cache keys.  Bump whenever the per-section
# extraction prompt or schema changes so stale cached results are evicted.
EXTRACTION_PROMPT_VERSION = "v2"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured PDF parsing (used by the compliance pipeline)
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


def parse_pdf_with_structure(
    pdf_path: str,
    hf_filter: HeaderFooterFilter | None = None,
) -> str:
    """
    Parse a PDF preserving headings, tables, and page boundaries.

    Uses PyMuPDF block-level extraction to detect heading levels from font
    sizes and ``find_tables()`` to extract tables as pipe-delimited text.
    Running headers, footers, and page numbers are stripped automatically
    via :func:`~backend.tools.header_footer_stripper.build_header_footer_filter`.

    Args:
        pdf_path:  Path to the PDF file.
        hf_filter: Pre-built :class:`~backend.tools.header_footer_stripper.HeaderFooterFilter`.
                   When ``None`` (default) the filter is built automatically
                   from the document.  Pass an empty ``HeaderFooterFilter()``
                   to disable stripping entirely.

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
        # --- Pass 1: font-size survey + header/footer detection ---------------
        font_size_counts = _collect_font_sizes(doc)
        heading_map = _build_heading_size_map(font_size_counts)

        if hf_filter is None:
            hf_filter = build_header_footer_filter(doc)
        stripped_count = 0

        parts: list[str] = []

        # --- Pass 2: build structured text page by page -----------------------
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1
            page_height = page.rect.height
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
                # Skip header/footer blocks
                if hf_filter.is_header_footer(block, page_height):
                    stripped_count += 1
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

        if stripped_count:
            logger.debug(
                "Stripped %d header/footer blocks from %s.",
                stripped_count,
                pdf_path,
            )
        return "\n".join(parts)
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Smolagents extraction tools
# ---------------------------------------------------------------------------
# Three tools are given to the ToolCallingAgent for each section:
#
#   get_section_text          — no-input, returns raw section text + tables
#   strip_headers_footers     — text → text, removes running header/footer lines
#   resolve_cross_reference   — section_id → text, looks up a referenced section
#
# The agent is free to call them in any order; the instructions guide it to
# follow: get → strip → resolve (per ref) → final_answer.
# ---------------------------------------------------------------------------


class GetSectionTextTool(Tool):
    """
    No-input smolagents tool that returns the raw text of the document section
    currently being processed, including any embedded tables.

    Instantiated once per extraction call (closure over ``section``).
    Always call this first; the text may contain running header/footer
    fragments — clean those with ``strip_headers_footers`` before analysis.
    """

    name = "get_section_text"
    description = (
        "Returns the raw text of the document section to analyze, including "
        "tables. Call this first. The text may contain running header/footer "
        "noise \u2014 pass it to strip_headers_footers before extracting requirements."
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


class StripHeadersFootersTool(Tool):
    """
    Smolagents tool that removes running headers, footers, and page numbers
    from raw section text.

    The filter is pre-built from the full document (in
    :func:`run_compliance_extractor_with_progress`) and passed as
    ``repeated_signatures`` so that all worker threads share the same
    read-only set without re-opening the PDF.

    Uses line-level normalization: a line whose normalized form
    (lowercase, digits → ``#``) matches a known header/footer signature is
    dropped.  Lines that are structurally part of the section (body text,
    table rows, heading markers) are preserved.
    """

    name = "strip_headers_footers"
    description = (
        "Remove running headers, footers, and page numbers from raw section text. "
        "Call this on the text returned by get_section_text before extracting "
        "requirements. Returns the cleaned text."
    )
    inputs = {
        "text": {
            "type": "string",
            "description": "Raw section text returned by get_section_text.",
        }
    }
    output_type = "string"

    def __init__(self, repeated_signatures: frozenset[str]) -> None:
        super().__init__()
        self._sigs = repeated_signatures

    def forward(self, text: str) -> str:  # type: ignore[override]
        if not self._sigs:
            return text  # no filter data — pass through unchanged
        cleaned: list[str] = []
        for line in text.splitlines():
            sig = _normalize_signature(line.strip())
            if sig and sig in self._sigs:
                continue  # drop — this line is a header/footer artifact
            cleaned.append(line)
        return "\n".join(cleaned)


class ResolveCrossReferenceTool(Tool):
    """
    Smolagents tool that looks up an internal cross-reference by section
    identifier and returns that section’s content.

    Call this when the (cleaned) section text contains phrases like
    ``\u201csee Section III.A\u201d``, ``\u201cas defined in Appendix B\u201d``, or
    ``\u201cpursuant to Section 4.2\u201d``.  The LLM can then use the returned text
    to understand conditional requirements that span multiple sections.

    The :class:`~backend.tools.cross_reference_resolver.SectionIndex` is
    pre-populated from all document segments before extraction begins.
    """

    name = "resolve_cross_reference"
    description = (
        "Look up a cross-referenced section by identifier and return its text. "
        "Use when the section contains phrases like \u2018see Section III.A\u2019, "
        "\u2018as defined in Appendix B\u2019, or \u2018per Section 4.2\u2019. "
        "Returns the referenced section content, or a not-found message."
    )
    inputs = {
        "section_id": {
            "type": "string",
            "description": (
                "Section identifier to look up, e.g. \u2018III.A\u2019, \u2018Appendix B\u2019, \u20184.2\u2019, "
                "or a full heading like \u2018III.A Initial Credentialing\u2019."
            ),
        }
    }
    output_type = "string"

    def __init__(self, section_index: SectionIndex) -> None:
        super().__init__()
        self._index = section_index

    def forward(self, section_id: str) -> str:  # type: ignore[override]
        if not section_id or not section_id.strip():
            return "Error: section_id must be a non-empty string."
        result = self._index.resolve(section_id.strip())
        if result is None:
            return f"No section found for identifier: {section_id!r}"
        heading, summary = result
        return f"Section: {heading}\n\n{summary}"


class BatchResolveCrossReferenceTool(Tool):
    """
    Batched variant of :class:`ResolveCrossReferenceTool`.

    Accepts a list of section identifiers and returns a single formatted
    string containing every resolved section.  Using this tool once per
    extraction saves N−1 LLM "tool call" round-trips compared to calling
    ``resolve_cross_reference`` individually for each reference.
    """

    name = "resolve_cross_references"
    description = (
        "Look up MULTIPLE cross-referenced sections at once and return their "
        "text joined together. Prefer this tool over resolve_cross_reference "
        "when the section contains two or more references (e.g. 'see Section "
        "III.A and Appendix B'): pass the full list in one call."
    )
    inputs = {
        "section_ids": {
            "type": "array",
            "description": (
                "List of section identifiers to look up, e.g. "
                "['III.A', 'Appendix B', '4.2']."
            ),
            "items": {"type": "string"},
        }
    }
    output_type = "string"

    def __init__(self, section_index: SectionIndex) -> None:
        super().__init__()
        self._index = section_index

    def forward(self, section_ids: list[str]) -> str:  # type: ignore[override]
        if not section_ids:
            return "Error: section_ids must be a non-empty list."
        blocks: list[str] = []
        seen: set[str] = set()
        for raw in section_ids:
            if not isinstance(raw, str):
                continue
            sid = raw.strip()
            if not sid or sid.lower() in seen:
                continue
            seen.add(sid.lower())
            result = self._index.resolve(sid)
            if result is None:
                blocks.append(f"[{sid}] No section found.")
                continue
            heading, summary = result
            blocks.append(f"[{sid}] Section: {heading}\n{summary}")
        if not blocks:
            return "No valid section identifiers provided."
        return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Section extraction prompt
# ---------------------------------------------------------------------------

_SECTION_EXTRACTION_INSTRUCTIONS = """\
You are a compliance requirement extractor. You have FOUR tools to help you
extract every obligation from a single regulatory document section.

AVAILABLE TOOLS:
  get_section_text          — (no input) returns the raw section text + tables.
  strip_headers_footers     — (text) removes running headers, footers, and page
                              numbers from the raw text. Always call this before
                              analysis so noise does not pollute your extraction.
  parse_nested_list         — (text) parses nested enumerations (1 → a → i → A)
                              into a JSON tree. Each node has 'marker', 'text',
                              'level', 'path', and 'children' keys. Use this when
                              the cleaned text contains deeply nested numbered or
                              lettered lists so you correctly attribute each
                              sub-item to its parent obligation. Returns '[]' if
                              no list structure is found (safe to call always).
  resolve_cross_references  — (section_ids) returns the bodies of MULTIPLE
                              referenced sections in a single call. PREFER this
                              tool when the section contains two or more
                              cross-references (e.g. 'III.A', 'Appendix B',
                              '4.2'). Call once with the full list instead of
                              once per reference.
  resolve_cross_reference   — (section_id) single-reference variant. Use only
                              when exactly one reference is present.

WORKFLOW:
  1. Call get_section_text to retrieve the raw text.
  2. Call strip_headers_footers(text=<raw text>) to clean it.
  2.5 (optional) If the cleaned text contains nested numbered or lettered lists,
      call parse_nested_list(text=<cleaned text>) to get a JSON tree. Walk the
      tree level-by-level: each parent node provides the obligation context and
      its children give the specific sub-requirements. Extract one requirement
      per leaf (or branch) rather than treating sub-items as flat top-level
      obligations.
  3. Scan the cleaned text for internal cross-references
     ("see Section X", "as defined in Appendix Y", "per Section 4.2", etc.).
     Collect ALL distinct referenced section ids into a single list and call
     resolve_cross_references(section_ids=[X, Y, Z]) exactly once to obtain
     their combined context.  Fall back to resolve_cross_reference(section_id=X)
     only when exactly one reference is present. Use the returned context only
     to clarify conditional logic — do not extract requirements FROM the
     referenced section.
  4. Extract all compliance obligations from the CLEANED text (step 2 output).
  5. Return a JSON array via final_answer.

WHAT TO EXTRACT:
- must, shall, required to, is responsible for, prohibited, may not,
  should (if normative), within X days, annually, quarterly, upon request
- Requirements embedded in tables (each row may be a separate requirement)
- Conditional requirements (“if X, then must Y”)

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
- Capture conditional logic: “if X, then Y” → condition=“if X”, action=“Y”.
- Tables: treat each row as a potential separate requirement.
- If a statement is ambiguous between mandatory and recommended, label it
  “conditional_mandatory” and note the ambiguity in the condition field.

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
# Retry helpers for transient API errors (503 / 429)
# ---------------------------------------------------------------------------

# Maximum number of attempts (1 initial + 4 retries → waits of ~4, 8, 16, 32 s
# plus jitter, capped at 60 s, so the worst-case extra wait is ~120 s total).
_MAX_ATTEMPTS = 5

# Keywords that identify a transient Gemini API error worth retrying.
# 429 RESOURCE_EXHAUSTED = rate limit hit.
# 503 UNAVAILABLE        = high-demand overload (the error seen in logs).
_RETRYABLE_SUBSTRINGS: tuple[str, ...] = (
    "503",
    "429",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "high demand",
    "rate limit",
    "quota",
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True when *exc* looks like a transient Gemini API error."""
    msg = str(exc).upper()
    return any(s.upper() in msg for s in _RETRYABLE_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Per-section LLM extraction
# ---------------------------------------------------------------------------


def _make_model() -> OpenAIModel:
    """Build the LLM model for compliance extraction.

    For Gemini models (``COMPLIANCE_LLM_MODEL_ID`` starting with ``"gemini/"``)
    the model is routed through the Gemini OpenAI-compatible endpoint using
    :class:`~smolagents.OpenAIModel`.  This avoids the Pydantic v2
    serialization warnings produced by :class:`~smolagents.LiteLLMModel`.

    For all other models, :class:`~smolagents.OpenAIModel` is used with the
    OpenAI API (or a custom ``COMPLIANCE_LLM_API_BASE`` endpoint).

    The underlying OpenAI client is configured with a
    ``COMPLIANCE_HTTP_TIMEOUT_SECONDS`` per-request HTTP timeout so a network
    stall surfaces as a retryable exception instead of blocking for the
    SDK's 600s default.
    """
    # Per-request HTTP timeout is passed through smolagents' ``client_kwargs``
    # escape hatch and ultimately reaches the OpenAI client constructor.
    client_kwargs: dict[str, Any] = {"timeout": COMPLIANCE_HTTP_TIMEOUT_SECONDS}

    if COMPLIANCE_LLM_MODEL_ID.startswith("gemini/"):
        # Strip the litellm-style "gemini/" provider prefix — OpenAIModel
        # talks directly to the Gemini OpenAI-compatible REST endpoint.
        model_id = COMPLIANCE_LLM_MODEL_ID.removeprefix("gemini/")
        return OpenAIModel(
            model_id=model_id,
            api_key=GEMINI_API_KEY,
            api_base=GEMINI_API_BASE,
            temperature=0.2,
            client_kwargs=client_kwargs,
        )

    # Default: OpenAI (or a custom OpenAI-compatible endpoint)
    kwargs: dict[str, Any] = {
        "model_id": COMPLIANCE_LLM_MODEL_ID,
        "api_key": OPENAI_API_KEY,
        "temperature": 0.2,
        "client_kwargs": client_kwargs,
    }
    if COMPLIANCE_LLM_API_BASE:
        kwargs["api_base"] = COMPLIANCE_LLM_API_BASE
    return OpenAIModel(**kwargs)


def run_section_extractor(
    section: DocumentSection,
    section_index: SectionIndex | None = None,
    hf_filter: HeaderFooterFilter | None = None,
) -> list[ComplianceRequirement]:
    """
    Run the per-section :class:`~smolagents.ToolCallingAgent` for one
    :class:`DocumentSection`.

    The agent receives three tools:

    * ``get_section_text`` — returns raw section text and tables.
    * ``strip_headers_footers`` — cleans running header/footer noise from text.
    * ``resolve_cross_reference`` — looks up cross-referenced sections by ID.

    Args:
        section:       The section to extract from.
        section_index: Pre-built cross-reference index for the full document.
                       If ``None``, an empty index is used and cross-reference
                       lookups will always return “not found”.
        hf_filter:     Pre-built header/footer filter for the full document.
                       If ``None``, the strip tool is a no-op pass-through.

    Returns an empty list (with a logged warning) if the agent fails or
    returns malformed JSON — never raises.
    """
    resolved_index = section_index or SectionIndex()
    get_text_tool       = GetSectionTextTool(section)
    strip_tool          = StripHeadersFootersTool(
        hf_filter.repeated_signatures if hf_filter else frozenset()
    )
    resolve_tool        = ResolveCrossReferenceTool(resolved_index)
    batch_resolve_tool  = BatchResolveCrossReferenceTool(resolved_index)
    parse_list_tool     = ParseNestedListTool()

    agent = ToolCallingAgent(
        tools=[get_text_tool, strip_tool, parse_list_tool, batch_resolve_tool, resolve_tool],
        model=_make_model(),
        # One batched resolve call replaces up to ~6 individual lookups, so
        # the typical upper bound drops from ~14 to ~9 steps.
        max_steps=10,
        name="section_requirement_extractor",
        description="Extracts compliance requirements from one document section.",
        instructions=_SECTION_EXTRACTION_INSTRUCTIONS,
        verbosity_level=0,
    )

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential_jitter(initial=4, max=60),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,   # wrap final failure in RetryError, handled below
    )
    def _call_agent() -> Any:
        return agent.run(
            f"Extract all compliance requirements from section '{section.heading}'. "
            "Follow the workflow: get_section_text → strip_headers_footers → "
            "resolve_cross_reference (for each cross-reference found) → "
            "final_answer with a JSON array of requirements."
        )

    try:
        result = _call_agent()
        return _parse_compliance_requirements(result, section)
    except RetryError as exc:
        # All _MAX_ATTEMPTS retryable failures exhausted
        cause = exc.last_attempt.exception()
        logger.warning(
            "Section %r: gave up after %d attempts (%s). Marking for human review.",
            section.heading,
            _MAX_ATTEMPTS,
            cause,
        )
        return []
    except Exception as exc:
        # Non-retryable error (bad JSON, agent logic failure, etc.)
        logger.warning(
            "Section %r: extraction failed: %s. Marking for human review.",
            section.heading,
            exc,
        )
        return []


# ---------------------------------------------------------------------------
# Parallelized extraction across all sections
# ---------------------------------------------------------------------------


def _section_cache_key(section: DocumentSection) -> str:
    """Hash used to key per-section extraction results in the artifact cache.

    Combines the section heading + full body + tables + the current prompt
    version so a prompt change invalidates stale results automatically.
    """
    body = section.text + "\n".join(section.tables)
    return extraction_cache.hash_text(
        body,
        section.heading,
        str(section.level),
        EXTRACTION_PROMPT_VERSION,
    )


def _cached_requirements_to_models(
    cached: list[dict[str, Any]],
    section: DocumentSection,
) -> list[ComplianceRequirement]:
    """Revive cached requirement dicts back into :class:`ComplianceRequirement`."""
    out: list[ComplianceRequirement] = []
    for item in cached:
        try:
            out.append(ComplianceRequirement.model_validate(item))
        except (ValidationError, TypeError) as exc:
            logger.warning(
                "Section %r: dropping malformed cached requirement: %s",
                section.heading, exc,
            )
    return out


async def _run_section_with_timeout(
    section: DocumentSection,
    section_index: SectionIndex | None = None,
    hf_filter: HeaderFooterFilter | None = None,
    timeout_seconds: float = WORKER_TIMEOUT_SECONDS,
    pdf_hash: str | None = None,
) -> list[ComplianceRequirement]:
    """
    Run :func:`run_section_extractor` in a worker thread with a wall-clock timeout.

    If the LLM call (or any tool call) blocks longer than *timeout_seconds*,
    the surrounding ``asyncio`` task is released with an empty result so the
    overall pipeline can continue instead of hanging forever on a single
    stuck section.  The underlying thread may still be running — Python cannot
    forcibly kill threads blocked in C extensions / network I/O — but it no
    longer blocks ``asyncio.as_completed`` from advancing.

    Before dispatching the worker we (a) check the on-disk extraction cache
    keyed by ``(pdf_hash, section_hash, prompt_version)`` — a hit short-
    circuits the LLM call entirely — and (b) acquire one token from the
    global RPM bucket so concurrent workers respect provider rate limits.

    Args:
        section:         The section to extract from.
        section_index:   Cross-reference index for the full document.
        hf_filter:       Header/footer filter for the full document.
        timeout_seconds: Wall-clock timeout per section.  Defaults to
                         :data:`~backend.config.WORKER_TIMEOUT_SECONDS`.
        pdf_hash:        Optional SHA-256 of the source PDF for cache keying.
                         When ``None`` the cache is bypassed.

    Returns:
        Extracted requirements, or ``[]`` on timeout / thread error.
    """
    # Cache fast path: identical (PDF, section body, prompt) ⇒ reuse results.
    section_hash: str | None = None
    if pdf_hash:
        section_hash = _section_cache_key(section)
        cached = extraction_cache.get_section_requirements(pdf_hash, section_hash)
        if cached is not None:
            logger.debug(
                "Section %r: cache hit (%d requirements).",
                section.heading, len(cached),
            )
            return _cached_requirements_to_models(cached, section)

    # Respect the global RPM bucket before we spend an LLM call.
    await acquire_global(1)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_section_extractor, section, section_index, hf_filter
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Section %r: extraction timed out after %ss; skipping.",
            section.heading,
            timeout_seconds,
        )
        return []

    # Write-through cache for successful extractions (including empty lists,
    # which represent a legitimate "no obligations here" outcome).
    if pdf_hash and section_hash:
        extraction_cache.put_section_requirements(
            pdf_hash,
            section_hash,
            [r.model_dump() for r in result],
        )
    return result


async def extract_from_all_sections(
    sections: list[DocumentSection],
    section_index: SectionIndex | None = None,
    hf_filter: HeaderFooterFilter | None = None,
    pdf_hash: str | None = None,
) -> list[ComplianceRequirement]:
    """
    Extract compliance requirements from all sections concurrently.

    Uses a semaphore to cap parallelism at ``COMPLIANCE_MAX_CONCURRENT_WORKERS``
    and wraps each worker in :func:`_run_section_with_timeout` so a single
    stuck LLM call cannot stall the whole pipeline.
    Each section runs in a thread via :func:`asyncio.to_thread`.

    Args:
        sections:      Obligation-bearing sections from the filtering step.
        section_index: Pre-built cross-reference index for the full document.
        hf_filter:     Pre-built header/footer filter for the full document.
        pdf_hash:      Optional SHA-256 of the source PDF for cache keying.

    Returns:
        Flat list of all extracted :class:`ComplianceRequirement` objects.
    """
    semaphore = asyncio.Semaphore(COMPLIANCE_MAX_CONCURRENT_WORKERS)

    async def extract_one(section: DocumentSection) -> list[ComplianceRequirement]:
        async with semaphore:
            return await _run_section_with_timeout(
                section, section_index, hf_filter, WORKER_TIMEOUT_SECONDS,
                pdf_hash=pdf_hash,
            )

    tasks = [asyncio.create_task(extract_one(s)) for s in sections]
    all_reqs: list[ComplianceRequirement] = []
    for coro in asyncio.as_completed(tasks):
        section_reqs = await coro
        all_reqs.extend(section_reqs)

    return all_reqs


# ---------------------------------------------------------------------------
# Batched small-section extraction (optimisation #3)
# ---------------------------------------------------------------------------
# Each call to ``run_section_extractor`` pays a full agentic round-trip
# (3–6 tool steps + final_answer).  For very small sections that cost
# dominates the actual extraction work.  ``_extract_small_sections_batch``
# packs up to ``SECTION_BATCH_MAX_COUNT`` small sections into a single
# direct LLM completion request and parses a per-section response.
#
# The batched path is purely additive: large sections continue to use the
# agentic pipeline, and batched output is funneled through the same
# ``_parse_compliance_requirements`` so validation rules are identical.
# ---------------------------------------------------------------------------


_BATCH_EXTRACTION_PROMPT = """\
You are a compliance requirement extractor.  You will be given MULTIPLE short
document sections delimited by [SECTION_START ...]/[SECTION_END] markers.
For EACH section, extract every compliance obligation.

Return ONE JSON object with this exact shape:

  {
    "sections": [
      {
        "heading": "<the section's exact heading>",
        "requirements": [ <requirement objects ...> ]
      },
      ...
    ]
  }

Each requirement object follows this schema:
  {
    "text":              "Does the P&P state that ...?",
    "exact_quote":       "verbatim text from the section",
    "reference":         "<section heading>, page <N>",
    "category":          "topic label",
    "obligation_type":   "mandatory | prohibition | conditional | recommended",
    "obligation_level":  "mandatory | conditional_mandatory | recommended | informational",
    "actor":             "who must act",
    "action_required":   "what must be done",
    "condition":         "trigger condition, or null",
    "timeframe":         "deadline/frequency, or null",
    "evidence_needed":   "what proves compliance, or null",
    "risk_area":         "Privacy | Security | Financial | Operations | Clinical | Administrative"
  }

If a section has no obligations return an empty requirements array for it.
Return ONLY the JSON object.  No prose, no markdown fences.
"""


def _build_batch_prompt(sections: list[DocumentSection]) -> str:
    """Concatenate small sections into a single batch-extraction user prompt."""
    parts: list[str] = [_BATCH_EXTRACTION_PROMPT, ""]
    for sec in sections:
        parts.append(
            f"[SECTION_START heading={sec.heading!r} page={sec.page_start}]"
        )
        if sec.text.strip():
            parts.append(sec.text.strip())
        for i, table in enumerate(sec.tables, start=1):
            parts.append(f"[TABLE {i}]\n{table}\n[/TABLE {i}]")
        parts.append("[SECTION_END]")
        parts.append("")
    return "\n".join(parts)


def _parse_batch_response(
    raw: str,
    sections: list[DocumentSection],
) -> list[ComplianceRequirement]:
    """Parse a batched-extraction JSON response back into per-section requirements."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Batch extraction: non-JSON response (%s). Raw (200): %s",
            exc, raw[:200],
        )
        return []

    heading_to_section = {s.heading: s for s in sections}
    all_reqs: list[ComplianceRequirement] = []
    for entry in payload.get("sections") or []:
        if not isinstance(entry, dict):
            continue
        heading = str(entry.get("heading") or "").strip()
        section = heading_to_section.get(heading)
        if section is None:
            # Soft-match: accept closest prefix when model slightly rewrites
            # the heading.
            for h, s in heading_to_section.items():
                if heading and (heading in h or h in heading):
                    section = s
                    break
        if section is None:
            logger.debug("Batch extraction: unknown heading %r; skipping.", heading)
            continue
        items = entry.get("requirements") or []
        all_reqs.extend(_parse_compliance_requirements(items, section))
    return all_reqs


def _run_batch_extraction_sync(
    sections: list[DocumentSection],
) -> list[ComplianceRequirement]:
    """Blocking worker: one LLM call → requirements for all *sections*.

    Uses the ``OpenAIModel`` client directly (bypassing the ToolCallingAgent)
    to save the ~3–6 tool-call round-trips the agent would otherwise perform.
    Pre-strips header/footer noise so the prompt stays small.
    """
    if not sections:
        return []
    model = _make_model()
    prompt = _build_batch_prompt(sections)
    try:
        # ``OpenAIModel`` exposes the underlying ``openai`` client as ``.client``.
        response = model.client.chat.completions.create(  # type: ignore[attr-defined]
            model=model.model_id,  # type: ignore[attr-defined]
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(
            "Batch extraction: LLM call failed (%d sections): %s",
            len(sections), exc,
        )
        return []
    return _parse_batch_response(raw, sections)


async def _run_batch_with_timeout(
    sections: list[DocumentSection],
    timeout_seconds: float = WORKER_TIMEOUT_SECONDS,
    pdf_hash: str | None = None,
) -> list[ComplianceRequirement]:
    """Async wrapper: cache-aware, rate-limited batched small-section extraction."""
    if not sections:
        return []

    # Cache lookup: combined (pdf_hash, concatenated section hashes) keys the batch.
    batch_hash: str | None = None
    if pdf_hash:
        parts = [_section_cache_key(s) for s in sections]
        batch_hash = extraction_cache.hash_text("|".join(parts), "batch", EXTRACTION_PROMPT_VERSION)
        cached = extraction_cache.get_section_requirements(pdf_hash, batch_hash)
        if cached is not None:
            # Route cached dicts through per-section validation to preserve behaviour.
            synthetic = sections[0]
            return _cached_requirements_to_models(cached, synthetic)

    await acquire_global(1)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_run_batch_extraction_sync, sections),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Batch extraction timed out after %ss (%d sections); falling back to empty.",
            timeout_seconds, len(sections),
        )
        return []

    if pdf_hash and batch_hash:
        extraction_cache.put_section_requirements(
            pdf_hash, batch_hash, [r.model_dump() for r in result],
        )
    return result


def partition_sections_for_batching(
    sections: list[DocumentSection],
    max_chars: int = SECTION_BATCH_MAX_CHARS,
    max_count: int = SECTION_BATCH_MAX_COUNT,
) -> tuple[list[list[DocumentSection]], list[DocumentSection]]:
    """Split *sections* into (small_batches, large_sections).

    - Sections whose ``char_count`` is below *max_chars* are grouped into
      batches of up to *max_count* elements, subject to a combined-char
      budget of ``max_chars * max_count``.
    - Sections >= *max_chars* keep the agentic per-section path.
    - When *max_chars* <= 0 batching is disabled and every section is routed
      to the large-section path.
    """
    if max_chars <= 0 or max_count <= 1:
        return [], list(sections)

    small_batches: list[list[DocumentSection]] = []
    large: list[DocumentSection] = []
    current: list[DocumentSection] = []
    current_chars = 0
    char_budget = max_chars * max_count

    for sec in sections:
        if sec.char_count >= max_chars or sec.tables:
            # Tables can be long and include structured data — let the
            # agentic extractor handle them directly.
            large.append(sec)
            continue
        if (
            len(current) >= max_count
            or current_chars + sec.char_count > char_budget
        ):
            if current:
                small_batches.append(current)
            current = [sec]
            current_chars = sec.char_count
        else:
            current.append(sec)
            current_chars += sec.char_count

    if current:
        small_batches.append(current)
    return small_batches, large


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


def _encode_texts(texts: list[str]) -> np.ndarray:
    """Top-level helper used by :class:`ProcessPoolExecutor` workers.

    Top-level so that the function is picklable — nested / lambda helpers
    cannot cross the process boundary.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _encode_with_process_pool(
    texts: list[str],
    max_workers: int | None,
) -> np.ndarray:
    """Encode *texts* using a :class:`ProcessPoolExecutor`.

    Falls back to in-process encoding when the pool fails to start (e.g. in
    restricted environments).  Splits the workload into roughly-equal shards
    so each worker only loads the SentenceTransformer model once.
    """
    import concurrent.futures as _cf

    n = len(texts)
    workers = max_workers if max_workers and max_workers > 0 else min(4, max(1, n // 32 or 1))
    if workers <= 1 or n < 16:
        # Process-pool start-up is ~1s; only worth it for non-trivial batches.
        return _encode_texts(texts)

    # Even shards, last one absorbs the remainder.
    shard_size = (n + workers - 1) // workers
    shards = [texts[i : i + shard_size] for i in range(0, n, shard_size)]
    try:
        with _cf.ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_encode_texts, shards))
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "Dedup: ProcessPoolExecutor failed (%s); falling back to in-process encoding.",
            exc,
        )
        return _encode_texts(texts)
    return np.vstack(results)


def deduplicate_requirements(
    requirements: list[ComplianceRequirement],
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    *,
    parallel_workers: int | None = None,
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
        parallel_workers: If > 1, encode embeddings using a process pool of
                          this size.  ``None`` defaults to
                          :data:`~backend.config.COMPLIANCE_EMBED_WORKERS`.
                          Set to 1 to force in-process encoding.

    Returns:
        Deduplicated list of :class:`ComplianceRequirement` objects.
    """
    if len(requirements) <= 1:
        return requirements

    texts = [r.text for r in requirements]
    resolved_workers = parallel_workers if parallel_workers is not None else COMPLIANCE_EMBED_WORKERS
    if resolved_workers and resolved_workers > 1:
        embeddings = _encode_with_process_pool(texts, resolved_workers)
    else:
        embeddings = _encode_texts(texts)
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
# Full pipeline entry point — streaming and simple variants
# ---------------------------------------------------------------------------


async def run_compliance_extractor_with_progress(
    pdf_path: str,
) -> AsyncGenerator[dict, None]:
    """
    Streaming extraction pipeline — yields progress dicts for each step.

    Each yielded dict is one of:

    - ``{"type": "progress", "step": str, "step_number": int,
         "total_steps": 7, "detail": str, ...}``
    - ``{"type": "complete", "requirements": list[dict],
         "total_requirements": int}``

    Step 4 (extraction) additionally includes
    ``sections_completed`` and ``sections_total`` for sub-progress tracking.

    The API server forwards these as SSE events to the frontend.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Yields:
        Progress and completion events as plain ``dict`` objects.
    """
    _steps = 8

    def _prog(step: str, step_number: int, detail: str, **extra: Any) -> dict:
        return {"type": "progress", "step": step, "step_number": step_number,
                "total_steps": _steps, "detail": detail, **extra}

    # Configure the global RPM bucket up-front so every worker spawned below
    # respects the provider's rate limit.  configure_global_bucket is a no-op
    # when COMPLIANCE_LLM_RPM == 0.
    if COMPLIANCE_LLM_RPM > 0 and get_global_bucket() is None:
        configure_global_bucket(COMPLIANCE_LLM_RPM)

    # Compute a stable hash of the PDF bytes — used as the cache key for
    # structured text, header/footer signatures, segmented sections, and
    # per-section extraction results.  Falls back to ``None`` (cache
    # bypassed) if the file cannot be hashed.
    pdf_hash: str | None = None
    try:
        pdf_hash = extraction_cache.hash_pdf_bytes(pdf_path)
    except OSError as exc:
        logger.warning("Cache key unavailable for %s: %s", pdf_path, exc)

    # Step 1: Build header/footer filter then parse with structure
    # The filter is built in a separate pass so it can be shared with all
    # per-section extraction workers without re-opening the PDF.
    yield _prog("parsing", 1, "Parsing PDF structure...")
    logger.info("Compliance extractor: parsing %s with structure preservation.", pdf_path)

    cached_text = extraction_cache.get_structured_text(pdf_hash) if pdf_hash else None
    cached_sigs = extraction_cache.get_hf_signatures(pdf_hash) if pdf_hash else None
    if cached_text is not None and cached_sigs is not None:
        structured_text = cached_text
        hf_filter = HeaderFooterFilter(repeated_signatures=cached_sigs)
        logger.info("Compliance extractor: cache hit for parsed PDF (%s).", pdf_hash)
    else:
        hf_filter = build_hf_filter_from_pdf(pdf_path)
        structured_text = parse_pdf_with_structure(pdf_path, hf_filter=hf_filter)
        if pdf_hash:
            extraction_cache.put_structured_text(pdf_hash, structured_text)
            extraction_cache.put_hf_signatures(pdf_hash, hf_filter.repeated_signatures)

    if not structured_text.strip():
        logger.warning("Compliance extractor: no text extracted from %s.", pdf_path)
        yield {"type": "complete", "requirements": [], "total_requirements": 0, "term_count": 0}
        return

    # Step 2: Segment
    yield _prog("segmenting", 2, "Segmenting document...")
    cached_sections = extraction_cache.get_sections(pdf_hash) if pdf_hash else None
    if cached_sections is not None:
        sections = cached_sections
        logger.info("Compliance extractor: cache hit for %d sections.", len(sections))
    else:
        sections = segment_document(structured_text)
        if pdf_hash:
            extraction_cache.put_sections(pdf_hash, sections)
    yield _prog("segmenting", 2, f"Found {len(sections)} sections")
    logger.info("Compliance extractor: %d sections found.", len(sections))

    # Steps 3 + 4 + cross-reference index build all depend only on *sections*
    # and are independent of one another.  Run them concurrently so the
    # user-visible wall-clock for these preparatory steps equals the slowest
    # rather than the sum.
    yield _prog(
        "term_extraction", 3,
        "Running term extraction, section filtering, and cross-reference indexing...",
    )

    terms_task = asyncio.create_task(asyncio.to_thread(
        extract_term_definitions, structured_text, sections, pdf_path,
    ))
    filter_task = asyncio.create_task(asyncio.to_thread(
        filter_obligation_sections, sections,
    ))
    index_task = asyncio.create_task(asyncio.to_thread(
        build_section_index, sections,
    ))

    terms, (filter_result), section_index = await asyncio.gather(
        terms_task, filter_task, index_task,
    )
    obligation_sections, skipped = filter_result

    # Upsert terms once the term extraction completes.  Runs in a thread so it
    # doesn't block the event loop on Chroma I/O.
    await asyncio.to_thread(upsert_term_definitions, terms)

    yield _prog("term_extraction", 3, f"Indexed {len(terms)} term definitions")
    yield _prog(
        "filtering", 4,
        f"{len(obligation_sections)} of {len(sections)} sections contain obligation language",
    )
    logger.info(
        "Compliance extractor: %d terms, %d/%d obligation sections, "
        "cross-ref index with %d key variants.",
        len(terms), len(obligation_sections), len(sections), len(section_index),
    )

    if not obligation_sections:
        logger.warning("Compliance extractor: no obligation sections found in %s.", pdf_path)
        yield {"type": "complete", "requirements": [], "total_requirements": 0, "term_count": len(terms)}
        return

    # Step 5: parallel extraction.  Small obligation sections are packed
    # into batches that share a single LLM request; large sections keep the
    # agentic per-section path.  Both paths run concurrently, bounded by the
    # same semaphore so total in-flight LLM calls stays <=
    # COMPLIANCE_MAX_CONCURRENT_WORKERS.
    small_batches, large_sections = partition_sections_for_batching(obligation_sections)
    total_units = len(small_batches) + len(large_sections)
    yield _prog(
        "extracting", 5, "Extracting requirements from sections...",
        sections_completed=0, sections_total=len(obligation_sections),
    )
    logger.info(
        "Compliance extractor: extraction plan — %d batched-small units, "
        "%d large sections (%d total units).",
        len(small_batches), len(large_sections), total_units,
    )

    raw_requirements: list[ComplianceRequirement] = []
    completed_sections = 0
    semaphore = asyncio.Semaphore(COMPLIANCE_MAX_CONCURRENT_WORKERS)

    async def _extract_one(section: DocumentSection) -> tuple[int, list[ComplianceRequirement]]:
        async with semaphore:
            reqs = await _run_section_with_timeout(
                section, section_index, hf_filter, WORKER_TIMEOUT_SECONDS,
                pdf_hash=pdf_hash,
            )
        return 1, reqs

    async def _extract_batch(batch: list[DocumentSection]) -> tuple[int, list[ComplianceRequirement]]:
        async with semaphore:
            reqs = await _run_batch_with_timeout(
                batch, WORKER_TIMEOUT_SECONDS, pdf_hash=pdf_hash,
            )
        return len(batch), reqs

    tasks: list[asyncio.Task] = []
    for sec in large_sections:
        tasks.append(asyncio.create_task(_extract_one(sec)))
    for batch in small_batches:
        tasks.append(asyncio.create_task(_extract_batch(batch)))

    for coro in asyncio.as_completed(tasks):
        units_done, section_reqs = await coro
        raw_requirements.extend(section_reqs)
        completed_sections += units_done
        yield _prog(
            "extracting", 5, "Extracting requirements from sections...",
            sections_completed=completed_sections,
            sections_total=len(obligation_sections),
        )

    logger.info("Compliance extractor: extracted %d raw requirements.", len(raw_requirements))

    if not raw_requirements:
        logger.warning("Compliance extractor: zero requirements from %s.", pdf_path)
        yield {"type": "complete", "requirements": [], "total_requirements": 0, "term_count": len(terms)}
        return

    # Step 6: Deduplicate
    yield _prog("deduplicating", 6, f"Deduplicating {len(raw_requirements)} requirements...")
    deduped = deduplicate_requirements(raw_requirements)
    yield _prog(
        "deduplicating", 6,
        f"Deduplicated {len(raw_requirements)} \u2192 {len(deduped)} requirements",
    )
    logger.info(
        "Compliance extractor: deduplicated %d \u2192 %d requirements.",
        len(raw_requirements), len(deduped),
    )

    # Step 7: Hierarchy
    yield _prog("hierarchy", 7, "Assembling requirement hierarchy")
    with_hierarchy = assemble_hierarchy(deduped, sections)

    # Step 8: Finalize
    yield _prog("finalizing", 8, "Assigning IDs")
    final = finalize_requirements(with_hierarchy)

    yield {
        "type": "complete",
        "requirements": [r.model_dump() for r in final],
        "total_requirements": len(final),
        "term_count": len(terms),
    }


async def run_compliance_extractor(pdf_path: str) -> list[Requirement]:
    """
    Simple (non-streaming) entry point — collects and returns the final list.

    Delegates to :func:`run_compliance_extractor_with_progress` and discards
    all intermediate progress events, returning only the final requirements.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        List of :class:`~backend.models.schemas.ComplianceRequirement` objects.
    """
    async for event in run_compliance_extractor_with_progress(pdf_path):
        if event["type"] == "complete":
            results: list[Requirement] = []
            for r in event["requirements"]:
                try:
                    results.append(ComplianceRequirement.model_validate(r))
                except Exception:
                    results.append(Requirement.model_validate(r))
            return results
    return []
