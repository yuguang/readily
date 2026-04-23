"""
LLM-based extractor for narrative regulatory documents.

Used as the fallback when a document is not a structured DHCS review form.
A smolagents ToolCallingAgent reads the full document text and extracts
compliance requirements as a JSON array.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

from pydantic import ValidationError
from smolagents import OpenAIModel, Tool, ToolCallingAgent

from backend.config import GEMINI_API_BASE, GEMINI_API_KEY, LLM_MODEL_ID
from backend.models.schemas import Requirement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document text tool
# ---------------------------------------------------------------------------


class GetDocumentTextTool(Tool):
    """
    A no-input smolagents tool that returns the full text of the document
    currently being processed.  Instantiated per extraction call so the
    closure over ``text`` is safe.
    """

    name = "get_document_text"
    description = (
        "Returns the full plain text of the regulatory document to analyse. "
        "Call this first to read the document before extracting requirements."
    )
    inputs: dict = {}
    output_type = "string"

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def forward(self) -> str:  # type: ignore[override]
        return self._text


# ---------------------------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------------------------

_EXTRACTION_INSTRUCTIONS = """\
You are a healthcare compliance analyst. Your task is to extract every
compliance requirement from a regulatory document.

Steps:
1. Call get_document_text to retrieve the full document.
2. Read the document carefully and identify EVERY distinct compliance
   requirement — statements that a Managed Care Plan (MCP) MUST, SHALL,
   or is REQUIRED to do.  Also include SHOULD and EXPECTED TO statements.
3. For each requirement, produce a JSON object with these fields:
   - "id": sequential integer starting at 1
   - "text": the requirement phrased as a yes/no question starting with
     "Does the P&P state that..."
   - "reference": section/page reference in the document
     (e.g. "Section III, page 10" or "Chapter 5")
   - "category": short topic label
     (e.g. "Eligibility", "Payment", "Provider Network", "Reporting")
4. Call final_answer with a JSON array of all requirement objects.
   Return ONLY the JSON array — no markdown fences, no preamble.

Example output:
[
  {
    "id": 1,
    "text": "Does the P&P state that MCPs must enrol eligible members within 30 days?",
    "reference": "Section IV, page 12",
    "category": "Enrollment"
  }
]
"""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_requirements(raw: Any) -> list[Requirement]:
    """
    Convert the agent's ``final_answer`` into a list of :class:`Requirement`.

    Handles three formats that smolagents may return:
    - A Python list of dicts (agent called ``final_answer`` with a list).
    - A JSON string (plain or markdown-fenced).
    - Partial / malformed JSON — logs a warning and returns what can be parsed.
    """
    # Already a list — agent returned structured data directly
    if isinstance(raw, list):
        requirements = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                requirements.append(Requirement(**item))
            except (TypeError, ValueError, ValidationError) as exc:
                logger.warning("Skipping malformed requirement item %r: %s", item, exc)
        return requirements

    text = str(raw).strip()

    # Strip markdown code fences if present  (```json ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    # Extract the outermost JSON array even if there's surrounding prose
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Narrative extractor returned non-JSON output: {exc}\n"
            f"Raw (first 500 chars): {str(raw)[:500]}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array from the agent, got {type(data).__name__!r}: "
            f"{str(data)[:200]}"
        )

    requirements = []
    for item in data:
        try:
            requirements.append(Requirement(**item))
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning("Skipping malformed requirement item %r: %s", item, exc)

    return requirements


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_narrative_extractor(full_text: str) -> list[Requirement]:
    """
    Extract compliance requirements from a narrative regulatory document.

    Creates a fresh :class:`smolagents.ToolCallingAgent` backed by Gemini that
    reads the full document text via :class:`GetDocumentTextTool` and returns
    structured :class:`~backend.models.schemas.Requirement` objects.

    Args:
        full_text: Plain text of the entire document (all pages joined).

    Returns:
        List of extracted :class:`~backend.models.schemas.Requirement` objects.

    Raises:
        ValueError: If the agent output cannot be parsed into requirements.
    """
    parse_pdf_tool = GetDocumentTextTool(full_text)

    model = OpenAIModel(
        model_id=LLM_MODEL_ID,
        api_key=GEMINI_API_KEY,
        api_base=GEMINI_API_BASE,
        temperature=0.2,
    )

    agent = ToolCallingAgent(
        tools=[parse_pdf_tool],
        model=model,
        name="narrative_extractor",
        description="Extracts compliance requirements from narrative regulatory text.",
        instructions=_EXTRACTION_INSTRUCTIONS,
        verbosity_level=0,
    )

    result = agent.run(
        "Call get_document_text to read the regulatory document, then extract "
        "all compliance requirements and return them as a JSON array via final_answer."
    )

    return _parse_requirements(result)


_NARRATIVE_TOTAL_STEPS = 6


async def run_narrative_extractor_with_progress(
    full_text: str, pdf_path: str
) -> AsyncGenerator[dict, None]:
    """Async generator that runs narrative extraction and yields SSE-compatible
    progress dicts, then a final completion dict.

    Yields dicts with ``type="progress"`` during processing and
    ``type="complete"`` when finished, matching the shape produced by
    :func:`~backend.agents.compliance_extractor.run_compliance_extractor_with_progress`.
    """
    loop = asyncio.get_running_loop()

    yield {
        "type": "progress",
        "step": "reading",
        "step_number": 1,
        "total_steps": _NARRATIVE_TOTAL_STEPS,
        "detail": "Reading document",
    }
    await asyncio.sleep(0)

    yield {
        "type": "progress",
        "step": "extracting",
        "step_number": 2,
        "total_steps": _NARRATIVE_TOTAL_STEPS,
        "detail": "Extracting requirements",
    }

    requirements = await loop.run_in_executor(None, run_narrative_extractor, full_text)

    try:
        from backend.agents.compliance_extractor import parse_pdf_with_structure
        from backend.tools.document_segmenter import segment_document
        from backend.tools.term_extractor import extract_term_definitions, upsert_term_definitions

        yield {
            "type": "progress",
            "step": "parsing",
            "step_number": 3,
            "total_steps": _NARRATIVE_TOTAL_STEPS,
            "detail": "Parsing document structure",
        }
        structured_text = await loop.run_in_executor(None, parse_pdf_with_structure, pdf_path)

        yield {
            "type": "progress",
            "step": "segmenting",
            "step_number": 4,
            "total_steps": _NARRATIVE_TOTAL_STEPS,
            "detail": "Segmenting sections",
        }
        sections = await loop.run_in_executor(None, segment_document, structured_text)

        yield {
            "type": "progress",
            "step": "defining",
            "step_number": 5,
            "total_steps": _NARRATIVE_TOTAL_STEPS,
            "detail": "Extracting term definitions",
        }
        terms = await loop.run_in_executor(
            None, extract_term_definitions, structured_text, sections, pdf_path
        )

        yield {
            "type": "progress",
            "step": "saving",
            "step_number": 6,
            "total_steps": _NARRATIVE_TOTAL_STEPS,
            "detail": "Saving terms",
        }
        await loop.run_in_executor(None, upsert_term_definitions, terms)
    except Exception as exc:
        logger.warning("Narrative term extraction failed for %s: %s", pdf_path, exc)

    yield {
        "type": "complete",
        "requirements": [r.model_dump() for r in requirements],
        "total_requirements": len(requirements),
    }
