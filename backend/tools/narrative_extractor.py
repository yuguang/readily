"""
LLM-based extractor for short narrative regulatory documents (≤20 pages).

Makes a single chat completion call to Gemini with the full document text
and parses the JSON array response into Requirement objects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

import openai
from pydantic import ValidationError

from backend.config import GEMINI_API_BASE, GEMINI_API_KEY, LLM_MODEL_ID
from backend.models.schemas import Requirement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a healthcare compliance analyst. Extract every compliance requirement
from the regulatory document provided by the user.

A requirement is any statement that a Managed Care Plan (MCP) MUST, SHALL,
or is REQUIRED to do. Also include SHOULD and EXPECTED TO statements.

For each requirement produce a JSON object with these fields:
- "id": sequential integer starting at 1
- "text": the requirement phrased as a yes/no question starting with
  "Does the P&P state that..."
- "reference": section/page reference in the document
  (e.g. "Section III, page 10" or "Chapter 5")
- "category": short topic label
  (e.g. "Eligibility", "Payment", "Provider Network", "Reporting")

Return ONLY a JSON array of requirement objects — no markdown fences, no preamble.

Example:
[
  {
    "id": 1,
    "text": "Does the P&P state that MCPs must enrol eligible members within 30 days?",
    "reference": "Section IV, page 12",
    "category": "Enrollment"
  }
]"""


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_requirements(raw: Any) -> list[Requirement]:
    """
    Parse a JSON string (or list) from the LLM into Requirement objects.

    Handles:
    - A plain JSON array string
    - A markdown-fenced JSON block (```json ... ```)
    - A JSON array embedded in surrounding prose
    - Malformed items are skipped with a warning
    """
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
            f"Expected a JSON array from the LLM, got {type(data).__name__!r}: "
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
    Extract compliance requirements from a short narrative regulatory document.

    Sends the full document text in a single chat completion request to Gemini
    and parses the JSON array response into :class:`~backend.models.schemas.Requirement`
    objects.

    Args:
        full_text: Plain text of the entire document (all pages joined).

    Returns:
        List of extracted :class:`~backend.models.schemas.Requirement` objects.

    Raises:
        ValueError: If the LLM response cannot be parsed into requirements.
        openai.OpenAIError: On API or network errors.
    """
    client = openai.OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_API_BASE)

    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": full_text},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content or ""
    return _parse_requirements(raw)


_NARRATIVE_TOTAL_STEPS = 5


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

    from backend.agents.compliance_extractor import parse_pdf_with_structure
    from backend.tools.document_segmenter import segment_document
    from backend.tools.term_extractor import extract_term_definitions, upsert_term_definitions

    yield {
        "type": "progress",
        "step": "parsing",
        "step_number": 1,
        "total_steps": _NARRATIVE_TOTAL_STEPS,
        "detail": "Parsing document structure",
    }
    structured_text = await loop.run_in_executor(None, parse_pdf_with_structure, pdf_path)

    yield {
        "type": "progress",
        "step": "extracting",
        "step_number": 2,
        "total_steps": _NARRATIVE_TOTAL_STEPS,
        "detail": "Extracting requirements",
    }

    requirements = await loop.run_in_executor(None, run_narrative_extractor, full_text)

    try:
        yield {
            "type": "progress",
            "step": "segmenting",
            "step_number": 3,
            "total_steps": _NARRATIVE_TOTAL_STEPS,
            "detail": "Segmenting sections",
        }
        sections = await loop.run_in_executor(None, segment_document, structured_text)

        yield {
            "type": "progress",
            "step": "defining",
            "step_number": 4,
            "total_steps": _NARRATIVE_TOTAL_STEPS,
            "detail": "Extracting term definitions",
        }
        terms = await loop.run_in_executor(
            None, extract_term_definitions, structured_text, sections, pdf_path
        )

        yield {
            "type": "progress",
            "step": "saving",
            "step_number": 5,
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
