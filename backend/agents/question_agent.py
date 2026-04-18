"""
Question Agent — answers one compliance requirement via RAG.

Each call to `run_question_agent` creates a fresh ToolCallingAgent instance,
ensuring no memory or state leakage between parallel workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from smolagents import OpenAIModel, ToolCallingAgent

from backend.config import GEMINI_API_BASE, GEMINI_API_KEY, LLM_MODEL_ID
from backend.models.schemas import AnswerType, Evaluation, Requirement
from backend.tools.policy_search import define_term, evaluate_citation, search_policies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent instructions (appended after the base system prompt)
# ---------------------------------------------------------------------------

QUESTION_AGENT_PROMPT = """\
You are a healthcare compliance reviewer. You are given ONE compliance requirement
(a yes/no question about a Managed Care Plan's Policy & Procedure document).

Your job:
1. Read the requirement carefully. Identify the key regulatory concepts.
2. If the requirement text contains an acronym or program term you are not 100%
   sure about (e.g. "ECM", "POF", "MCP", "SUD", "I/DD", "TBI"), call
   `define_term` with that term BEFORE searching. Use the returned definition
   to formulate better, concept-level search queries.
3. Generate 2-3 search queries that would find the relevant policy section.
   - Use specific regulatory terms from the requirement.
   - Try variations: acronyms, full names (from the definition you looked up),
     and related concepts.
4. Call `search_policies` with each query.
5. From the returned passages, identify the one that best answers the requirement.
6. Determine: does this passage satisfy the requirement? Answer YES, NO, or PARTIAL.
7. Return your answer using `final_answer` with this exact JSON structure:
   {
     "answer": "yes" | "no" | "partial",
     "citation_text": "exact quoted text from the policy",
     "source_file": "path from the passage metadata",
     "page_number": <int>,
     "confidence": <float 0-1>,
     "reasoning": "brief explanation"
   }

IMPORTANT:
- Only cite text that ACTUALLY appears in the retrieved passages. Never fabricate.
- If no passage adequately addresses the requirement, answer "no" with low confidence.
- A "partial" answer means the policy addresses the topic but misses specific details.
- Term definitions from `define_term` are reference material ONLY — never cite them
  as the `citation_text` for a YES/NO/PARTIAL answer. Citations must come from
  `search_policies` results.\
"""


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def get_model() -> OpenAIModel:
    """Create the LLM model used by all agents.

    Uses Gemini's OpenAI-compatible endpoint so the agent can leverage
    native function-calling rather than going through LiteLLM's abstraction.
    """
    return OpenAIModel(
        model_id=LLM_MODEL_ID,
        api_key=GEMINI_API_KEY,
        api_base=GEMINI_API_BASE,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_question_agent(model: OpenAIModel) -> ToolCallingAgent:
    """Create a fresh ToolCallingAgent for evaluating one compliance requirement."""
    return ToolCallingAgent(
        tools=[search_policies, define_term, evaluate_citation],
        model=model,
        max_steps=8,
        name="question_reviewer",
        description="Reviews a single compliance requirement against the policy corpus.",
        instructions=QUESTION_AGENT_PROMPT,
    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_agent_result(result: Any, requirement_id: int) -> Evaluation:
    """
    Parse the agent's ``final_answer`` output into an :class:`Evaluation`.

    Handles three formats returned by smolagents:
    - A plain ``dict`` (when the agent calls ``final_answer`` with a dict).
    - A JSON string (when the agent stringifies the dict first).
    - A markdown-fenced JSON block (e.g. ```json\\n{...}\\n```).

    Falls back to a low-confidence "no" evaluation when parsing fails.
    """
    try:
        if isinstance(result, dict):
            data = result
        elif isinstance(result, str):
            text = result.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.splitlines()
                # Drop the opening fence line; drop the closing fence if present
                inner = lines[1:]
                if inner and inner[-1].strip() == "```":
                    inner = inner[:-1]
                text = "\n".join(inner)
            data = json.loads(text)
        else:
            raise ValueError(f"Unexpected result type: {type(result)!r}")

        return Evaluation(
            requirement_id=requirement_id,
            answer=AnswerType(data["answer"].lower()),
            citation_text=data.get("citation_text", ""),
            source_file=data.get("source_file", ""),
            page_number=int(data.get("page_number", 0)),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
        )
    except Exception as exc:
        logger.warning(
            "Failed to parse agent result for requirement %d: %s. Result: %r",
            requirement_id,
            exc,
            result,
        )
        return _fallback_evaluation(requirement_id, reasoning=f"Parse error: {exc}")


def _fallback_evaluation(requirement_id: int, reasoning: str = "") -> Evaluation:
    """Return a low-confidence 'no' evaluation used for error / timeout cases."""
    return Evaluation(
        requirement_id=requirement_id,
        answer=AnswerType.NO,
        citation_text="",
        source_file="",
        page_number=0,
        confidence=0.0,
        reasoning=reasoning or "Agent failed to produce a valid result.",
        needs_human_review=True,
    )


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 10
_BASE_DELAY = 8.0  # seconds


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is a transient 503/UNAVAILABLE error."""
    msg = str(exc)
    return "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_question_agent(
    requirement: Requirement, model: OpenAIModel
) -> Evaluation:
    """
    Run a fresh QuestionAgent for *one* requirement and return its Evaluation.

    A new agent instance is created for every call so that no conversation
    history or tool state is shared between parallel workers.

    Retries up to ``_MAX_RETRIES`` times with exponential backoff on transient
    503/UNAVAILABLE errors from the upstream model API.
    """
    task = (
        f"Evaluate this compliance requirement against the policy corpus:\n\n"
        f"Requirement #{requirement.id}: {requirement.text}\n"
        f"Reference: {requirement.reference or 'N/A'}"
    )

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        # Create a fresh agent each attempt (no state leakage between retries)
        agent = create_question_agent(model)
        try:
            result = agent.run(task)
            return parse_agent_result(result, requirement.id)
        except Exception as exc:
            if _is_retryable(exc) and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Requirement %d — transient error (attempt %d/%d), retrying in %.1fs: %s",
                    requirement.id,
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                last_exc = exc
            else:
                logger.error(
                    "Agent failed for requirement %d after %d attempt(s): %s",
                    requirement.id,
                    attempt + 1,
                    exc,
                )
                return _fallback_evaluation(
                    requirement.id,
                    reasoning=f"Agent error: {exc}",
                )

    # Exhausted all retries
    logger.error(
        "Agent failed for requirement %d after %d retries: %s",
        requirement.id,
        _MAX_RETRIES,
        last_exc,
    )
    return _fallback_evaluation(
        requirement.id,
        reasoning=f"Agent error after {_MAX_RETRIES} retries: {last_exc}",
    )
