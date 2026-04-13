"""
Batch Critic — reflection pass for low-confidence evaluations.

Instead of running a separate LLM call per question (which would double
the total LLM cost), the critic batches all low-confidence results into a
single prompt and lets the model review them in one shot.

Only evaluations with ``confidence < CONFIDENCE_THRESHOLD`` (default 0.7)
are reviewed.  The critic may:

- Set ``needs_human_review = True`` on items it deems uncertain.
- Override the ``answer`` if it disagrees with the agent.
- Append a short explanation to ``reasoning``.

Public API
----------
run_batch_critic   – async; returns updated list[Evaluation]
parse_critic_response  – pure function; exposed for unit testing
"""

from __future__ import annotations

import json
import logging
import re

from smolagents import OpenAIModel

from backend.config import CONFIDENCE_THRESHOLD, GEMINI_API_BASE, GEMINI_API_KEY, LLM_MODEL_ID
from backend.models.schemas import AnswerType, Evaluation, Requirement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_critic_response(response_text: str) -> list[dict]:
    """
    Extract a JSON array of critic verdicts from raw LLM output.

    The LLM is prompted to return a JSON array where each item has:
    - ``requirement_id`` (int)
    - ``verified_answer`` (str: "yes" / "no" / "partial")
    - ``needs_human_review`` (bool)
    - ``reason`` (str)

    Handles:
    - Bare JSON array.
    - Markdown-fenced JSON (```json ... ``` or ``` ... ```).
    - JSON embedded anywhere in the response (regex fallback).

    Returns an empty list when parsing fails entirely.
    """
    text = response_text.strip()

    # 1. Strip markdown code fence if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2. Try to parse the full text as JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # Might be wrapped in an object key
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    return val
    except json.JSONDecodeError:
        pass

    # 3. Regex fallback — find the first JSON array in the text
    array_match = re.search(r"\[[\s\S]*?\]", text)
    if array_match:
        try:
            data = json.loads(array_match.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning("parse_critic_response: could not extract a JSON array from response")
    return []


# ---------------------------------------------------------------------------
# Batch critic
# ---------------------------------------------------------------------------


async def run_batch_critic(
    evaluations: list[Evaluation],
    requirements: list[Requirement],
) -> list[Evaluation]:
    """
    Review low-confidence evaluations in a single batched LLM call.

    Evaluations with ``confidence >= CONFIDENCE_THRESHOLD`` are passed through
    unchanged.  The critic may update ``needs_human_review``, ``answer``, and
    ``reasoning`` on flagged items.

    Parameters
    ----------
    evaluations:
        All evaluations from the dispatcher (may include high-confidence ones).
    requirements:
        The original requirements list — used to look up requirement text.

    Returns
    -------
    list[Evaluation]
        The same list with low-confidence items potentially updated in-place.
    """
    low_conf = [e for e in evaluations if e.confidence < CONFIDENCE_THRESHOLD]

    if not low_conf:
        logger.info("Critic: no low-confidence evaluations to review.")
        return evaluations

    logger.info(
        "Critic: reviewing %d low-confidence evaluation(s) (threshold=%.2f).",
        len(low_conf),
        CONFIDENCE_THRESHOLD,
    )

    # Build review payload
    req_map = {r.id: r for r in requirements}
    review_items = []
    for ev in low_conf:
        req = req_map.get(ev.requirement_id)
        review_items.append(
            {
                "requirement_id": ev.requirement_id,
                "requirement_text": req.text if req else "(unknown)",
                "agent_answer": ev.answer.value,
                "citation": ev.citation_text,
                "source": ev.source_file,
                "confidence": ev.confidence,
            }
        )

    prompt = f"""\
You are a quality reviewer for a compliance audit. Review these {len(review_items)} \
low-confidence findings and determine if each should be flagged for human review.

For each item, respond with a JSON array (and nothing else) where each element has:
- "requirement_id": <int>
- "verified_answer": "yes" | "no" | "partial"  (your assessment)
- "needs_human_review": true | false
- "reason": "<brief explanation>"

Items to review:
{json.dumps(review_items, indent=2)}
"""

    # Single LLM call for the entire batch
    model = OpenAIModel(
        model_id=LLM_MODEL_ID,
        api_key=GEMINI_API_KEY,
        api_base=GEMINI_API_BASE,
        temperature=0.2,
    )
    response = model([{"role": "user", "content": prompt}])

    # smolagents OpenAIModel returns a ChatMessage; extract text content
    response_text: str = getattr(response, "content", "") or str(response)

    critic_results = parse_critic_response(response_text)

    # Apply verdicts back to the evaluation objects
    eval_map = {e.requirement_id: e for e in evaluations}
    for cr in critic_results:
        req_id = cr.get("requirement_id")
        ev = eval_map.get(req_id)
        if ev is None:
            logger.warning("Critic returned unknown requirement_id %r — skipping.", req_id)
            continue

        ev.needs_human_review = bool(cr.get("needs_human_review", False))

        verified = str(cr.get("verified_answer", "")).lower()
        try:
            new_answer = AnswerType(verified)
            if new_answer != ev.answer:
                ev.answer = new_answer
                reason = cr.get("reason", "")
                ev.reasoning += f" [Critic: {reason}]"
        except ValueError:
            logger.warning("Critic returned unrecognised answer %r for req %d.", verified, req_id)

    return evaluations
