"""
Parallel Dispatcher — fans out N requirement evaluations to concurrent workers.

Uses asyncio with a semaphore to bound concurrent LLM calls.  smolagents
agents are synchronous (blocking HTTP), so each worker runs in a thread via
``asyncio.to_thread``.  The semaphore prevents hitting API rate limits.

Public API
----------
dispatch_review   – collect all results, call optional on_result callback
stream_review     – async generator yielding SSEEvent objects (+ critic pass)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable

from backend.agents.question_agent import (
    _fallback_evaluation,
    create_question_agent,
    get_model,
    parse_agent_result,
)
from backend.config import MAX_CONCURRENT_WORKERS, WORKER_TIMEOUT_SECONDS
from backend.models.schemas import Evaluation, Requirement, SSEEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synchronous worker entry point (runs inside asyncio.to_thread)
# ---------------------------------------------------------------------------


def run_question_agent_sync(requirement: Requirement) -> Evaluation:
    """
    Blocking entry point called by each thread-pool worker.

    Creates a fresh model and agent instance per call so that no state is
    shared between concurrent threads.
    """
    model = get_model()
    agent = create_question_agent(model)
    task = (
        f"Evaluate this compliance requirement against the policy corpus:\n\n"
        f"Requirement #{requirement.id}: {requirement.text}\n"
        f"Reference: {requirement.reference or 'N/A'}"
    )
    try:
        result = agent.run(task)
    except Exception as exc:
        logger.error(
            "Agent failed for requirement %d: %s",
            requirement.id,
            exc,
        )
        return _fallback_evaluation(requirement.id, reasoning=f"Agent error: {exc}")
    return parse_agent_result(result, requirement.id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_error_evaluation(requirement_id: int, error: str) -> Evaluation:
    """Return an error evaluation with ``needs_human_review=True``."""
    return _fallback_evaluation(requirement_id, reasoning=f"Worker error: {error}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def dispatch_review(
    requirements: list[Requirement],
    on_result: Callable[[Evaluation], None] | None = None,
) -> list[Evaluation]:
    """
    Fan out requirements to parallel workers and collect results.

    Parameters
    ----------
    requirements:
        All compliance requirements to evaluate.
    on_result:
        Optional callback invoked with each :class:`~backend.models.schemas.Evaluation`
        as soon as a worker completes (useful for SSE streaming outside an
        async generator context).

    Returns
    -------
    list[Evaluation]
        Results in completion order (not necessarily requirement order).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)
    results: list[Evaluation] = []

    async def worker(req: Requirement) -> Evaluation:
        async with semaphore:
            try:
                evaluation = await asyncio.wait_for(
                    asyncio.to_thread(run_question_agent_sync, req),
                    timeout=WORKER_TIMEOUT_SECONDS,
                )
                return evaluation
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker timed out after %ds for requirement %d",
                    WORKER_TIMEOUT_SECONDS,
                    req.id,
                )
                return make_error_evaluation(req.id, f"Worker timed out after {WORKER_TIMEOUT_SECONDS}s")
            except Exception as exc:
                logger.error(
                    "Worker failed for requirement %d: %s",
                    req.id,
                    exc,
                )
                return make_error_evaluation(req.id, str(exc))

    tasks = [asyncio.create_task(worker(r)) for r in requirements]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        if on_result:
            on_result(result)

    return results


# ---------------------------------------------------------------------------
# SSE streaming generator
# ---------------------------------------------------------------------------


async def stream_review(
    requirements: list[Requirement],
) -> AsyncGenerator[SSEEvent, None]:
    """
    Async generator that yields :class:`~backend.models.schemas.SSEEvent` objects.

    Sequence of emitted events:

    1. One ``"evaluation"`` event per completed requirement (completion order).
    2. One ``"progress"`` event with ``status="running_critic"`` after all workers finish.
    3. One ``"critic_complete"`` event with the updated evaluations after the
       critic pass.

    The caller (FastAPI endpoint) converts these into HTTP SSE frames.
    """
    # Import here to avoid circular imports at module load time
    from backend.agents.critic import run_batch_critic  # noqa: PLC0415

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)
    completed = 0
    total = len(requirements)
    all_results: list[Evaluation] = []

    async def worker(req: Requirement) -> Evaluation:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(run_question_agent_sync, req),
                    timeout=WORKER_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Worker timed out after %ds for requirement %d",
                    WORKER_TIMEOUT_SECONDS,
                    req.id,
                )
                return make_error_evaluation(req.id, f"Worker timed out after {WORKER_TIMEOUT_SECONDS}s")
            except Exception as exc:
                logger.error(
                    "Stream worker failed for requirement %d: %s",
                    req.id,
                    exc,
                )
                return make_error_evaluation(req.id, str(exc))

    tasks = [asyncio.create_task(worker(r)) for r in requirements]

    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            completed += 1
            all_results.append(result)
            yield SSEEvent(
                event="evaluation",
                data={
                    "evaluation": result.model_dump(),
                    "progress": completed,
                    "total": total,
                },
            )
        except Exception as exc:  # pragma: no cover — workers should never raise
            completed += 1
            yield SSEEvent(
                event="error",
                data={"error": str(exc), "progress": completed, "total": total},
            )

    # --- Critic pass ---
    yield SSEEvent(event="progress", data={"status": "running_critic"})

    updated = await run_batch_critic(all_results, requirements)

    yield SSEEvent(
        event="critic_complete",
        data={"evaluations": [e.model_dump() for e in updated]},
    )
