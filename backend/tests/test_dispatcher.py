"""
Unit tests for backend/agents/dispatcher.py

Covers:
- run_question_agent_sync: delegates to create_question_agent + parse_agent_result
- make_error_evaluation: sets needs_human_review=True
- dispatch_review: runs all workers, calls on_result callback, handles partial
  failure, handles timeout
- stream_review: yields evaluation events in completion order, yields
  progress/critic_complete events, runs critic pass at the end
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.dispatcher import (
    dispatch_review,
    make_error_evaluation,
    run_question_agent_sync,
    stream_review,
)
from backend.models.schemas import AnswerType, Evaluation, Requirement, SSEEvent


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_requirement(id: int = 1) -> Requirement:
    return Requirement(
        id=id,
        text=f"Does the policy cover requirement {id}?",
        reference=f"APL 25-00{id}",
    )


def make_evaluation(requirement_id: int, confidence: float = 0.9) -> Evaluation:
    return Evaluation(
        requirement_id=requirement_id,
        answer=AnswerType.YES,
        citation_text="The policy states...",
        source_file="policy.pdf",
        page_number=1,
        confidence=confidence,
        reasoning="Explicit coverage found.",
    )


REQUIREMENTS = [make_requirement(i) for i in range(1, 5)]


# ---------------------------------------------------------------------------
# run_question_agent_sync
# ---------------------------------------------------------------------------


class TestRunQuestionAgentSync:
    def test_returns_evaluation_on_success(self):
        """Successful agent.run → parsed Evaluation returned."""
        valid_result = {
            "answer": "yes",
            "citation_text": "The policy covers this.",
            "source_file": "policy.pdf",
            "page_number": 2,
            "confidence": 0.85,
            "reasoning": "Explicit match.",
        }
        req = make_requirement(1)

        with (
            patch("backend.agents.dispatcher.get_model") as mock_get_model,
            patch("backend.agents.dispatcher.create_question_agent") as mock_create,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = valid_result
            mock_create.return_value = mock_agent
            mock_get_model.return_value = MagicMock()

            ev = run_question_agent_sync(req)

        assert ev.requirement_id == 1
        assert ev.answer == AnswerType.YES
        assert ev.confidence == pytest.approx(0.85)
        assert ev.needs_human_review is False

    def test_agent_exception_returns_fallback(self):
        """Agent raises → fallback evaluation with needs_human_review=True."""
        req = make_requirement(2)

        with (
            patch("backend.agents.dispatcher.get_model") as mock_get_model,
            patch("backend.agents.dispatcher.create_question_agent") as mock_create,
        ):
            mock_agent = MagicMock()
            mock_agent.run.side_effect = RuntimeError("max steps exceeded")
            mock_create.return_value = mock_agent
            mock_get_model.return_value = MagicMock()

            ev = run_question_agent_sync(req)

        assert ev.requirement_id == 2
        assert ev.answer == AnswerType.NO
        assert ev.needs_human_review is True
        assert "Agent error" in ev.reasoning

    def test_creates_fresh_model_and_agent_per_call(self):
        """Each call must create a new model + agent (no shared state)."""
        req = make_requirement(1)
        valid_result = {"answer": "yes", "confidence": 0.9}

        with (
            patch("backend.agents.dispatcher.get_model") as mock_get_model,
            patch("backend.agents.dispatcher.create_question_agent") as mock_create,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = valid_result
            mock_create.return_value = mock_agent
            mock_get_model.return_value = MagicMock()

            run_question_agent_sync(req)
            run_question_agent_sync(req)

        assert mock_get_model.call_count == 2
        assert mock_create.call_count == 2

    def test_task_contains_requirement_text(self):
        """Task string passed to agent.run includes requirement id and text."""
        req = make_requirement(3)
        valid_result = {"answer": "no", "confidence": 0.5}

        with (
            patch("backend.agents.dispatcher.get_model") as mock_get_model,
            patch("backend.agents.dispatcher.create_question_agent") as mock_create,
        ):
            mock_agent = MagicMock()
            mock_agent.run.return_value = valid_result
            mock_create.return_value = mock_agent
            mock_get_model.return_value = MagicMock()

            run_question_agent_sync(req)

        task_str = mock_agent.run.call_args[0][0]
        assert req.text in task_str
        assert str(req.id) in task_str


# ---------------------------------------------------------------------------
# make_error_evaluation
# ---------------------------------------------------------------------------


class TestMakeErrorEvaluation:
    def test_sets_needs_human_review(self):
        ev = make_error_evaluation(5, "something went wrong")
        assert ev.requirement_id == 5
        assert ev.needs_human_review is True
        assert ev.answer == AnswerType.NO
        assert ev.confidence == 0.0

    def test_includes_error_in_reasoning(self):
        ev = make_error_evaluation(7, "timeout")
        assert "timeout" in ev.reasoning


# ---------------------------------------------------------------------------
# dispatch_review
# ---------------------------------------------------------------------------


class TestDispatchReview:
    def _make_side_effect(self, evals: list[Evaluation]):
        """Return a side_effect that pops from evals in order."""
        it = iter(evals)
        return lambda req: next(it)

    def test_returns_all_results(self):
        """All requirements produce an Evaluation in the returned list."""
        reqs = [make_requirement(i) for i in range(1, 4)]
        expected = [make_evaluation(r.id) for r in reqs]

        with patch(
            "backend.agents.dispatcher.run_question_agent_sync",
            side_effect=[e for e in expected],
        ):
            results = asyncio.run(dispatch_review(reqs))

        assert len(results) == 3
        result_ids = {e.requirement_id for e in results}
        assert result_ids == {1, 2, 3}

    def test_on_result_callback_called_per_result(self):
        """on_result is invoked exactly once per completed requirement."""
        reqs = [make_requirement(i) for i in range(1, 4)]
        evals = [make_evaluation(r.id) for r in reqs]
        callback_calls: list[Evaluation] = []

        with patch(
            "backend.agents.dispatcher.run_question_agent_sync",
            side_effect=evals,
        ):
            asyncio.run(dispatch_review(reqs, on_result=callback_calls.append))

        assert len(callback_calls) == 3

    def test_partial_failure_still_returns_all(self):
        """If some workers raise inside run_question_agent_sync, the rest complete."""
        reqs = [make_requirement(i) for i in range(1, 4)]

        def side_effect(req: Requirement) -> Evaluation:
            if req.id == 2:
                raise RuntimeError("worker crashed")
            return make_evaluation(req.id)

        with patch(
            "backend.agents.dispatcher.run_question_agent_sync",
            side_effect=side_effect,
        ):
            results = asyncio.run(dispatch_review(reqs))

        # All 3 results present; failing worker gets an error evaluation
        assert len(results) == 3
        by_id = {e.requirement_id: e for e in results}
        assert by_id[2].needs_human_review is True
        assert by_id[1].needs_human_review is False
        assert by_id[3].needs_human_review is False

    def test_timeout_returns_error_evaluation(self):
        """Worker that exceeds timeout gets an error evaluation with needs_human_review."""
        reqs = [make_requirement(1)]

        async def slow_thread_call(*args, **kwargs):
            await asyncio.sleep(999)

        with (
            patch("backend.agents.dispatcher.WORKER_TIMEOUT_SECONDS", 0),
            patch(
                "backend.agents.dispatcher.run_question_agent_sync",
                return_value=make_evaluation(1),
            ),
        ):
            results = asyncio.run(dispatch_review(reqs))

        assert len(results) == 1
        assert results[0].needs_human_review is True

    def test_empty_requirements_returns_empty_list(self):
        results = asyncio.run(dispatch_review([]))
        assert results == []

    def test_semaphore_bounds_concurrency(self):
        """
        With MAX_CONCURRENT_WORKERS=2 and 4 requirements, at most 2 workers
        should hold the semaphore simultaneously.  We verify this by tracking
        concurrent executions via a counter.
        """
        reqs = [make_requirement(i) for i in range(1, 5)]
        max_concurrent = 2
        concurrent_count = 0
        peak = 0

        def side_effect(req: Requirement) -> Evaluation:
            nonlocal concurrent_count, peak
            concurrent_count += 1
            peak = max(peak, concurrent_count)
            # Yield briefly so other threads can run
            import time

            time.sleep(0.01)
            concurrent_count -= 1
            return make_evaluation(req.id)

        with (
            patch("backend.agents.dispatcher.MAX_CONCURRENT_WORKERS", max_concurrent),
            patch(
                "backend.agents.dispatcher.run_question_agent_sync",
                side_effect=side_effect,
            ),
        ):
            results = asyncio.run(dispatch_review(reqs))

        assert len(results) == 4
        # Peak concurrency must never exceed the semaphore limit
        assert peak <= max_concurrent


# ---------------------------------------------------------------------------
# stream_review
# ---------------------------------------------------------------------------


class TestStreamReview:
    async def _collect(self, requirements, mock_sync_fn, mock_critic=None):
        """Drain the stream_review async generator and return all events."""
        events: list[SSEEvent] = []

        critic_target = mock_critic or AsyncMock(
            side_effect=lambda evals, reqs: evals
        )

        with (
            patch(
                "backend.agents.dispatcher.run_question_agent_sync",
                side_effect=mock_sync_fn,
            ),
            patch(
                "backend.agents.critic.run_batch_critic",
                critic_target,
            ),
            patch(
                "backend.agents.dispatcher.stream_review.__wrapped__"
                if hasattr(stream_review, "__wrapped__")
                else "backend.agents.critic.run_batch_critic",
                critic_target,
            ),
        ):
            async for event in stream_review(requirements):
                events.append(event)
        return events

    def test_yields_evaluation_events_for_each_requirement(self):
        """One 'evaluation' SSEEvent must be emitted per requirement."""
        reqs = [make_requirement(i) for i in range(1, 4)]
        evals = [make_evaluation(r.id) for r in reqs]

        async def run():
            critic_mock = AsyncMock(side_effect=lambda ev, rq: ev)
            events: list[SSEEvent] = []
            with (
                patch(
                    "backend.agents.dispatcher.run_question_agent_sync",
                    side_effect=evals,
                ),
                patch(
                    "backend.agents.critic.run_batch_critic",
                    critic_mock,
                ),
            ):
                async for event in stream_review(reqs):
                    events.append(event)
            return events

        events = asyncio.run(run())
        eval_events = [e for e in events if e.event == "evaluation"]
        assert len(eval_events) == 3

    def test_yields_progress_then_critic_complete(self):
        """After all evaluation events, stream must yield progress then critic_complete."""
        reqs = [make_requirement(1)]
        evals = [make_evaluation(1)]

        async def run():
            critic_mock = AsyncMock(side_effect=lambda ev, rq: ev)
            events: list[SSEEvent] = []
            with (
                patch(
                    "backend.agents.dispatcher.run_question_agent_sync",
                    side_effect=evals,
                ),
                patch(
                    "backend.agents.critic.run_batch_critic",
                    critic_mock,
                ),
            ):
                async for event in stream_review(reqs):
                    events.append(event)
            return events

        events = asyncio.run(run())
        event_types = [e.event for e in events]
        assert "progress" in event_types
        assert "critic_complete" in event_types
        # progress must come before critic_complete
        assert event_types.index("progress") < event_types.index("critic_complete")

    def test_progress_event_counter_increments(self):
        """Each evaluation event must have an incrementing progress counter."""
        reqs = [make_requirement(i) for i in range(1, 4)]
        evals = [make_evaluation(r.id) for r in reqs]

        async def run():
            critic_mock = AsyncMock(side_effect=lambda ev, rq: ev)
            events: list[SSEEvent] = []
            with (
                patch(
                    "backend.agents.dispatcher.run_question_agent_sync",
                    side_effect=evals,
                ),
                patch(
                    "backend.agents.critic.run_batch_critic",
                    critic_mock,
                ),
            ):
                async for event in stream_review(reqs):
                    events.append(event)
            return events

        events = asyncio.run(run())
        eval_events = [e for e in events if e.event == "evaluation"]
        progress_values = [e.data["progress"] for e in eval_events]
        totals = [e.data["total"] for e in eval_events]
        assert sorted(progress_values) == list(range(1, 4))
        assert all(t == 3 for t in totals)

    def test_critic_receives_all_evaluations(self):
        """run_batch_critic must be called with all collected evaluations."""
        reqs = [make_requirement(i) for i in range(1, 4)]
        evals = [make_evaluation(r.id) for r in reqs]
        critic_called_with: list = []

        async def fake_critic(ev_list, req_list):
            critic_called_with.extend(ev_list)
            return ev_list

        async def run():
            events: list[SSEEvent] = []
            with (
                patch(
                    "backend.agents.dispatcher.run_question_agent_sync",
                    side_effect=evals,
                ),
                patch(
                    "backend.agents.critic.run_batch_critic",
                    side_effect=fake_critic,
                ),
            ):
                async for event in stream_review(reqs):
                    events.append(event)
            return events

        asyncio.run(run())
        assert len(critic_called_with) == 3

    def test_empty_requirements_still_yields_critic_events(self):
        """Even with no requirements, progress + critic_complete must be emitted."""

        async def run():
            critic_mock = AsyncMock(side_effect=lambda ev, rq: ev)
            events: list[SSEEvent] = []
            with patch(
                "backend.agents.critic.run_batch_critic",
                critic_mock,
            ):
                async for event in stream_review([]):
                    events.append(event)
            return events

        events = asyncio.run(run())
        event_types = [e.event for e in events]
        assert "progress" in event_types
        assert "critic_complete" in event_types
