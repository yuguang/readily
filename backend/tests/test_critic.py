"""
Unit tests for backend/agents/critic.py

Covers:
- parse_critic_response: bare JSON array, fenced JSON (with and without lang),
  dict-wrapped array, regex fallback, invalid input → empty list
- run_batch_critic: skips LLM when no low-confidence evals, updates
  needs_human_review, updates answer when critic disagrees, appends critic
  reason to reasoning, ignores unknown requirement_ids from critic, handles
  unrecognised answer values gracefully
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.critic import parse_critic_response, run_batch_critic
from backend.models.schemas import AnswerType, Evaluation, Requirement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_requirement(id: int = 1) -> Requirement:
    return Requirement(id=id, text=f"Does the policy cover requirement {id}?")


def make_evaluation(
    requirement_id: int,
    answer: AnswerType = AnswerType.YES,
    confidence: float = 0.9,
    reasoning: str = "Explicit coverage found.",
    needs_human_review: bool = False,
) -> Evaluation:
    return Evaluation(
        requirement_id=requirement_id,
        answer=answer,
        citation_text="The policy states...",
        source_file="policy.pdf",
        page_number=1,
        confidence=confidence,
        reasoning=reasoning,
        needs_human_review=needs_human_review,
    )


SAMPLE_CRITIC_ITEMS = [
    {
        "requirement_id": 1,
        "verified_answer": "no",
        "needs_human_review": True,
        "reason": "Citation does not address the requirement.",
    },
    {
        "requirement_id": 2,
        "verified_answer": "yes",
        "needs_human_review": False,
        "reason": "Agent was correct.",
    },
]


# ---------------------------------------------------------------------------
# parse_critic_response
# ---------------------------------------------------------------------------


class TestParseCriticResponse:
    def test_bare_json_array(self):
        text = json.dumps(SAMPLE_CRITIC_ITEMS)
        result = parse_critic_response(text)
        assert len(result) == 2
        assert result[0]["requirement_id"] == 1
        assert result[1]["verified_answer"] == "yes"

    def test_fenced_json_with_lang(self):
        text = "```json\n" + json.dumps(SAMPLE_CRITIC_ITEMS) + "\n```"
        result = parse_critic_response(text)
        assert len(result) == 2

    def test_fenced_json_no_lang(self):
        text = "```\n" + json.dumps(SAMPLE_CRITIC_ITEMS) + "\n```"
        result = parse_critic_response(text)
        assert len(result) == 2

    def test_dict_wrapped_array(self):
        """Handles {'items': [...]} style responses."""
        wrapped = {"items": SAMPLE_CRITIC_ITEMS}
        text = json.dumps(wrapped)
        result = parse_critic_response(text)
        assert len(result) == 2

    def test_array_embedded_in_prose(self):
        """Regex fallback: array buried in surrounding text."""
        text = "Here are my findings:\n" + json.dumps(SAMPLE_CRITIC_ITEMS) + "\nEnd of review."
        result = parse_critic_response(text)
        assert len(result) == 2

    def test_invalid_json_returns_empty(self):
        result = parse_critic_response("this is not json at all")
        assert result == []

    def test_empty_string_returns_empty(self):
        result = parse_critic_response("")
        assert result == []

    def test_empty_array(self):
        result = parse_critic_response("[]")
        assert result == []

    def test_single_item_array(self):
        single = [SAMPLE_CRITIC_ITEMS[0]]
        result = parse_critic_response(json.dumps(single))
        assert len(result) == 1
        assert result[0]["needs_human_review"] is True


# ---------------------------------------------------------------------------
# run_batch_critic
# ---------------------------------------------------------------------------


class TestRunBatchCritic:
    def _run(self, evaluations, requirements, critic_response_text="[]"):
        """Helper to run run_batch_critic with a mocked LLM response."""
        mock_response = MagicMock()
        mock_response.content = critic_response_text

        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_response
        mock_model_cls.return_value = mock_model_instance

        with patch("backend.agents.critic.OpenAIModel", mock_model_cls):
            return asyncio.run(run_batch_critic(evaluations, requirements))

    def test_skips_llm_when_no_low_confidence(self):
        """No LLM call when all evaluations are above the threshold."""
        evals = [make_evaluation(1, confidence=0.9), make_evaluation(2, confidence=0.8)]
        reqs = [make_requirement(1), make_requirement(2)]

        mock_model_cls = MagicMock()
        with patch("backend.agents.critic.OpenAIModel", mock_model_cls):
            result = asyncio.run(run_batch_critic(evals, reqs))

        mock_model_cls.assert_not_called()
        assert result == evals

    def test_returns_same_list_unchanged_when_no_low_confidence(self):
        evals = [make_evaluation(1, confidence=0.75)]
        reqs = [make_requirement(1)]
        result = asyncio.run(run_batch_critic(evals, reqs))
        assert result is evals

    def test_updates_needs_human_review_flag(self):
        """Critic sets needs_human_review=True for flagged items."""
        evals = [make_evaluation(1, confidence=0.5, needs_human_review=False)]
        reqs = [make_requirement(1)]

        critic_output = json.dumps(
            [{"requirement_id": 1, "verified_answer": "yes", "needs_human_review": True, "reason": "Uncertain"}]
        )
        result = self._run(evals, reqs, critic_response_text=critic_output)

        assert result[0].needs_human_review is True

    def test_clears_needs_human_review_flag(self):
        """Critic can set needs_human_review=False on a previously flagged item."""
        evals = [make_evaluation(1, confidence=0.4, needs_human_review=True)]
        reqs = [make_requirement(1)]

        critic_output = json.dumps(
            [{"requirement_id": 1, "verified_answer": "yes", "needs_human_review": False, "reason": "Actually fine"}]
        )
        result = self._run(evals, reqs, critic_response_text=critic_output)

        assert result[0].needs_human_review is False

    def test_updates_answer_when_critic_disagrees(self):
        """If critic verified_answer differs from agent answer, answer is updated."""
        evals = [make_evaluation(1, answer=AnswerType.YES, confidence=0.5)]
        reqs = [make_requirement(1)]

        critic_output = json.dumps(
            [{"requirement_id": 1, "verified_answer": "no", "needs_human_review": True, "reason": "Wrong match"}]
        )
        result = self._run(evals, reqs, critic_response_text=critic_output)

        assert result[0].answer == AnswerType.NO

    def test_does_not_update_answer_when_critic_agrees(self):
        """If critic agrees with the agent, answer stays unchanged."""
        evals = [make_evaluation(1, answer=AnswerType.YES, confidence=0.5, reasoning="Original")]
        reqs = [make_requirement(1)]

        critic_output = json.dumps(
            [{"requirement_id": 1, "verified_answer": "yes", "needs_human_review": False, "reason": "Confirmed"}]
        )
        result = self._run(evals, reqs, critic_response_text=critic_output)

        assert result[0].answer == AnswerType.YES
        # reasoning should NOT have [Critic: ...] appended since answer unchanged
        assert "[Critic:" not in result[0].reasoning

    def test_appends_critic_reason_to_reasoning_on_answer_change(self):
        """Critic reason is appended to reasoning when the answer changes."""
        original_reasoning = "Original agent reasoning."
        evals = [make_evaluation(1, answer=AnswerType.YES, confidence=0.5, reasoning=original_reasoning)]
        reqs = [make_requirement(1)]

        critic_output = json.dumps(
            [{"requirement_id": 1, "verified_answer": "partial", "needs_human_review": True, "reason": "Partial only"}]
        )
        result = self._run(evals, reqs, critic_response_text=critic_output)

        assert original_reasoning in result[0].reasoning
        assert "[Critic: Partial only]" in result[0].reasoning

    def test_ignores_unknown_requirement_ids(self):
        """Critic response referencing an unknown id is silently skipped."""
        evals = [make_evaluation(1, confidence=0.5)]
        reqs = [make_requirement(1)]

        critic_output = json.dumps(
            [{"requirement_id": 999, "verified_answer": "no", "needs_human_review": True, "reason": "Ghost"}]
        )
        # Should not raise; eval[0] should be unchanged
        result = self._run(evals, reqs, critic_response_text=critic_output)
        assert result[0].requirement_id == 1

    def test_handles_unrecognised_answer_gracefully(self):
        """Invalid verified_answer values are skipped without raising."""
        evals = [make_evaluation(1, answer=AnswerType.YES, confidence=0.5)]
        reqs = [make_requirement(1)]

        critic_output = json.dumps(
            [{"requirement_id": 1, "verified_answer": "maybe", "needs_human_review": True, "reason": "Odd"}]
        )
        result = self._run(evals, reqs, critic_response_text=critic_output)
        # answer unchanged
        assert result[0].answer == AnswerType.YES
        # needs_human_review updated (it's processed before answer validation)
        assert result[0].needs_human_review is True

    def test_only_low_confidence_sent_to_critic(self):
        """High-confidence evaluations are not included in the critic prompt."""
        high_conf = make_evaluation(1, confidence=0.95)
        low_conf = make_evaluation(2, confidence=0.4)
        reqs = [make_requirement(1), make_requirement(2)]

        prompt_captured: list[str] = []
        mock_response = MagicMock()
        mock_response.content = "[]"

        mock_model_instance = MagicMock()

        def capture_call(messages):
            prompt_captured.append(messages[0]["content"])
            return mock_response

        mock_model_instance.side_effect = capture_call
        mock_model_cls = MagicMock(return_value=mock_model_instance)

        with patch("backend.agents.critic.OpenAIModel", mock_model_cls):
            asyncio.run(run_batch_critic([high_conf, low_conf], reqs))

        # Prompt should mention requirement 2 but not requirement 1
        assert len(prompt_captured) == 1
        assert '"requirement_id": 2' in prompt_captured[0]
        assert '"requirement_id": 1' not in prompt_captured[0]

    def test_empty_evaluations_list(self):
        """Empty input returns empty list without touching the LLM."""
        mock_model_cls = MagicMock()
        with patch("backend.agents.critic.OpenAIModel", mock_model_cls):
            result = asyncio.run(run_batch_critic([], []))
        mock_model_cls.assert_not_called()
        assert result == []
