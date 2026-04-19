"""
Unit tests for backend/agents/question_agent.py

Covers:
- parse_agent_result: dict input, JSON string, markdown-fenced JSON, parse failure, missing fields
- _fallback_evaluation: defaults and custom reasoning
- run_question_agent: successful run, agent exception, new agent per call
- QUESTION_AGENT_PROMPT: key content assertions
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.question_agent import (
    QUESTION_AGENT_PROMPT,
    _fallback_evaluation,
    create_question_agent,
    parse_agent_result,
    run_question_agent,
)
from backend.models.schemas import AnswerType, Evaluation, Requirement


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_requirement(id: int = 1) -> Requirement:
    return Requirement(
        id=id,
        text="Does the P&P describe the process for prior authorization?",
        reference="APL 25-008, page 2",
    )


VALID_DICT = {
    "answer": "yes",
    "citation_text": "Prior authorization requests shall be processed within 72 hours.",
    "source_file": "GG/GG.1508.pdf",
    "page_number": 3,
    "confidence": 0.92,
    "reasoning": "The policy explicitly covers prior authorization timelines.",
}

VALID_JSON_STRING = json.dumps(
    {
        "answer": "partial",
        "citation_text": "The plan maintains a utilization management program.",
        "source_file": "UM/UM.001.pdf",
        "page_number": 7,
        "confidence": 0.6,
        "reasoning": "Covers the topic but missing specific timelines.",
    }
)

VALID_FENCED_JSON = (
    "```json\n"
    + json.dumps(
        {
            "answer": "no",
            "citation_text": "",
            "source_file": "",
            "page_number": 0,
            "confidence": 0.1,
            "reasoning": "No relevant passage found.",
        }
    )
    + "\n```"
)

VALID_FENCED_NO_LANG = (
    "```\n"
    + json.dumps(
        {
            "answer": "yes",
            "citation_text": "The policy states...",
            "source_file": "file.pdf",
            "page_number": 1,
            "confidence": 0.8,
            "reasoning": "Explicit coverage.",
        }
    )
    + "\n```"
)


# ---------------------------------------------------------------------------
# parse_agent_result
# ---------------------------------------------------------------------------


class TestParseAgentResult:
    def test_valid_dict(self):
        ev = parse_agent_result(VALID_DICT, requirement_id=1)
        assert isinstance(ev, Evaluation)
        assert ev.requirement_id == 1
        assert ev.answer == AnswerType.YES
        assert ev.confidence == pytest.approx(0.92)
        assert ev.source_file == "GG/GG.1508.pdf"
        assert ev.page_number == 3
        assert ev.needs_human_review is False

    def test_valid_json_string(self):
        ev = parse_agent_result(VALID_JSON_STRING, requirement_id=2)
        assert ev.answer == AnswerType.PARTIAL
        assert ev.confidence == pytest.approx(0.6)
        assert ev.requirement_id == 2

    def test_markdown_fenced_json_with_lang(self):
        ev = parse_agent_result(VALID_FENCED_JSON, requirement_id=3)
        assert ev.answer == AnswerType.NO
        assert ev.confidence == pytest.approx(0.1)
        assert ev.requirement_id == 3

    def test_markdown_fenced_json_no_lang(self):
        ev = parse_agent_result(VALID_FENCED_NO_LANG, requirement_id=4)
        assert ev.answer == AnswerType.YES
        assert ev.confidence == pytest.approx(0.8)

    def test_answer_case_insensitive(self):
        data = dict(VALID_DICT, answer="YES")
        ev = parse_agent_result(data, requirement_id=5)
        assert ev.answer == AnswerType.YES

    def test_missing_optional_fields_use_defaults(self):
        ev = parse_agent_result({"answer": "yes"}, requirement_id=6)
        assert ev.answer == AnswerType.YES
        assert ev.citation_text == ""
        assert ev.source_file == ""
        assert ev.page_number == 0
        assert ev.confidence == pytest.approx(0.0)

    def test_invalid_json_string_returns_fallback(self):
        ev = parse_agent_result("this is not json", requirement_id=7)
        assert ev.answer == AnswerType.NO
        assert ev.confidence == 0.0
        assert ev.needs_human_review is True
        assert ev.requirement_id == 7

    def test_unexpected_type_returns_fallback(self):
        ev = parse_agent_result(42, requirement_id=8)
        assert ev.needs_human_review is True
        assert ev.requirement_id == 8

    def test_invalid_answer_enum_returns_fallback(self):
        data = dict(VALID_DICT, answer="maybe")
        ev = parse_agent_result(data, requirement_id=9)
        assert ev.needs_human_review is True

    def test_null_numeric_fields_coerced_to_defaults(self):
        """Regression: LLM returning ``page_number: null`` / ``confidence: null``
        must not blow up with ``int(None)`` / ``float(None)``.
        """
        data = {
            "answer": "no",
            "citation_text": "some text",
            "source_file": "file.pdf",
            "page_number": None,
            "confidence": None,
            "reasoning": "no data",
        }
        ev = parse_agent_result(data, requirement_id=10)
        assert ev.answer == AnswerType.NO
        assert ev.page_number == 0
        assert ev.confidence == pytest.approx(0.0)
        assert ev.needs_human_review is False  # parsed cleanly, no fallback

    def test_null_string_fields_coerced_to_empty(self):
        """Regression: LLM returning ``citation_text: null`` etc. must not
        trigger a pydantic validation error on the required ``str`` fields.
        """
        data = {
            "answer": "no",
            "citation_text": None,
            "source_file": None,
            "page_number": 0,
            "confidence": 0.0,
            "reasoning": None,
        }
        ev = parse_agent_result(data, requirement_id=11)
        assert ev.citation_text == ""
        assert ev.source_file == ""
        assert ev.reasoning == ""
        assert ev.needs_human_review is False

    def test_all_null_fields_via_json_string(self):
        """Regression: full-null payload as a JSON string is parsed cleanly."""
        payload = json.dumps(
            {
                "answer": "no",
                "citation_text": None,
                "source_file": None,
                "page_number": None,
                "confidence": None,
                "reasoning": None,
            }
        )
        ev = parse_agent_result(payload, requirement_id=12)
        assert ev.answer == AnswerType.NO
        assert ev.page_number == 0
        assert ev.confidence == pytest.approx(0.0)
        assert ev.citation_text == ""
        assert ev.source_file == ""

    def test_non_numeric_page_number_string_coerced(self):
        """A page_number like 'N/A' should not raise; fall back to 0."""
        data = dict(VALID_DICT, page_number="N/A")
        ev = parse_agent_result(data, requirement_id=13)
        assert ev.page_number == 0

    def test_stringified_numeric_fields_coerced(self):
        """Numeric fields returned as numeric strings should still parse."""
        data = dict(VALID_DICT, page_number="5", confidence="0.42")
        ev = parse_agent_result(data, requirement_id=14)
        assert ev.page_number == 5
        assert ev.confidence == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# _fallback_evaluation
# ---------------------------------------------------------------------------


class TestFallbackEvaluation:
    def test_defaults(self):
        ev = _fallback_evaluation(requirement_id=99)
        assert ev.requirement_id == 99
        assert ev.answer == AnswerType.NO
        assert ev.confidence == 0.0
        assert ev.needs_human_review is True
        assert ev.citation_text == ""
        assert ev.source_file == ""
        assert ev.page_number == 0

    def test_default_reasoning_not_empty(self):
        ev = _fallback_evaluation(requirement_id=1)
        assert ev.reasoning != ""

    def test_custom_reasoning(self):
        ev = _fallback_evaluation(requirement_id=1, reasoning="timeout occurred")
        assert "timeout occurred" in ev.reasoning


# ---------------------------------------------------------------------------
# run_question_agent
# ---------------------------------------------------------------------------


class TestRunQuestionAgent:
    def test_successful_run_returns_evaluation(self):
        """Agent returns a valid dict → parsed into an Evaluation."""
        mock_model = MagicMock()
        req = make_requirement(id=1)

        with patch(
            "backend.agents.question_agent.create_question_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run.return_value = VALID_DICT
            mock_create.return_value = mock_agent

            ev = asyncio.run(run_question_agent(req, mock_model))

        assert ev.requirement_id == 1
        assert ev.answer == AnswerType.YES
        assert ev.confidence == pytest.approx(0.92)
        assert ev.needs_human_review is False

    def test_agent_exception_returns_fallback(self):
        """Agent raises (e.g. max_steps exceeded) → fallback with needs_human_review."""
        mock_model = MagicMock()
        req = make_requirement(id=2)

        with patch(
            "backend.agents.question_agent.create_question_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run.side_effect = RuntimeError("max steps exceeded")
            mock_create.return_value = mock_agent

            ev = asyncio.run(run_question_agent(req, mock_model))

        assert ev.requirement_id == 2
        assert ev.answer == AnswerType.NO
        assert ev.needs_human_review is True
        assert "Agent error" in ev.reasoning

    def test_new_agent_created_per_call(self):
        """Each invocation of run_question_agent must create a fresh agent."""
        mock_model = MagicMock()
        req = make_requirement(id=1)

        with patch(
            "backend.agents.question_agent.create_question_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run.return_value = VALID_DICT
            mock_create.return_value = mock_agent

            asyncio.run(run_question_agent(req, mock_model))
            asyncio.run(run_question_agent(req, mock_model))

        assert mock_create.call_count == 2

    def test_task_contains_requirement_text(self):
        """The task string passed to agent.run includes the requirement text."""
        mock_model = MagicMock()
        req = make_requirement(id=1)

        with patch(
            "backend.agents.question_agent.create_question_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run.return_value = VALID_DICT
            mock_create.return_value = mock_agent

            asyncio.run(run_question_agent(req, mock_model))

        call_args = mock_agent.run.call_args
        task_str = call_args[0][0]
        assert req.text in task_str
        assert str(req.id) in task_str

    def test_requirement_without_reference(self):
        """A requirement with no reference renders 'N/A' in the task."""
        mock_model = MagicMock()
        req = Requirement(id=5, text="Does the policy cover X?")

        with patch(
            "backend.agents.question_agent.create_question_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.run.return_value = VALID_DICT
            mock_create.return_value = mock_agent

            asyncio.run(run_question_agent(req, mock_model))

        task_str = mock_agent.run.call_args[0][0]
        assert "N/A" in task_str


# ---------------------------------------------------------------------------
# QUESTION_AGENT_PROMPT content
# ---------------------------------------------------------------------------


class TestQuestionAgentPrompt:
    def test_contains_answer_types(self):
        assert "yes" in QUESTION_AGENT_PROMPT
        assert "no" in QUESTION_AGENT_PROMPT
        assert "partial" in QUESTION_AGENT_PROMPT

    def test_contains_final_answer_instruction(self):
        assert "final_answer" in QUESTION_AGENT_PROMPT

    def test_contains_anti_fabrication_warning(self):
        assert "Never fabricate" in QUESTION_AGENT_PROMPT

    def test_contains_json_field_names(self):
        for field in ("answer", "citation_text", "source_file", "page_number", "confidence", "reasoning"):
            assert field in QUESTION_AGENT_PROMPT
