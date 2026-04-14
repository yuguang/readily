"""
Tests for Component 6: FastAPI Server (main.py).

Covers:
- POST /upload: success, non-PDF rejection, extraction failure, session created
- POST /review/{session_id}/start: success, 404, 409 when already complete
- GET /review/{session_id}/stream: SSE events, session state updates
- GET /review/{session_id}/results: success, 404
- PATCH /review/{session_id}/results/{requirement_id}: field updates, 404s
- POST /review/{session_id}/bulk-approve: full and partial approve, 404
- GET /review/{session_id}/export: CSV format, header row, data rows, 404
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from backend.models.schemas import (
    AnswerType,
    Evaluation,
    Requirement,
    ReviewSession,
    SSEEvent,
)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _req(id: int = 1) -> Requirement:
    return Requirement(
        id=id,
        text=f"Does the P&P state requirement {id}?",
        reference=f"APL 25-008, page {id}",
    )


def _eval(requirement_id: int = 1, confidence: float = 0.9) -> Evaluation:
    return Evaluation(
        requirement_id=requirement_id,
        answer=AnswerType.YES,
        citation_text="The policy clearly states...",
        source_file="GG/policy.pdf",
        page_number=3,
        confidence=confidence,
        reasoning="Explicit coverage found.",
    )


def _session(
    session_id: str = "aaaabbbb-0000-0000-0000-000000000000",
    requirements: list[Requirement] | None = None,
    evaluations: list[Evaluation] | None = None,
    status: str = "reviewing",
) -> ReviewSession:
    return ReviewSession(
        id=session_id,
        filename="test.pdf",
        doc_type="structured",
        requirements=requirements if requirements is not None else [_req(1), _req(2)],
        evaluations=evaluations if evaluations is not None else [],
        created_at=datetime.now(timezone.utc),
        status=status,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SESSION_ID = "aaaabbbb-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Ensure the in-memory session store is empty before and after each test."""
    from backend.main import sessions, _session_pdf_paths  # noqa: PLC0415

    sessions.clear()
    _session_pdf_paths.clear()
    yield
    sessions.clear()
    _session_pdf_paths.clear()


@pytest.fixture()
def client():
    """TestClient with the ChromaDB startup check mocked out."""
    with patch("backend.main.get_chroma_collection") as mock_chroma:
        mock_chroma.return_value.count.return_value = 42
        from backend.main import app  # noqa: PLC0415

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def seeded(client):
    """A (client, session) pair with a pre-populated session in the store."""
    from backend.main import sessions  # noqa: PLC0415

    sess = _session()
    sessions[_SESSION_ID] = sess
    return client, sess


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


class TestUpload:
    def test_valid_pdf_returns_200(self, client):
        # Route 1 requires ≥ 3 structured questions
        reqs = [_req(1), _req(2), _req(3)]
        with (
            patch("backend.main.parse_pdf", return_value=[{"page_number": 1, "text": "..."}]),
            patch("backend.main.parse_review_form", return_value=reqs),
        ):
            resp = client.post(
                "/upload",
                files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "report.pdf"
        assert body["doc_type"] == "structured"
        assert body["extraction_status"] == "complete"
        assert len(body["requirements"]) == 3
        assert "session_id" in body

    def test_non_pdf_returns_400(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("doc.docx", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    def test_extraction_failure_returns_422(self, client):
        with patch(
            "backend.main.parse_pdf",
            side_effect=RuntimeError("parse error"),
        ):
            resp = client.post(
                "/upload",
                files={"file": ("bad.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            )
        assert resp.status_code == 422

    def test_upload_creates_session_in_store(self, client):
        from backend.main import sessions  # noqa: PLC0415

        reqs = [_req(1)]
        # Simulate short narrative (1 page, 0 structured questions)
        with (
            patch("backend.main.parse_pdf", return_value=[{"page_number": 1, "text": "narrative"}]),
            patch("backend.main.parse_review_form", return_value=[]),
            patch("backend.main.run_narrative_extractor", return_value=reqs),
        ):
            resp = client.post(
                "/upload",
                files={"file": ("guide.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            )
        assert resp.status_code == 200
        sid = resp.json()["session_id"]
        assert sid in sessions
        assert sessions[sid].doc_type == "narrative"

    def test_upload_response_session_id_is_uuid(self, client):
        import uuid  # noqa: PLC0415

        with (
            patch("backend.main.parse_pdf", return_value=[{"page_number": 1, "text": "..."}]),
            patch("backend.main.parse_review_form", return_value=[]),
            patch("backend.main.run_narrative_extractor", return_value=[]),
        ):
            resp = client.post(
                "/upload",
                files={"file": ("x.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
            )
        sid = resp.json()["session_id"]
        uuid.UUID(sid)  # raises if not a valid UUID

    def test_long_doc_returns_processing_status(self, client):
        """A PDF with >20 pages and no structured questions returns extraction_status='processing'."""
        many_pages = [{"page_number": i, "text": f"page {i}"} for i in range(1, 22)]  # 21 pages
        with (
            patch("backend.main.parse_pdf", return_value=many_pages),
            patch("backend.main.parse_review_form", return_value=[]),
        ):
            resp = client.post(
                "/upload",
                files={"file": ("long.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] == "compliance"
        assert body["extraction_status"] == "processing"
        assert body["requirements"] == []


# ---------------------------------------------------------------------------
# POST /review/{session_id}/start
# ---------------------------------------------------------------------------


class TestStartReview:
    def test_known_session_returns_started(self, seeded):
        client, _ = seeded
        resp = client.post(f"/review/{_SESSION_ID}/start")
        assert resp.status_code == 200
        assert resp.json() == {"status": "started"}

    def test_unknown_session_returns_404(self, client):
        resp = client.post("/review/no-such-session/start")
        assert resp.status_code == 404

    def test_complete_session_returns_409(self, seeded):
        client, sess = seeded
        sess.status = "complete"
        resp = client.post(f"/review/{_SESSION_ID}/start")
        assert resp.status_code == 409

    def test_critic_session_returns_409(self, seeded):
        client, sess = seeded
        sess.status = "critic"
        resp = client.post(f"/review/{_SESSION_ID}/start")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /review/{session_id}/results
# ---------------------------------------------------------------------------


class TestGetResults:
    def test_returns_session(self, seeded):
        client, _ = seeded
        resp = client.get(f"/review/{_SESSION_ID}/results")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == _SESSION_ID
        assert body["filename"] == "test.pdf"
        assert body["doc_type"] == "structured"

    def test_unknown_session_returns_404(self, client):
        resp = client.get("/review/no-such-session/results")
        assert resp.status_code == 404

    def test_includes_evaluations(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(requirement_id=1)]
        resp = client.get(f"/review/{_SESSION_ID}/results")
        assert resp.status_code == 200
        assert len(resp.json()["evaluations"]) == 1


# ---------------------------------------------------------------------------
# PATCH /review/{session_id}/results/{requirement_id}
# ---------------------------------------------------------------------------


class TestUpdateEvaluation:
    def test_update_answer(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(requirement_id=1)]
        resp = client.patch(
            f"/review/{_SESSION_ID}/results/1",
            json={"answer": "no", "status": "edited"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "no"
        assert body["status"] == "edited"

    def test_update_reviewer_notes(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(requirement_id=1)]
        resp = client.patch(
            f"/review/{_SESSION_ID}/results/1",
            json={"reviewer_notes": "Needs a second look."},
        )
        assert resp.status_code == 200
        assert resp.json()["reviewer_notes"] == "Needs a second look."

    def test_update_citation_text(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(requirement_id=1)]
        resp = client.patch(
            f"/review/{_SESSION_ID}/results/1",
            json={"citation_text": "Updated excerpt from page 5."},
        )
        assert resp.status_code == 200
        assert resp.json()["citation_text"] == "Updated excerpt from page 5."

    def test_partial_update_leaves_other_fields_unchanged(self, seeded):
        client, sess = seeded
        ev = _eval(requirement_id=1)
        original_confidence = ev.confidence
        sess.evaluations = [ev]
        client.patch(
            f"/review/{_SESSION_ID}/results/1",
            json={"answer": "partial"},
        )
        assert sess.evaluations[0].confidence == original_confidence

    def test_unknown_requirement_returns_404(self, seeded):
        client, sess = seeded
        sess.evaluations = []
        resp = client.patch(
            f"/review/{_SESSION_ID}/results/999",
            json={"answer": "yes"},
        )
        assert resp.status_code == 404

    def test_unknown_session_returns_404(self, client):
        resp = client.patch(
            "/review/no-session/results/1",
            json={"answer": "yes"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /review/{session_id}/bulk-approve
# ---------------------------------------------------------------------------


class TestBulkApprove:
    def test_approve_all(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1), _eval(2)]
        resp = client.post(
            f"/review/{_SESSION_ID}/bulk-approve",
            json={"requirement_ids": [1, 2]},
        )
        assert resp.status_code == 200
        assert resp.json()["approved_count"] == 2
        assert all(ev.status == "approved" for ev in sess.evaluations)

    def test_approve_subset(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1), _eval(2)]
        resp = client.post(
            f"/review/{_SESSION_ID}/bulk-approve",
            json={"requirement_ids": [1]},
        )
        assert resp.status_code == 200
        assert resp.json()["approved_count"] == 1
        assert sess.evaluations[0].status == "approved"
        assert sess.evaluations[1].status == "pending"

    def test_approve_nonexistent_ids_returns_zero(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1)]
        resp = client.post(
            f"/review/{_SESSION_ID}/bulk-approve",
            json={"requirement_ids": [99]},
        )
        assert resp.status_code == 200
        assert resp.json()["approved_count"] == 0

    def test_unknown_session_returns_404(self, client):
        resp = client.post(
            "/review/no-session/bulk-approve",
            json={"requirement_ids": [1]},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /review/{session_id}/export
# ---------------------------------------------------------------------------


class TestExport:
    def test_returns_csv_content_type(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1)]
        resp = client.get(f"/review/{_SESSION_ID}/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_response_is_attachment(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1)]
        resp = client.get(f"/review/{_SESSION_ID}/export")
        assert "attachment" in resp.headers["content-disposition"]

    def test_csv_has_header_row(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1)]
        resp = client.get(f"/review/{_SESSION_ID}/export")
        first_line = resp.text.splitlines()[0]
        assert "requirement_id" in first_line
        assert "answer" in first_line
        assert "confidence" in first_line

    def test_csv_data_row_count(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1), _eval(2)]
        resp = client.get(f"/review/{_SESSION_ID}/export")
        lines = [l for l in resp.text.splitlines() if l]
        assert len(lines) == 3  # header + 2 data rows

    def test_csv_includes_requirement_text(self, seeded):
        client, sess = seeded
        sess.evaluations = [_eval(1)]
        resp = client.get(f"/review/{_SESSION_ID}/export")
        assert "Does the P&P state requirement 1?" in resp.text

    def test_empty_evaluations_returns_header_only(self, seeded):
        client, sess = seeded
        sess.evaluations = []
        resp = client.get(f"/review/{_SESSION_ID}/export")
        assert resp.status_code == 200
        lines = [l for l in resp.text.splitlines() if l]
        assert len(lines) == 1  # header only

    def test_unknown_session_returns_404(self, client):
        resp = client.get("/review/no-session/export")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /review/{session_id}/stream (SSE)
# ---------------------------------------------------------------------------


def _make_sse_events(ev: Evaluation) -> list[SSEEvent]:
    """Build a minimal but realistic sequence of SSE events."""
    return [
        SSEEvent(
            event="evaluation",
            data={"evaluation": ev.model_dump(mode="json"), "progress": 1, "total": 1},
        ),
        SSEEvent(event="progress", data={"status": "running_critic"}),
        SSEEvent(
            event="critic_complete",
            data={"evaluations": [ev.model_dump(mode="json")]},
        ),
    ]


def _parse_sse_lines(text: str) -> list[dict]:
    """Return a list of {event, data} dicts parsed from raw SSE text."""
    events = []
    current: dict = {}
    for line in text.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:"):].strip())
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


class TestStreamReview:
    def test_unknown_session_returns_404(self, client):
        resp = client.get("/review/no-such-session/stream")
        assert resp.status_code == 404

    def test_stream_content_type_is_event_stream(self, seeded):
        client, sess = seeded
        ev = _eval(1)

        async def mock_stream(requirements):
            for e in _make_sse_events(ev):
                yield e

        with patch("backend.main.stream_review", mock_stream):
            resp = client.get(f"/review/{_SESSION_ID}/stream")

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_evaluation_event(self, seeded):
        client, sess = seeded
        ev = _eval(1)

        async def mock_stream(requirements):
            for e in _make_sse_events(ev):
                yield e

        with patch("backend.main.stream_review", mock_stream):
            resp = client.get(f"/review/{_SESSION_ID}/stream")

        events = _parse_sse_lines(resp.text)
        event_names = [e["event"] for e in events]
        assert "evaluation" in event_names

    def test_stream_emits_done_event(self, seeded):
        client, sess = seeded
        ev = _eval(1)

        async def mock_stream(requirements):
            for e in _make_sse_events(ev):
                yield e

        with patch("backend.main.stream_review", mock_stream):
            resp = client.get(f"/review/{_SESSION_ID}/stream")

        events = _parse_sse_lines(resp.text)
        done_events = [e for e in events if e.get("event") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["total_evaluations"] == 1

    def test_stream_updates_session_evaluations(self, seeded):
        client, sess = seeded
        ev = _eval(requirement_id=1)

        async def mock_stream(requirements):
            for e in _make_sse_events(ev):
                yield e

        with patch("backend.main.stream_review", mock_stream):
            client.get(f"/review/{_SESSION_ID}/stream")

        # After stream completes, session should have evaluations
        assert len(sess.evaluations) == 1
        assert sess.evaluations[0].requirement_id == 1

    def test_stream_sets_session_status_complete(self, seeded):
        client, sess = seeded
        ev = _eval(1)

        async def mock_stream(requirements):
            for e in _make_sse_events(ev):
                yield e

        with patch("backend.main.stream_review", mock_stream):
            client.get(f"/review/{_SESSION_ID}/stream")

        assert sess.status == "complete"

    def test_stream_empty_requirements(self, seeded):
        """Session with no requirements still emits a done event."""
        client, sess = seeded
        sess.requirements = []

        async def mock_stream(requirements):
            yield SSEEvent(event="progress", data={"status": "running_critic"})
            yield SSEEvent(event="critic_complete", data={"evaluations": []})

        with patch("backend.main.stream_review", mock_stream):
            resp = client.get(f"/review/{_SESSION_ID}/stream")

        events = _parse_sse_lines(resp.text)
        done = [e for e in events if e.get("event") == "done"]
        assert len(done) == 1
        assert done[0]["data"]["total_evaluations"] == 0


# ---------------------------------------------------------------------------
# GET /upload/{session_id}/extraction-stream (SSE)
# ---------------------------------------------------------------------------


class TestExtractionStream:
    """Tests for the long-doc extraction SSE endpoint."""

    def _seed_extracting_session(self):
        """Insert a session in 'extracting' status and return it."""
        from backend.main import sessions, _session_pdf_paths  # noqa: PLC0415

        sess = _session(status="extracting", requirements=[])
        sessions[_SESSION_ID] = sess
        _session_pdf_paths[_SESSION_ID] = "/fake/path.pdf"
        return sess

    def test_unknown_session_returns_404(self, client):
        resp = client.get("/upload/no-such-session/extraction-stream")
        assert resp.status_code == 404

    def test_no_pdf_path_returns_404(self, client):
        """Session exists but _session_pdf_paths has no entry → 404."""
        from backend.main import sessions  # noqa: PLC0415

        sess = _session(status="extracting", requirements=[])
        sessions[_SESSION_ID] = sess
        # Deliberately do NOT add to _session_pdf_paths
        resp = client.get(f"/upload/{_SESSION_ID}/extraction-stream")
        assert resp.status_code == 404

    def test_stream_content_type_is_event_stream(self, client):
        sess = self._seed_extracting_session()

        async def mock_extractor(pdf_path):
            yield {"type": "complete", "requirements": [], "total_requirements": 0}

        with patch("backend.main.run_compliance_extractor_with_progress", mock_extractor):
            resp = client.get(f"/upload/{_SESSION_ID}/extraction-stream")

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_extraction_progress_events(self, client):
        self._seed_extracting_session()

        async def mock_extractor(pdf_path):
            yield {
                "type": "progress", "step": "parsing",
                "step_number": 1, "total_steps": 7, "detail": "Parsing...",
            }
            yield {"type": "complete", "requirements": [], "total_requirements": 0}

        with patch("backend.main.run_compliance_extractor_with_progress", mock_extractor):
            resp = client.get(f"/upload/{_SESSION_ID}/extraction-stream")

        events = _parse_sse_lines(resp.text)
        progress_events = [e for e in events if e.get("event") == "extraction_progress"]
        assert len(progress_events) == 1
        assert progress_events[0]["data"]["step"] == "parsing"

    def test_stream_emits_extraction_complete_event(self, client):
        self._seed_extracting_session()
        reqs_data = [{"id": 1, "text": "Does the P&P state X?"}]

        async def mock_extractor(pdf_path):
            yield {"type": "complete", "requirements": reqs_data, "total_requirements": 1}

        with patch("backend.main.run_compliance_extractor_with_progress", mock_extractor):
            resp = client.get(f"/upload/{_SESSION_ID}/extraction-stream")

        events = _parse_sse_lines(resp.text)
        complete_events = [e for e in events if e.get("event") == "extraction_complete"]
        assert len(complete_events) == 1
        assert complete_events[0]["data"]["total_requirements"] == 1

    def test_stream_updates_session_requirements(self, client):
        sess = self._seed_extracting_session()
        reqs_data = [{"id": 1, "text": "Does the P&P state X?"}, {"id": 2, "text": "Does the P&P state Y?"}]

        async def mock_extractor(pdf_path):
            yield {"type": "complete", "requirements": reqs_data, "total_requirements": 2}

        with patch("backend.main.run_compliance_extractor_with_progress", mock_extractor):
            client.get(f"/upload/{_SESSION_ID}/extraction-stream")

        assert len(sess.requirements) == 2
        assert sess.requirements[0].id == 1
        assert sess.requirements[1].id == 2

    def test_stream_sets_session_status_reviewing(self, client):
        sess = self._seed_extracting_session()

        async def mock_extractor(pdf_path):
            yield {"type": "complete", "requirements": [], "total_requirements": 0}

        with patch("backend.main.run_compliance_extractor_with_progress", mock_extractor):
            client.get(f"/upload/{_SESSION_ID}/extraction-stream")

        assert sess.status == "reviewing"
