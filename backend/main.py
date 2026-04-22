"""
FastAPI Server — Component 6.

Thin integration layer that wires Components 1–5 together and exposes REST + SSE
endpoints to the React frontend.  All business logic lives in Components 2–5.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from backend.agents.compliance_extractor import (
    parse_pdf_with_structure,
    run_compliance_extractor_with_progress,
)
from backend.agents.dispatcher import stream_review
from backend.config import LONG_DOC_PAGE_THRESHOLD, UPLOADS_DIR
from backend.models.schemas import (
    BulkApproveRequest,
    Evaluation,
    Requirement,
    ReviewSession,
    UpdateEvaluationRequest,
    UploadResponse,
)
from backend.tools.narrative_extractor import run_narrative_extractor
from backend.tools.pdf_parser import parse_pdf
from backend.tools.policy_search import get_chroma_collection
from backend.tools.document_segmenter import segment_document
from backend.tools.term_extractor import extract_term_definitions, upsert_term_definitions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan & App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify ChromaDB is populated on startup."""
    try:
        collection = get_chroma_collection()
        count = collection.count()
        if count == 0:
            logger.warning(
                "ChromaDB is empty — run ingestion first: python -m backend.ingestion.ingest"
            )
        else:
            logger.info("ChromaDB ready with %d document chunks.", count)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not verify ChromaDB on startup: %s", exc)
    yield


app = FastAPI(title="Readily API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session storage
# ---------------------------------------------------------------------------

sessions: dict[str, ReviewSession] = {}

# Maps session_id → uploaded PDF path (kept for the extraction-stream endpoint)
_session_pdf_paths: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_session(session_id: str) -> ReviewSession:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")
    return session


def _get_evaluation(session: ReviewSession, requirement_id: int) -> Evaluation:
    for ev in session.evaluations:
        if ev.requirement_id == requirement_id:
            return ev
    raise HTTPException(
        status_code=404,
        detail=(
            f"Evaluation for requirement {requirement_id} not found "
            f"in session {session.id!r}."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile) -> UploadResponse:
    """Upload a regulatory PDF and extract compliance requirements.

    Saves the file to ``uploads/{session_id}.pdf``, then routes by length:

    - **Short narrative** (≤20 pages): single-pass LLM extraction runs
      synchronously; response includes full list with
      ``extraction_status="complete"``.
    - **Long compliance doc** (>20 pages): returns
      immediately with ``extraction_status="processing"`` and an empty
      requirements list.  The frontend connects to
      ``GET /upload/{session_id}/extraction-stream`` for real-time progress.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    session_id = str(uuid.uuid4())
    pdf_path = UPLOADS_DIR / f"{session_id}.pdf"

    contents = await file.read()
    pdf_path.write_bytes(contents)

    try:
        pages = parse_pdf(str(pdf_path))

        if not pages:
            # Empty / image-only PDF — nothing to extract
            doc_type, requirements, extraction_status = "narrative", [], "complete"
        else:
            full_text = "\n".join(p["text"] for p in pages)

            # Route 1: long compliance doc — return immediately
            if len(pages) > LONG_DOC_PAGE_THRESHOLD:
                doc_type, requirements, extraction_status = "compliance", [], "processing"
            # Route 2: short narrative — single-pass LLM (synchronous)
            else:
                requirements = run_narrative_extractor(full_text)
                try:
                    structured_text = parse_pdf_with_structure(str(pdf_path))
                    sections = segment_document(structured_text)
                    terms = extract_term_definitions(
                        structured_text,
                        sections,
                        str(pdf_path),
                    )
                    upsert_term_definitions(terms)
                except Exception as exc:
                    logger.warning(
                        "Narrative term extraction failed for %s: %s",
                        pdf_path,
                        exc,
                    )
                doc_type, extraction_status = "narrative", "complete"

    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        logger.error("Failed to process %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=422, detail=f"Could not process PDF: {exc}"
        ) from exc

    session = ReviewSession(
        id=session_id,
        filename=file.filename,
        doc_type=doc_type,
        requirements=requirements,
        created_at=datetime.now(timezone.utc),
        status="extracting" if extraction_status == "processing" else "reviewing",
    )
    sessions[session_id] = session
    _session_pdf_paths[session_id] = str(pdf_path)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        doc_type=doc_type,
        extraction_status=extraction_status,
        requirements=requirements,
    )


@app.get("/upload/{session_id}/extraction-stream")
async def stream_extraction_progress(session_id: str) -> EventSourceResponse:
    """SSE endpoint — streams compliance extraction progress for long documents.

    Connect after receiving ``extraction_status="processing"`` from
    ``POST /upload``.  Events emitted:

    * ``extraction_progress`` — one per pipeline step (and sub-step during
      section extraction), payload matches the progress dict schema from
      :func:`~backend.agents.compliance_extractor.run_compliance_extractor_with_progress`.
    * ``extraction_complete`` — final event; payload includes the full
      ``requirements`` list and ``total_requirements`` count.
    """
    session = _get_session(session_id)
    pdf_path = _session_pdf_paths.get(session_id)
    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"No PDF path found for session {session_id!r}.",
        )

    async def event_generator():
        try:
            async for progress in run_compliance_extractor_with_progress(pdf_path):
                if progress["type"] == "progress":
                    yield {
                        "event": "extraction_progress",
                        "data": json.dumps(progress),
                    }
                elif progress["type"] == "complete":
                    # Populate session requirements from the completed extraction
                    session.requirements = [
                        Requirement(**r) for r in progress["requirements"]
                    ]
                    session.status = "reviewing"
                    yield {
                        "event": "extraction_complete",
                        "data": json.dumps({
                            "total_requirements": progress["total_requirements"],
                            "requirements": progress["requirements"],
                        }),
                    }
        except Exception as exc:
            logger.error(
                "Extraction stream failed for session %s: %s", session_id, exc
            )
            yield {
                "event": "extraction_error",
                "data": json.dumps({"error": str(exc)}),
            }

    return EventSourceResponse(event_generator())


@app.post("/review/{session_id}/start")
async def start_review(session_id: str) -> dict:
    """Acknowledge that a review should start.

    The actual review work is streamed via ``GET /review/{session_id}/stream``.
    This endpoint validates the session state and returns a confirmation so the
    frontend can sequence the two calls cleanly.
    """
    session = _get_session(session_id)
    if session.status not in ("extracting", "reviewing"):
        raise HTTPException(
            status_code=409,
            detail=f"Session is already in status {session.status!r}.",
        )
    session.status = "reviewing"
    return {"status": "started"}


@app.get("/review/{session_id}/stream")
async def stream_review_results(session_id: str) -> EventSourceResponse:
    """SSE endpoint — streams evaluation results as they complete.

    Events emitted (in order):

    * ``evaluation``      — one per completed requirement
    * ``progress``        — signals that the critic pass is starting
    * ``critic_complete`` — critic pass finished, includes updated evaluations
    * ``done``            — all work complete
    """
    session = _get_session(session_id)

    async def event_generator():
        session.status = "reviewing"

        async for sse_event in stream_review(session.requirements):
            if sse_event.event == "evaluation":
                session.evaluations.append(
                    Evaluation(**sse_event.data["evaluation"])
                )
                session.progress = sse_event.data["progress"]
            elif sse_event.event == "critic_complete":
                # Replace evaluations with the critic-updated list
                session.evaluations = [
                    Evaluation(**e) for e in sse_event.data["evaluations"]
                ]
                session.status = "critic"

            yield {"event": sse_event.event, "data": json.dumps(sse_event.data)}

        session.status = "complete"
        yield {
            "event": "done",
            "data": json.dumps({"total_evaluations": len(session.evaluations)}),
        }

    return EventSourceResponse(event_generator())


@app.get("/review/{session_id}/results", response_model=ReviewSession)
async def get_results(session_id: str) -> ReviewSession:
    """Return the full :class:`ReviewSession` including all evaluations."""
    return _get_session(session_id)


@app.patch(
    "/review/{session_id}/results/{requirement_id}",
    response_model=Evaluation,
)
async def update_evaluation(
    session_id: str,
    requirement_id: int,
    body: UpdateEvaluationRequest,
) -> Evaluation:
    """Apply human-reviewer edits to a single evaluation."""
    session = _get_session(session_id)
    evaluation = _get_evaluation(session, requirement_id)

    if body.answer is not None:
        evaluation.answer = body.answer
    if body.citation_text is not None:
        evaluation.citation_text = body.citation_text
    if body.reviewer_notes is not None:
        evaluation.reviewer_notes = body.reviewer_notes
    if body.status is not None:
        evaluation.status = body.status

    return evaluation


@app.post("/review/{session_id}/bulk-approve")
async def bulk_approve(session_id: str, body: BulkApproveRequest) -> dict:
    """Set ``status='approved'`` on all evaluations in *requirement_ids*."""
    session = _get_session(session_id)
    target_ids = set(body.requirement_ids)
    approved = 0
    for ev in session.evaluations:
        if ev.requirement_id in target_ids:
            ev.status = "approved"
            approved += 1
    return {"approved_count": approved}


@app.get("/review/{session_id}/export")
async def export_results(session_id: str) -> StreamingResponse:
    """Export all evaluation results as a CSV file download."""
    session = _get_session(session_id)
    req_map = {r.id: r for r in session.requirements}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "requirement_id",
        "requirement_text",
        "reference",
        "answer",
        "confidence",
        "citation_text",
        "source_file",
        "page_number",
        "needs_human_review",
        "status",
        "reviewer_notes",
        "reasoning",
    ])
    for ev in session.evaluations:
        req = req_map.get(ev.requirement_id)
        writer.writerow([
            ev.requirement_id,
            req.text if req else "",
            req.reference if req else "",
            ev.answer.value,
            ev.confidence,
            ev.citation_text,
            ev.source_file,
            ev.page_number,
            ev.needs_human_review,
            ev.status,
            ev.reviewer_notes,
            ev.reasoning,
        ])

    output.seek(0)
    filename = f"review_{session_id[:8]}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
