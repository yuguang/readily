# Component 6: FastAPI Server

**Files**: `backend/main.py`, `backend/config.py`
**Dependencies**: All backend components (1-5)
**Role**: Wires all components together and exposes REST + SSE endpoints to the frontend.

## Purpose
Thin integration layer. All business logic lives in Components 2-5. The API server handles HTTP routing, file uploads, session storage, and SSE streaming.

## Endpoints

### `POST /upload`
Upload a regulatory PDF and extract requirements.

**Request**: `multipart/form-data` with a `file` field (PDF)
**Response**: `UploadResponse`
```json
{
  "session_id": "uuid",
  "filename": "Example Input Doc - Easy.pdf",
  "doc_type": "structured",
  "requirements": [
    { "id": 1, "text": "Does the P&P state that...", "reference": "APL 25-008, page 1" },
    ...
  ]
}
```

**Flow**:
1. Save uploaded file to `uploads/{session_id}.pdf`
2. Parse PDF and classify doc type (fast — regex detection only)
3. **Structured / short narrative**: extract requirements synchronously, return full `UploadResponse`
4. **Long narrative (>20 pages)**: return immediately with `extraction_status: "processing"` and empty `requirements` list. Extraction runs async; the frontend connects to the extraction SSE stream.

**Response variants**:
- Fast path (structured/short): `{ session_id, filename, doc_type, extraction_status: "complete", requirements: [...] }`
- Slow path (long): `{ session_id, filename, doc_type: "compliance", extraction_status: "processing", requirements: [] }`

### `GET /upload/{session_id}/extraction-stream`
SSE endpoint — streams progress of the compliance extraction pipeline for long documents.

**Response**: `text/event-stream`
```
event: extraction_progress
data: {"step": "parsing", "step_number": 1, "total_steps": 7, "detail": "Parsing PDF structure..."}

event: extraction_progress
data: {"step": "segmenting", "step_number": 2, "total_steps": 7, "detail": "Found 52 sections"}

event: extraction_progress
data: {"step": "filtering", "step_number": 3, "total_steps": 7, "detail": "28 of 52 sections contain obligation language"}

event: extraction_progress
data: {"step": "extracting", "step_number": 4, "total_steps": 7, "sections_completed": 12, "sections_total": 28, "detail": "Extracting requirements from sections..."}

event: extraction_progress
data: {"step": "deduplicating", "step_number": 5, "total_steps": 7, "detail": "Deduplicating 87 → 62 requirements"}

event: extraction_progress
data: {"step": "hierarchy", "step_number": 6, "total_steps": 7, "detail": "Assembling requirement hierarchy"}

event: extraction_progress
data: {"step": "finalizing", "step_number": 7, "total_steps": 7, "detail": "Assigning IDs"}

event: extraction_complete
data: {"requirements": [...], "total_requirements": 62}
```

**Implementation**:
```python
@app.get("/upload/{session_id}/extraction-stream")
async def stream_extraction_progress(session_id: str):
    session = sessions[session_id]

    async def event_generator():
        async for progress in run_compliance_extractor_with_progress(session.pdf_path):
            if progress["type"] == "progress":
                yield {"event": "extraction_progress", "data": json.dumps(progress)}
            elif progress["type"] == "complete":
                session.requirements = [Requirement(**r) for r in progress["requirements"]]
                session.status = "extracting"  # ready for review
                yield {"event": "extraction_complete", "data": json.dumps(progress)}

    return EventSourceResponse(event_generator())
```

### `POST /review/{session_id}/start`
Start the parallel review for a session.

**Request**: Empty body (or optional config overrides)
**Response**: `{ "status": "started" }`

**Flow**:
1. Look up session, get requirements
2. Launch `stream_review()` (Component 5) as a background task
3. Store evaluations in the session as they complete

### `GET /review/{session_id}/stream`
SSE endpoint — streams evaluation results as they complete.

**Response**: `text/event-stream`
```
event: evaluation
data: {"evaluation": {...}, "progress": 5, "total": 64}

event: evaluation
data: {"evaluation": {...}, "progress": 6, "total": 64}

event: progress
data: {"status": "running_critic"}

event: critic_complete
data: {"updated_count": 7}

event: done
data: {"total_evaluations": 64}
```

**Implementation** (using `sse-starlette`):
```python
from sse_starlette.sse import EventSourceResponse

@app.get("/review/{session_id}/stream")
async def stream_review_results(session_id: str):
    session = sessions[session_id]

    async def event_generator():
        async for sse_event in stream_review(session.requirements):
            # Also store evaluation in session
            if sse_event.event == "evaluation":
                session.evaluations.append(
                    Evaluation(**sse_event.data["evaluation"])
                )
                session.progress = sse_event.data["progress"]
            yield {"event": sse_event.event, "data": json.dumps(sse_event.data)}

        # Run critic pass
        session.status = "critic"
        session.evaluations = await run_batch_critic(
            session.evaluations, session.requirements
        )
        yield {"event": "critic_complete", "data": json.dumps({"updated_count": sum(1 for e in session.evaluations if e.needs_human_review)})}
        session.status = "complete"
        yield {"event": "done", "data": json.dumps({"total_evaluations": len(session.evaluations)})}

    return EventSourceResponse(event_generator())
```

### `GET /review/{session_id}/results`
Get all evaluation results for a session.

**Response**: `ReviewSession` (full object with all evaluations)

### `PATCH /review/{session_id}/results/{requirement_id}`
Update a single evaluation (human reviewer edits).

**Request**: `UpdateEvaluationRequest`
**Response**: Updated `Evaluation`

### `POST /review/{session_id}/bulk-approve`
Bulk-approve high-confidence evaluations.

**Request**: `BulkApproveRequest` (list of requirement_ids)
**Response**: `{ "approved_count": N }`

### `GET /review/{session_id}/export`
Export final results as CSV.

**Response**: `text/csv` download

## Session Storage
For simplicity, use an in-memory dict:
```python
sessions: dict[str, ReviewSession] = {}
```
For production, this would be Redis or a database. But for the take-home, in-memory is fine.

## CORS
Enable CORS for the React dev server:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Startup
On startup, verify ChromaDB is populated:
```python
@app.on_event("startup")
async def startup():
    collection = get_chroma_collection()
    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB is empty — run ingestion first: python -m backend.ingestion.ingest")
```

## Testing
- **Unit**: Test each endpoint with mock session data.
- **Integration**: Upload Easy PDF → start review → stream SSE → verify evaluations.
- **Error cases**: Upload non-PDF, start review for non-existent session, stream for already-complete session.
