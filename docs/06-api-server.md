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
2. Call `classify_and_extract(pdf_path)` (Component 3)
3. Create a `ReviewSession` and store in memory (dict keyed by session_id)
4. Return requirements for frontend confirmation

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
