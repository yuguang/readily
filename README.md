# Readily — Agentic Compliance Review

Automates healthcare P&P policy audits against regulatory checklists. AI agents review each compliance question against 373+ policy PDFs using RAG, then stream results to a React UI for human approval.

## Agentic Workflows

### End-to-End Pipeline

```mermaid
flowchart LR
    Upload["📄 Upload PDF"] --> Router
    Router --> Short["Narrative Agent"]
    Router --> Long["Compliance Extraction\nPipeline"]
    Short --> Confirm
    Long -->|SSE progress| Confirm
    Confirm --> Dispatcher["Parallel Dispatcher\n(8 workers)"]
    Dispatcher --> QA1["Question Agent"]
    Dispatcher --> QA2["Question Agent"]
    Dispatcher --> QAN["Question Agent"]
    QA1 -->|SSE| Review["📋 Review Table"]
    QA2 -->|SSE| Review
    QAN -->|SSE| Review
    Review --> Critic["Batch Critic\n(low-conf only)"]
    Critic --> HITL["👤 Human Review\nApprove / Edit / Reject"]
```

### 1. Requirement Extraction Router

Two-way routing based on document length. Runs once per uploaded PDF.

```mermaid
flowchart TD
    PDF["Uploaded PDF"] --> Parse["Parse PDF\n(PyMuPDF)"]
    Parse --> Length{"Page count\n> 20?"}
    Length -->|"≤ 20 pages"| Narrative["📝 Narrative Agent\nSingle-pass \nReads full text → extracts requirements and term definitions"]
    Length -->|"> 20 pages"| Compliance["📚 Compliance Extraction Pipeline\n"]
    Narrative --> Reqs
    Compliance -->|"Enriched schema\n+ progress streaming"| Reqs
```

### 2. Compliance Extraction Agent (Long Documents)

For long regulatory PDFs, segments the document, filters for obligation language, extracts per-section in parallel, then deduplicates and assembles a hierarchy.

```mermaid
flowchart TD
    PDF["PDF"] --> S1["1. Structured Parsing\nPyMuPDF: headings + tables + page numbers"]
    S1 --> S2["2. Section Segmentation\nSplit on headings"]
    S2 --> S3["3. Obligation Filtering\nRule-based keyword match\n(must / shall / required / prohibited)"]
    S3 -->|"~x sections pass\n~y skipped"| S4

    S4["4. Per-Section LLM Extraction\nToolCallingAgent (parallelized, 8 workers)\n─────────────────────────────\nget_text_tool · strip_tool · parse_list_tool\nbatch_resolve_tool · resolve_tool"]

    S4 --> S5["5. Deduplication\nEmbedding similarity > 0.90\n"]
    S5 --> S6["6. Hierarchy Assembly\nLink sub-requirements to parents\n(H1 → H2 → H3 nesting)"]
    S6 --> S7["7. Finalize\nAssign sequential IDs"]
    S7 --> Out["list of ComplianceRequirement\n(enriched schema)"]

    style S3 fill:#fef3c7,stroke:#f59e0b
    style S4 fill:#dbeafe,stroke:#3b82f6
    style S5 fill:#fce7f3,stroke:#ec4899
```

### 3. Parallel Dispatcher + Batch Critic

Fans out N requirements to concurrent workers, streams results via SSE, then runs a reflection pass on low-confidence results.

```mermaid
flowchart TD
    Reqs["Requirements"] --> Dispatch["Parallel Dispatcher\nasyncio.Semaphore"]
    Dispatch --> W1["Worker 1\nQuestion Agent"]
    Dispatch --> W2["Worker 2\nQuestion Agent"]
    Dispatch --> W3["..."]
    Dispatch --> W8["Worker 8\nQuestion Agent"]
    W1 -->|"SSE: evaluation"| Collect["Collect Results\n(stream to UI as each completes)"]
    W2 -->|"SSE: evaluation"| Collect
    W3 -->|"SSE: evaluation"| Collect
    W8 -->|"SSE: evaluation"| Collect
    Collect --> Filter{"confidence\n< 0.7?"}
    Filter -->|"~10-15% low-conf"| Critic["Batch Critic\nSingle LLM call reviews all\nlow-confidence results"]
    Filter -->|"High confidence"| Done["✅ Final Evaluations"]
    Critic -->|"Update flags:\nneeds_human_review"| Done

    style Dispatch fill:#dbeafe,stroke:#3b82f6
    style Critic fill:#fef3c7,stroke:#f59e0b
```

### 3.1 Question Agent (RAG per Requirement)

One self-contained `ToolCallingAgent` is created per compliance requirement. Each parallel worker instantiates its own agent with no shared state.

```mermaid
flowchart TD
    Req["Requirement #N\n'Does the P&P state that...'"] --> Expand["Expand into 2-3\nsearch queries"]
    Expand --> Search1["🔍 search_policies\n(query 1)"]
    Expand --> Search2["🔍 search_policies\n(query 2)"]
    Search1 --> Pick["Select best passage\nfrom top-10 results"]
    Search2 --> Pick
    Pick --> Eval{"Passage satisfies\nrequirement?"}
    Eval -->|Clear answer| Answer["Return Evaluation\n{answer, citation, confidence, reasoning}"]
    Eval -->|Borderline| Cite["🔍 evaluate_citation\n(second opinion tool)"]
    Cite --> Answer
    Eval -->|No good match| Retry{"Retries\nleft?"}
    Retry -->|Yes| Expand
    Retry -->|No| NoAnswer["Return 'no'\nlow confidence"]

    style Search1 fill:#dbeafe,stroke:#3b82f6
    style Search2 fill:#dbeafe,stroke:#3b82f6
    style Cite fill:#dbeafe,stroke:#3b82f6
```

### 5. Ingestion Pipeline (One-Time Setup)

Parses 373 policy PDFs, chunks them, generates embeddings, and stores in ChromaDB.

```mermaid
flowchart LR
    PDFs["373 Policy PDFs\n(10 folders)"] --> Parse["PyMuPDF\nExtract text + page numbers"]
    Parse --> Chunk["Semantic Chunking\n~2000 chars, 100 char overlap"]
    Chunk --> Embed["Sentence Transformers\nall-MiniLM-L6-v2"]
    Embed --> Store["ChromaDB\npolicy_documents collection"]
    Store --> Ready["Ready for\nsearch_policies queries"]
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- A [Gemini API key](https://aistudio.google.com/apikey)

## Setup

### 1. Clone and enter the project

```sh
git clone <repo-url> && cd readily
```

### 2. Backend

Create a virtual environment and install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Frontend

```sh
cd frontend
npm install
cd ..
```

### 4. Environment variables

Create a `.env` file in the project root (or export directly):

```sh
export GEMINI_API_KEY="your-gemini-api-key"
```

Optional overrides (defaults shown):

```sh
export LLM_MODEL_ID="gemini-2.5-pro"
export MAX_CONCURRENT_WORKERS=8
export CONFIDENCE_THRESHOLD=0.7
```

Alternatively:

```sh
cp .env.sample .env
```

### 5. Policy data

Place the policy PDFs under `data/Public Policies/` with the expected folder structure:

```
data/Public Policies/
├── AA/   (19 PDFs)
├── CMC/  (4 PDFs)
├── DD/   (11 PDFs)
├── EE/   (12 PDFs)
├── FF/   (24 PDFs)
├── GA/   (5 PDFs)
├── GG/   (144 PDFs)
├── HH/   (47 PDFs)
├── MA/   (69 PDFs)
└── PA/   (38 PDFs)
```

## Running

### Step 1: Ingest policy PDFs into ChromaDB (one-time)

```sh
python -m backend.ingestion.ingest
```

This parses all PDFs, chunks them, generates embeddings, and stores them in `chroma_db/`. Takes ~5–10 minutes for the full corpus.

### Step 2: Start the backend

```sh
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a review form PDF |
| GET | `/upload/{id}/extraction-stream` | SSE progress for long doc extraction |
| POST | `/review/{id}/start` | Start parallel review |
| GET | `/review/{id}/stream` | SSE stream of evaluation results |
| GET | `/review/{id}/results` | All evaluations |
| PATCH | `/review/{id}/results/{n}` | Edit one evaluation |
| POST | `/review/{id}/bulk-approve` | Bulk approve |
| GET | `/review/{id}/export` | Export CSV |

### Step 3: Start the frontend

```sh
cd frontend
npm run dev
```

Opens at `http://localhost:5173`. Upload a review form PDF (e.g. `data/Example Input Doc - Easy.pdf`) to start.

## Testing

```sh
# Backend unit tests (no API key required, ~5s)
pytest backend/tests/

# Frontend typecheck
cd frontend && npm run typecheck
```

## Evaluation Suite

`backend/evals/` contains a quality evaluation suite for the requirement extraction workflow. Unlike unit tests, evals make real LLM calls and measure output quality against defined thresholds (recall, format adherence, field completeness).

**Prerequisites:** `GEMINI_API_KEY` must be set.

```sh
# Run all evals
pytest -m eval backend/evals/ -v -s

# Run the offline parser robustness tests (no API key needed)
pytest backend/evals/eval_extraction.py::TestParseRequirementsRobustness -v

# Exclude evals from a normal test run
pytest -m "not eval"
```

### What each eval class checks

| Class | API calls | Measures |
|---|---|---|
| `TestNarrativeExtractorQuality` | 1 | Format adherence ≥90%, reference/category completeness ≥75%, count within 50–150% of ground truth, unique IDs |
| `TestNarrativeExtractorRecall` | 1 | Semantic recall ≥60% of 64 known requirements (embedding cosine similarity), hallucination rate ≤10% |
| `TestParseRequirementsRobustness` | 0 | Output parser handles all LLM response variants: plain JSON, markdown fences, prose wrapping, malformed items, extra fields |
| `TestRouterEndToEnd` | 2 | Easy PDF → `"narrative"`, Hard PDF → `"compliance"` with no mocks |
| `TestComplianceExtractorFieldRichness` | 1 | Enriched `ComplianceRequirement` fields populated: `obligation_type` ≥40%, `actor` ≥40%, `evidence_needed` ≥30%; vocabulary and parent ID integrity |
| `TestNarrativeExtractorStability` | 2 | Two independent runs produce counts within 20% of each other |

Each test prints a metric table to stdout (use `-s` to see it). The `TestRouterEndToEnd` and `TestNarrativeExtractorStability` classes are the slowest — run targeted classes when iterating on a specific component.

### Question agent evals (`backend/evals/eval_question_agent.py`)

Tests the ToolCallingAgent that answers individual compliance requirements via RAG.

**Additional prerequisite:** ChromaDB must be populated (`python -m backend.ingestion.ingest`).

```sh
# Full eval suite (~15–30 LLM calls):
pytest -m eval backend/evals/eval_question_agent.py -v -s

# Offline-only (no API key, no ChromaDB needed):
pytest backend/evals/eval_question_agent.py::TestParseAgentResultEdgeCases -v
```

| Class | API calls | ChromaDB | Measures |
|---|---|---|---|
| `TestEvaluateCitationTool` | 7 | no | Returns valid JSON with answer/confidence/reasoning; correct yes/no for clear-match and off-topic passages; confidence in [0,1] |
| `TestQuestionAgentOutputQuality` | 5 | yes | All 5 requirements return valid Evaluations; fallback rate ≤20%; avg confidence ≥0.40; yes answers have citations, source files, and page numbers |
| `TestCitationGrounding` | 5 | yes | Each "yes" citation has cosine similarity ≥0.50 to a corpus passage (anti-fabrication); citations ≤800 chars; no citations sourced from the term collection |
| `TestParseAgentResultEdgeCases` | 0 | no | Parser handles all realistic LLM output variants: null coercion, case normalisation, fenced blocks, missing fields, fallback behaviour |
| `TestSearchPoliciesTool` | 0 | yes | Result format (Score/Source/Text lines), top_k limits, positive similarity score on targeted queries |
| `TestDefineTermTool` | 0 | yes | Exact abbreviation match (upper/lower), full-term match, embedding fallback, "No definition found" for unknown terms |
| `TestAcronymAwareness` | 2 | yes | ECM/POF requirement resolves correctly; reasoning mentions "care management" or "ECM" |
| `TestAgentStability` | 4 | yes | Same requirement gives same answer on two independent runs; confidence delta ≤0.30 |

## Project Structure

```
readily/
├── backend/
│   ├── agents/
│   │   ├── dispatcher.py            # Parallel fan-out (asyncio semaphore)
│   │   ├── question_agent.py        # Per-question ToolCallingAgent
│   │   ├── critic.py                # Batch reflection on low-confidence results
│   │   ├── extractor.py             # Requirement extraction (3-way routing)
│   │   └── compliance_extractor.py  # Long-doc extraction pipeline
│   ├── tools/
│   │   ├── pdf_parser.py            # PyMuPDF text extraction
│   │   ├── review_form_parser.py    # Structured form regex parser
│   │   ├── policy_search.py         # ChromaDB vector search
│   │   ├── narrative_extractor.py   # Single-pass LLM extraction
│   │   └── document_segmenter.py    # Section segmentation for long docs
│   ├── ingestion/
│   │   ├── ingest.py                # Bulk PDF → ChromaDB pipeline
│   │   └── chunker.py               # Semantic-aware text chunking
│   ├── models/
│   │   └── schemas.py               # Pydantic models (shared contracts)
│   ├── config.py
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/client.ts
│       ├── components/              # UploadForm, ReviewTable, EvidenceCard, etc.
│       ├── hooks/                   # useReview (SSE), useExtractionProgress
│       └── types.ts
├── data/                            # Policy PDFs + example input docs
├── docs/                            # Design documents (00–08)
└── README.md
```

## Tech Stack

- **Backend**: Python, FastAPI, smolagents (`OpenAIModel` + `ToolCallingAgent`), ChromaDB, PyMuPDF, sentence-transformers
- **Frontend**: React 18, TypeScript, Vite, TailwindCSS
- **LLM**: Gemini 2.5 Pro via Gemini's OpenAI-compatible endpoint
- **Vector Store**: ChromaDB (local persistence, zero-config)

## Design Docs

Detailed component designs are in `docs/`:

- [00 — Architecture Overview](docs/00-architecture.md)
- [01 — Data Models](docs/01-data-models.md)
- [02 — Ingestion Pipeline](docs/02-ingestion-pipeline.md)
- [03 — Requirement Extraction](docs/03-requirement-extraction.md)
- [04 — Question Agent + RAG](docs/04-question-agent.md)
- [05 — Parallel Dispatcher + Critic](docs/05-parallel-dispatcher.md)
- [06 — API Server](docs/06-api-server.md)
- [07 — React Frontend](docs/07-react-frontend.md)
- [08 — Compliance Extraction Agent](docs/08-compliance-extraction-agent.md)
