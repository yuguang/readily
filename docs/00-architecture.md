# Architecture Overview

## Problem
Healthcare orgs manually audit their P&P policies against regulatory documents. A reviewer reads a 64-question checklist, searches through 373+ policy PDFs, and marks Yes/No with a citation. This system automates that review with human-in-the-loop approval.

## System Diagram
```
React UI (Upload, Review, Approve)
       │  SSE stream (results arrive as workers complete)
┌──────┴──────┐
│  FastAPI    │  ── POST /upload, POST /review, GET /review/{id}/stream
│  Backend    │  ── GET /review/{id}/results, PATCH /review/{id}/results/{n}
└──────┬──────┘
       │
  Phase 1 (one-time)        Phase 2 (per upload)         Phase 3 (parallel)
  ┌─────────────┐          ┌──────────────┐          ┌─────────────────────┐
  │  Ingestion  │          │ Requirement  │          │  Parallel Dispatcher│
  │  Pipeline   │──embed──▶│ Extractor    │──split──▶│  (asyncio semaphore)│
  │  373 PDFs   │          │ (Routing)    │          └────────┬────────────┘
  └──────┬──────┘          └──────────────┘       ┌──────────┼──────────┐
         ▼                                        ▼          ▼          ▼
  ┌──────────────┐                         ┌──────────┐┌──────────┐┌──────────┐
  │  ChromaDB    │◀── search_policies ────│ Worker 1 ││ Worker 2 ││ Worker N │
  │  Vector Store│                         │ Question ││ Question ││ Question │
  └──────────────┘                         │  Agent   ││  Agent   ││  Agent   │
                                           └──────────┘└──────────┘└──────────┘
                                                  │ (all complete)
                                                  ▼
                                           ┌──────────────┐
                                           │ Batch Critic  │
                                           │ (low-conf     │
                                           │  only)        │
                                           └──────────────┘
```

## Components

| # | Component | Doc | Owner | Dependencies |
|---|-----------|-----|-------|-------------|
| 1 | Data Models (shared contracts) | [01-data-models.md](01-data-models.md) | — | None (start here) |
| 2 | Ingestion Pipeline | [02-ingestion-pipeline.md](02-ingestion-pipeline.md) | — | Data Models |
| 3 | Requirement Extraction | [03-requirement-extraction.md](03-requirement-extraction.md) | — | Data Models |
| 4 | Question Agent + RAG | [04-question-agent.md](04-question-agent.md) | — | Data Models, Ingestion (ChromaDB must be populated) |
| 5 | Parallel Dispatcher + Critic | [05-parallel-dispatcher.md](05-parallel-dispatcher.md) | — | Data Models, Question Agent |
| 6 | FastAPI Server | [06-api-server.md](06-api-server.md) | — | All backend components |
| 7 | React Frontend | [07-react-frontend.md](07-react-frontend.md) | — | API Server (contract only) |

## Parallelizable Work Streams

These can be developed simultaneously:

```
                    ┌── Component 2: Ingestion Pipeline ──────────┐
                    │                                              │
Component 1:  ──────┼── Component 3: Requirement Extraction ──────┼──▶ Component 6  ──▶ Integration
Data Models         │                                              │    API Server      Testing
(start first)       ├── Component 4: Question Agent + RAG ────────┤
                    │                                              │
                    └── Component 7: React Frontend (mock API) ───┘
                                                                   │
                                          Component 5: Dispatcher ─┘
                                          (needs 4 first)
```

- **Components 2, 3, 4, 7** can all be built in parallel once Data Models (1) is agreed.
- **Component 5** (Dispatcher) depends on 4 (Question Agent) being defined but can stub it.
- **Component 6** (API Server) wires everything together.
- **Component 7** (Frontend) can develop against a mock API and integrate later.

## Shared Contracts
All components communicate through the Pydantic models defined in Component 1. This is the most important file to stabilize first — see [01-data-models.md](01-data-models.md).

Key shared types:
- `Requirement` — extracted from a review form
- `Evaluation` — the agent's answer for one requirement
- `ReviewSession` — groups all evaluations for one uploaded document

## Tech Stack
- **Backend**: Python 3.11+, FastAPI, smolagents, ChromaDB, PyMuPDF, sentence-transformers
- **Frontend**: React (Vite), TypeScript, TailwindCSS
- **LLM**: Gemini 2.5 Pro via smolagents `OpenAIModel` (Gemini OpenAI-compatible endpoint)
- **Vector Store**: ChromaDB (local persistence)

## Design Patterns Used
From `agentic-design-patterns-docs/`:

| Pattern | Where Applied |
|---------|--------------|
| Prompt Chaining | Sequential pipeline per question: expand → retrieve → evaluate |
| Parallelization | Fan-out all questions to concurrent workers via asyncio |
| RAG | ChromaDB vector search over 373 policy PDFs |
| Routing | Classify uploaded doc type → structured vs. narrative extraction |
| Reflection | Batch critic verifies low-confidence citations |
| Tool Use | PDF parsing, vector search, citation evaluation tools |
| Guardrails | Never fabricate citations; flag low-confidence for human review |
| Human-in-the-Loop | React UI for edit/approve/reject per finding |
