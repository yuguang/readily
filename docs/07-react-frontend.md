# Component 7: React Frontend

**Directory**: `frontend/`
**Dependencies**: API Server contract (Component 6) — can develop against mock API
**Can be built in parallel with**: Components 2, 3, 4, 5

## Purpose
A React single-page app with three views: Upload → Requirements Confirmation → Review Table. Uses SSE to progressively display results as parallel workers complete.

## Tech Stack
- React 18+ with TypeScript
- Vite for build/dev
- TailwindCSS for styling
- No heavy component library — keep it simple

## Views / User Flow

### 1. Upload View (`UploadForm.tsx`)
- Drag-and-drop or file picker for PDF upload
- **Short docs**: shows spinner, transitions to Requirements Confirmation on completion
- **Long docs** (`extraction_status: "processing"`): shows a multi-step progress bar with real-time updates via SSE

**API calls**:
- `POST /upload` with `multipart/form-data` — returns immediately for long docs
- `GET /upload/{session_id}/extraction-stream` — SSE stream for extraction progress (long docs only)

#### Extraction Progress Bar (Long Documents)
When the upload response has `extraction_status: "processing"`, the upload zone transforms into a progress display:

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│  Processing: Example Input Doc - Hard.pdf            │
│  145 pages · compliance document                     │
│                                                      │
│  Step 4 of 7: Extracting requirements                │
│  ████████████████░░░░░░░░░░  12 / 28 sections        │
│                                                      │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐        │
│  │ ✓ │ │ ✓ │ │ ✓ │ │ ● │ │   │ │   │ │   │        │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘        │
│  Parse Segment Filter Extract Dedup Hier  Done       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Step indicators**: 7 circles in a row, each representing a pipeline step. Completed steps show a checkmark, the active step pulses, and future steps are grayed out.

**Sub-progress**: Step 4 (extraction) is the longest and shows a secondary progress bar with `sections_completed / sections_total` from the SSE events.

**State**: managed by a `useExtractionProgress` hook (below). On `extraction_complete`, transitions to Requirements Confirmation.

#### `useExtractionProgress` Hook
```typescript
interface ExtractionStep {
  step: string;
  step_number: number;
  total_steps: number;
  detail: string;
  sections_completed?: number;
  sections_total?: number;
}

function useExtractionProgress(sessionId: string | null) {
  const [currentStep, setCurrentStep] = useState<ExtractionStep | null>(null);
  const [requirements, setRequirements] = useState<Requirement[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    const es = new EventSource(`/api/upload/${sessionId}/extraction-stream`);

    es.addEventListener('extraction_progress', (e) => {
      setCurrentStep(JSON.parse(e.data));
    });

    es.addEventListener('extraction_complete', (e) => {
      const data = JSON.parse(e.data);
      setRequirements(data.requirements);
      es.close();
    });

    es.onerror = () => {
      setError('Connection lost during extraction. Refresh to check status.');
      es.close();
    };

    return () => es.close();
  }, [sessionId]);

  return { currentStep, requirements, error };
}
```

### 2. Requirements Confirmation View (`RequirementsList.tsx`)
- Displays the list of extracted requirements (from `UploadResponse`)
- Shows doc type badge: "Structured (64 questions)" or "Narrative (N requirements)"
- User can review the extracted requirements before starting the review
- "Start Review" button triggers the parallel review

**API call**: `POST /review/{session_id}/start`

### 3. Review Table View (`ReviewTable.tsx`)
The main view. A table that progressively fills in as SSE results arrive.

**Columns**:
| # | Requirement (truncated) | Answer | Confidence | Evidence | Status | Actions |
|---|------------------------|--------|------------|----------|--------|---------|

**Behavior**:
- Rows start as "Pending" (gray) with a spinner
- As each SSE `evaluation` event arrives, the row fills in with answer + citation
- Color-coding based on confidence:
  - Green (≥0.8): High confidence — likely correct
  - Yellow (0.5-0.8): Medium — should review
  - Red (<0.5 or `needs_human_review`): Needs attention
- Progress bar at the top: "12 of 64 complete"

**Per-row actions**:
- **Expand**: Click row to show `EvidenceCard` with full citation text, source doc, page, and reasoning
- **Approve**: One-click approve (sets status to "approved")
- **Edit**: Inline edit of answer (dropdown: Yes/No/Partial), citation text (textarea), and reviewer notes
- **Reject**: Mark as rejected with required note

**Bulk actions** (toolbar at top):
- "Approve All High-Confidence" — bulk-approves all green rows
- "Export CSV" — downloads results

### Evidence Card (`EvidenceCard.tsx`)
Expanded view for a single evaluation:
```
┌─────────────────────────────────────────────────────┐
│ Requirement #18                                      │
│ Does the P&P state only general inpatient care is   │
│ subject to Prior Authorization...                    │
│                                                      │
│ Answer: ✅ Yes    Confidence: 0.92                   │
│                                                      │
│ Citation:                                            │
│ ┌─────────────────────────────────────────────────┐ │
│ │ "Only general inpatient care shall be subject   │ │
│ │ to prior authorization regardless of whether    │ │
│ │ services are rendered by an in-network or       │ │
│ │ out-of-network provider..."                     │ │
│ └─────────────────────────────────────────────────┘ │
│ Source: GG/GG.4521_CEO20250129.pdf, Page 8          │
│                                                      │
│ Reasoning: The policy explicitly states that only   │
│ general inpatient care requires prior auth...        │
│                                                      │
│ Reviewer Notes: [editable textarea]                  │
│                                                      │
│ [Approve]  [Edit]  [Reject]                         │
└─────────────────────────────────────────────────────┘
```

## State Management

### `useReview` Hook
Manages SSE subscription and review state.

```typescript
function useReview(sessionId: string) {
  const [evaluations, setEvaluations] = useState<Map<number, Evaluation>>(new Map());
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [phase, setPhase] = useState<'idle' | 'reviewing' | 'critic' | 'complete'>('idle');

  const startReview = async () => {
    await fetch(`/api/review/${sessionId}/start`, { method: 'POST' });
    setPhase('reviewing');

    const eventSource = new EventSource(`/api/review/${sessionId}/stream`);

    eventSource.addEventListener('evaluation', (e) => {
      const data = JSON.parse(e.data);
      setEvaluations(prev => new Map(prev).set(
        data.evaluation.requirement_id,
        data.evaluation
      ));
      setProgress({ completed: data.progress, total: data.total });
    });

    eventSource.addEventListener('critic_complete', () => {
      setPhase('critic');
      // Refresh all evaluations to get updated needs_human_review flags
      refreshEvaluations();
    });

    eventSource.addEventListener('done', () => {
      setPhase('complete');
      eventSource.close();
    });
  };

  const updateEvaluation = async (requirementId: number, updates: Partial<Evaluation>) => {
    const resp = await fetch(`/api/review/${sessionId}/results/${requirementId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
    const updated = await resp.json();
    setEvaluations(prev => new Map(prev).set(requirementId, updated));
  };

  const bulkApprove = async (ids: number[]) => {
    await fetch(`/api/review/${sessionId}/bulk-approve`, {
      method: 'POST',
      body: JSON.stringify({ requirement_ids: ids }),
    });
    // Update local state
    setEvaluations(prev => {
      const next = new Map(prev);
      ids.forEach(id => {
        const e = next.get(id);
        if (e) next.set(id, { ...e, status: 'approved' });
      });
      return next;
    });
  };

  return { evaluations, progress, phase, startReview, updateEvaluation, bulkApprove };
}
```

## API Client (`api/client.ts`)
```typescript
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080';

export async function uploadDocument(file: File): Promise<UploadResponse> { ... }
export async function startReview(sessionId: string): Promise<void> { ... }
export async function getResults(sessionId: string): Promise<ReviewSession> { ... }
export async function updateEvaluation(sessionId: string, reqId: number, updates: Partial<Evaluation>): Promise<Evaluation> { ... }
export async function bulkApprove(sessionId: string, ids: number[]): Promise<void> { ... }
export async function exportCsv(sessionId: string): Promise<Blob> { ... }
```

## Development with Mock API
To develop the frontend before the backend is ready, create a mock:

```typescript
// src/mocks/mockData.ts
export const mockRequirements: Requirement[] = [
  { id: 1, text: "Does the P&P state that under existing Contract requirements...", reference: "APL 25-008, page 1" },
  // ... 64 items
];

export const mockEvaluations: Evaluation[] = [
  { requirement_id: 1, answer: "yes", citation_text: "...", confidence: 0.95, ... },
  // ...
];
```

Use a simple timer to simulate SSE streaming in dev mode.

## Testing
- **Visual**: Manually verify all three views render correctly
- **SSE**: Test progressive rendering with simulated SSE events
- **Edit flow**: Update an evaluation, verify optimistic UI update + API call
- **Bulk approve**: Select all high-confidence, approve, verify state change
- **Export**: Download CSV, verify contents match displayed data
