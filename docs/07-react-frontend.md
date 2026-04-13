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
- Shows upload progress spinner
- On success, transitions to Requirements Confirmation

**API call**: `POST /upload` with `multipart/form-data`

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
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
