# Component 1: Data Models (Shared Contracts)

**File**: `backend/models/schemas.py`
**Dependencies**: None — this must be completed first.
**Blocks**: All other components.

## Purpose
Define the Pydantic models that form the API contract between all components. Every component imports from this single file, so changes here require coordination.

## Models

### `Requirement`
Represents a single compliance question extracted from a review form.

```python
class Requirement(BaseModel):
    id: int                          # 1-indexed position in the review form
    text: str                        # Full question text, e.g. "Does the P&P state that..."
    reference: str | None = None     # APL reference, e.g. "APL 25-008, page 2"
    category: str | None = None      # Optional grouping (for narrative docs)
```

### `ComplianceRequirement` (extends `Requirement`)
Enriched requirement produced by the compliance extraction agent (Component 8) for long regulatory documents. All extra fields are optional so this is fully backward-compatible — structured and short-narrative extractors continue to produce base `Requirement` objects.

```python
class ComplianceRequirement(Requirement):
    obligation_type: str | None = None      # "mandatory" | "prohibition" | "conditional" | "recommended"
    obligation_level: str | None = None     # "mandatory" | "conditional_mandatory" | "recommended" | "informational"
    actor: str | None = None                # Who must act: "MCP", "Compliance Officer", "Provider"
    action_required: str | None = None      # What must be done (verb phrase)
    condition: str | None = None            # Trigger: "if PHI is involved", "upon detection"
    timeframe: str | None = None            # "within 30 days", "annually", "quarterly"
    evidence_needed: str | None = None      # What proves compliance: "training records", "audit logs"
    risk_area: str | None = None            # "Privacy", "Security", "Financial", "Operations"
    parent_id: int | None = None            # Links sub-requirements to their parent
    exact_quote: str | None = None          # Verbatim text from the source document
    section_heading: str | None = None      # Heading of the source section
```

### `Passage`
A chunk retrieved from the policy vector store.

```python
class Passage(BaseModel):
    text: str                        # The chunk text
    source_file: str                 # e.g. "GG/GG.1508_CEO20250129_v20241231.pdf"
    page_number: int                 # 1-indexed page within the PDF
    relevance_score: float           # Cosine similarity or re-rank score (0-1)
```

### `Evaluation`
The agent's answer for one requirement — the core output of Phase 3.

```python
class AnswerType(str, Enum):
    YES = "yes"
    NO = "no"
    PARTIAL = "partial"

class Evaluation(BaseModel):
    requirement_id: int
    answer: AnswerType
    citation_text: str               # Exact excerpt from the policy doc
    source_file: str                 # Which policy PDF
    page_number: int                 # Page within that PDF
    confidence: float                # 0.0 - 1.0
    reasoning: str                   # Why the agent chose this answer
    needs_human_review: bool = False # Flagged by critic or low confidence
    status: str = "pending"          # pending | approved | edited | rejected
    reviewer_notes: str = ""         # Human reviewer's notes
```

### `ReviewSession`
Groups all evaluations for one uploaded review form.

```python
class ReviewSession(BaseModel):
    id: str                          # UUID
    filename: str                    # Original uploaded filename
    doc_type: str                    # "structured" or "narrative"
    requirements: list[Requirement]
    evaluations: list[Evaluation] = []
    created_at: datetime
    status: str = "extracting"       # extracting | reviewing | critic | complete
    progress: int = 0                # Number of evaluations completed so far
```

### `SSEEvent`
Wrapper for server-sent events streamed to the frontend.

```python
class SSEEvent(BaseModel):
    event: str                       # "evaluation" | "progress" | "critic_complete" | "error"
    data: dict                       # Payload (usually a serialized Evaluation)
```

## API Request/Response Types

```python
class UploadResponse(BaseModel):
    session_id: str
    filename: str
    doc_type: str
    extraction_status: str           # "complete" or "processing" (long docs)
    requirements: list[Requirement]  # Empty list when extraction_status="processing"

class StartReviewRequest(BaseModel):
    session_id: str

class UpdateEvaluationRequest(BaseModel):
    answer: AnswerType | None = None
    citation_text: str | None = None
    reviewer_notes: str | None = None
    status: str | None = None         # "approved" | "edited" | "rejected"

class BulkApproveRequest(BaseModel):
    requirement_ids: list[int]
```

## Validation Rules
- `confidence` must be between 0.0 and 1.0
- `requirement_id` must match a valid requirement in the session
- `status` must be one of: `pending`, `approved`, `edited`, `rejected`
- `answer` must be one of: `yes`, `no`, `partial`

## Testing
- Unit tests for model validation (e.g., confidence bounds, enum values)
- Test serialization/deserialization roundtrip for all models
