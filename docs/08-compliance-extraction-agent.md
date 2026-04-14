# Component 8: Compliance Extraction Agent (Long Documents)

**Files**: `backend/agents/compliance_extractor.py`, `backend/tools/document_segmenter.py`
**Dependencies**: Data Models (Component 1), PDF Parser (Component 2)
**Can be built in parallel with**: Components 4, 5, 7

## Purpose
A specialized extraction pipeline for long regulatory PDFs (>20 pages) where requirements are buried in prose, tables, appendices, and nested conditional logic. The existing narrative extractor (Component 3) sends the full document text to a single LLM call — that breaks down on 100+ page documents due to context window limits, lost detail, and inability to handle tables or hierarchical structure.

This agent treats extraction as an **information extraction workflow**: segment first, then extract per-section, then normalize, deduplicate, and classify.

## When This Agent Runs
The router in `backend/agents/extractor.py` dispatches here when:
1. Structured regex extraction fails (< 3 numbered questions found), AND
2. The document is longer than `LONG_DOC_PAGE_THRESHOLD` pages (default 20)

Short narrative docs (≤20 pages) still use the simpler single-pass `narrative_extractor`.

## Pipeline Overview

```
PDF (145 pages)
     │
     ▼
┌─────────────────────────┐
│ 1. Structured Parsing   │  ← PyMuPDF: text + headings + tables + page numbers
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 2. Section Segmentation │  ← Split on headings/subheadings, keep tables separate
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 3. Obligation Filtering │  ← Rule-based: keep sections with must/shall/required/etc.
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Per-Section LLM      │  ← ToolCallingAgent extracts + normalizes requirements
│    Extraction            │     (parallelized across sections)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 5. Deduplication        │  ← Embedding similarity to merge near-duplicates
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 6. Hierarchy Assembly   │  ← Link sub-requirements to parent obligations
└───────────┬─────────────┘
            ▼
  list[Requirement]
```

## Step 1: Structured PDF Parsing

Enhance the existing `parse_pdf` output to preserve document structure.

```python
@tool
def parse_pdf_with_structure(pdf_path: str) -> str:
    """
    Parse a PDF preserving headings, tables, and page boundaries.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Structured text with [PAGE N], [HEADING], and [TABLE] markers.
    """
```

Output format (text with structural markers):
```
[PAGE 1]
[HEADING 1] Title of the Document
Introduction text...

[PAGE 3]
[HEADING 2] Section III: Provider Network Requirements
Narrative text...

[TABLE]
| Requirement | Responsible Party | Deadline |
| Annual training | Compliance Officer | Q1 each year |
[/TABLE]

[HEADING 3] III.A Credentialing
The MCP shall maintain a credentialing process...
```

**Implementation notes**:
- Use PyMuPDF's block-level extraction (`page.get_text("dict")`) to detect font sizes for heading levels
- Detect tables using PyMuPDF's `find_tables()` (available in pymupdf >= 1.23.0)
- Preserve page boundaries so downstream citations can map back to page numbers

## Step 2: Section Segmentation

Split the structured text into logical sections. This is deterministic — no LLM needed.

**File**: `backend/tools/document_segmenter.py`

```python
@dataclass
class DocumentSection:
    heading: str            # Section heading text
    level: int              # Heading level (1, 2, 3)
    text: str               # Full section body text
    tables: list[str]       # Any tables within this section
    page_start: int         # Starting page number
    page_end: int           # Ending page number
    char_count: int         # Length for chunking decisions

def segment_document(structured_text: str) -> list[DocumentSection]:
    """
    Split structured PDF text into sections based on heading markers.

    Handles:
    - Nested headings (H1 > H2 > H3)
    - Tables extracted as separate items within their parent section
    - Appendices and annexes as top-level sections
    - Sections with no heading (preamble, cover page) grouped as "Preamble"
    """
```

**Design decisions**:
- Split on `[HEADING N]` markers from Step 1
- Sections that are still too long (>5000 chars) get sub-split on paragraph boundaries
- Tables are kept as separate items within their parent section so they get dedicated extraction attention
- Empty sections (just a heading with no body) are merged into the next section

## Step 3: Obligation Language Filtering

Rule-based pre-filter to skip sections that contain no compliance-relevant language. This avoids wasting LLM calls on cover pages, table of contents, definitions sections, and acknowledgements.

```python
OBLIGATION_PATTERNS = [
    r"\b(?:must|shall|required to|is responsible for)\b",
    r"\b(?:prohibited|may not|must not|shall not)\b",
    r"\b(?:should|expected to|is expected)\b",
    r"\b(?:within \d+ (?:days|business days|calendar days))\b",
    r"\b(?:annually|quarterly|monthly|upon request|no later than)\b",
    r"\b(?:comply with|in accordance with|pursuant to)\b",
    r"\b(?:ensure that|maintain|establish|implement|develop)\b",
]

def filter_obligation_sections(
    sections: list[DocumentSection],
) -> tuple[list[DocumentSection], list[DocumentSection]]:
    """
    Split sections into (obligation_sections, skipped_sections).

    A section passes if it contains at least one obligation pattern match.
    Tables always pass (requirements are often embedded in table rows).
    """
```

**Why this step matters**: A 145-page PDF might have 50+ sections, but only 20–30 contain actual obligations. Skipping the rest cuts LLM calls (and cost) roughly in half.

## Step 4: Per-Section LLM Extraction

A `ToolCallingAgent` that processes **one section at a time** and extracts normalized requirements.

### Enriched Requirement Schema

Each extracted item follows a richer schema than the base `Requirement` model (extra fields are optional to maintain backward compatibility with the rest of the system):

```python
class ComplianceRequirement(Requirement):
    """Extended requirement with compliance-specific metadata."""
    obligation_type: str | None = None      # "mandatory" | "prohibition" | "conditional" | "recommended"
    obligation_level: str | None = None     # "mandatory" | "conditional_mandatory" | "recommended" | "informational"
    actor: str | None = None                # Who must act: "MCP", "Compliance Officer", "Provider", etc.
    action_required: str | None = None      # What must be done (verb phrase)
    condition: str | None = None            # Trigger: "if PHI is involved", "upon detection", etc.
    timeframe: str | None = None            # "within 30 days", "annually", "quarterly"
    evidence_needed: str | None = None      # What proves compliance: "training records", "audit logs"
    risk_area: str | None = None            # "Privacy", "Security", "Financial", "Operations"
    parent_id: int | None = None            # Links sub-requirements to their parent
    exact_quote: str | None = None          # Verbatim text from the source (distinct from `text` which is rephrased as a question)
    section_heading: str | None = None      # Heading of the source section
```

### Section Extraction Agent

```python
import os
from smolagents import ToolCallingAgent, OpenAIModel, tool

@tool
def get_section_text(section_json: str) -> str:
    """
    Returns the text of a document section for analysis.

    Args:
        section_json: Not used — the section text is pre-loaded.
    """
    # Closure over the current section; instantiated per call
    ...

model = OpenAIModel(
    model_id="gemini-2.5-pro",
    api_key=os.environ["GEMINI_API_KEY"],
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0.2,
)

section_extractor = ToolCallingAgent(
    tools=[get_section_text],
    model=model,
    max_steps=5,
    name="section_requirement_extractor",
    description="Extracts compliance requirements from one document section.",
)
```

### Section Extraction Prompt

```
You are a compliance requirement extractor. You will receive ONE section of a
regulatory document. Extract every compliance obligation from this section.

WHAT TO EXTRACT:
- Statements using: must, shall, required to, is responsible for, prohibited,
  may not, should (if normative), within X days, annually, quarterly, upon request
- Requirements embedded in tables (each row may be a separate requirement)
- Conditional requirements ("if X, then must Y")

WHAT TO SKIP:
- Definitions (unless they contain embedded obligations)
- Background/context paragraphs with no actionable language
- Examples and illustrations (unless they clarify a requirement)
- Section headings and page numbers

For each requirement, output a JSON object:
{
  "text": "Does the P&P state that [rephrased as yes/no question]?",
  "exact_quote": "verbatim text from the document",
  "reference": "Section heading, page N",
  "category": "topic label",
  "obligation_type": "mandatory | prohibition | conditional | recommended",
  "obligation_level": "mandatory | conditional_mandatory | recommended | informational",
  "actor": "who must act",
  "action_required": "what must be done",
  "condition": "trigger condition, or null",
  "timeframe": "deadline/frequency, or null",
  "evidence_needed": "what proves compliance, or null",
  "risk_area": "Privacy | Security | Financial | Operations | Clinical | Administrative"
}

IMPORTANT:
- Preserve the EXACT obligation language — do not summarize or paraphrase the
  exact_quote field.
- Capture conditional logic: "if X, then Y" → condition="if X", action="Y".
- Tables: treat each row as a potential separate requirement.
- If a statement is ambiguous between mandatory and recommended, label it
  "conditional_mandatory" and note the ambiguity in the condition field.

Return a JSON array of requirement objects via final_answer.
```

### Parallelization

Sections are independent, so extraction can be parallelized the same way as the Question Agent workers:

```python
async def extract_from_all_sections(
    sections: list[DocumentSection],
) -> list[ComplianceRequirement]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

    async def extract_one(section: DocumentSection) -> list[ComplianceRequirement]:
        async with semaphore:
            return await asyncio.to_thread(
                run_section_extractor, section
            )

    tasks = [asyncio.create_task(extract_one(s)) for s in sections]
    all_reqs = []
    for coro in asyncio.as_completed(tasks):
        section_reqs = await coro
        all_reqs.extend(section_reqs)

    return all_reqs
```

With 25 obligation sections and 8 concurrent workers, this completes in ~4 batches instead of 25 sequential calls.

## Step 5: Deduplication

The same obligation often appears in multiple places: a summary, a detailed section, an appendix checklist, and a table. We want one canonical requirement with linked supporting references.

**Strategy**: Embedding-based similarity using the same `all-MiniLM-L6-v2` model from ingestion.

```python
def deduplicate_requirements(
    requirements: list[ComplianceRequirement],
    similarity_threshold: float = 0.90,
) -> list[ComplianceRequirement]:
    """
    Merge near-duplicate requirements.

    For each cluster of similar requirements (cosine similarity > threshold):
    1. Pick the most complete one as canonical (longest exact_quote, most fields populated).
    2. Merge references from duplicates into the canonical's reference field.
    3. Drop the duplicates.
    """
```

**Why 0.90 threshold**: Lower thresholds merge genuinely different requirements that happen to share vocabulary. Higher thresholds miss obvious duplicates with minor wording differences. 0.90 is a safe starting point; tunable via config.

## Step 6: Hierarchy Assembly

Link sub-requirements to their parent obligations based on section nesting.

```python
def assemble_hierarchy(
    requirements: list[ComplianceRequirement],
    sections: list[DocumentSection],
) -> list[ComplianceRequirement]:
    """
    Set parent_id on sub-requirements based on document structure.

    Rules:
    - Requirements from H3 sections get parent_id from the nearest H2 requirement.
    - Requirements from table rows get parent_id from the enclosing section's
      top-level requirement.
    - Top-level (H1/H2) requirements have parent_id=None.
    """
```

This preserves the document's logical structure so reviewers can see:
```
Requirement #12: MCPs must maintain a credentialing program (Section III)
  ├── #12a: Initial credentialing within 60 days (Section III.A)
  ├── #12b: Re-credentialing every 3 years (Section III.B)
  └── #12c: Credentialing records retained for 10 years (Table III-1, Row 3)
```

## Step 7: ID Assignment and Final Output

After dedup and hierarchy, assign sequential IDs and convert to `Requirement` objects:

```python
def finalize_requirements(
    requirements: list[ComplianceRequirement],
) -> list[Requirement]:
    """
    Assign sequential IDs (1-indexed) and return as base Requirement objects
    with enriched fields populated as optional extras.
    """
    for i, req in enumerate(requirements, start=1):
        req.id = i
    return requirements
```

## Full Pipeline Entry Point

```python
async def run_compliance_extractor(pdf_path: str) -> list[Requirement]:
    """
    Full extraction pipeline for long regulatory documents.

    Steps:
    1. Parse PDF with structure preservation
    2. Segment into sections
    3. Filter to obligation-bearing sections
    4. Extract requirements per section (parallelized)
    5. Deduplicate
    6. Assemble hierarchy
    7. Assign IDs and return
    """
    # Step 1: Structured parse
    structured_text = parse_pdf_with_structure(pdf_path)

    # Step 2: Segment
    sections = segment_document(structured_text)

    # Step 3: Filter
    obligation_sections, skipped = filter_obligation_sections(sections)
    logger.info(
        "Kept %d/%d sections with obligation language (skipped %d).",
        len(obligation_sections), len(sections), len(skipped),
    )

    # Step 4: Parallel per-section extraction
    raw_requirements = await extract_from_all_sections(obligation_sections)

    # Step 5: Deduplicate
    deduped = deduplicate_requirements(raw_requirements)
    logger.info("Deduplicated %d → %d requirements.", len(raw_requirements), len(deduped))

    # Step 6: Hierarchy
    with_hierarchy = assemble_hierarchy(deduped, sections)

    # Step 7: Finalize
    return finalize_requirements(with_hierarchy)
```

## Table Handling

Tables deserve special attention because compliance documents frequently embed deadlines, evidence requirements, and role assignments in table form.

```python
def extract_table_requirements(
    table_text: str,
    section_heading: str,
    page_number: int,
) -> list[ComplianceRequirement]:
    """
    Extract requirements from a single table.

    Strategy:
    - Parse table into rows (split on | or tab delimiters)
    - Use column headers to map fields (e.g. "Requirement", "Responsible Party", "Deadline")
    - Each row with an obligation becomes one ComplianceRequirement
    - If column mapping fails, fall back to LLM extraction on the raw table text
    """
```

**Common table patterns in compliance docs**:
- Control matrices: Control ID | Description | Owner | Frequency | Evidence
- Retention schedules: Record Type | Retention Period | Destruction Method
- Role/responsibility charts: Function | MCP | DHCS | Delegate

## Configuration

New settings in `backend/config.py`:

```python
# Compliance extraction
LONG_DOC_PAGE_THRESHOLD = int(os.getenv("LONG_DOC_PAGE_THRESHOLD", "20"))
SECTION_MAX_CHARS = 5000          # Max chars per section before sub-splitting
DEDUP_SIMILARITY_THRESHOLD = 0.90
```

## Error Handling

- **Section too long for context window**: Sub-split on paragraph boundaries (reuse chunker logic from Component 2)
- **LLM returns malformed JSON for a section**: Log warning, skip section, flag with `needs_human_review`
- **Zero obligations extracted**: Return empty list with a logged warning — the human reviewer sees "0 requirements found" in the UI
- **Table parse failure**: Fall back to raw text LLM extraction for that table

## Testing

- **Unit**: `segment_document` on a synthetic multi-heading document, verify section count and heading levels
- **Unit**: `filter_obligation_sections` with sections containing/missing obligation language
- **Unit**: `deduplicate_requirements` with known near-duplicate pairs
- **Integration**: Run full pipeline on `data/Example Input Doc - Hard.pdf` (145 pages), verify:
  - Reasonable requirement count (expect 30–100 for a 145-page regulatory doc)
  - Each requirement has `exact_quote`, `reference`, and `obligation_type` populated
  - No obvious duplicates in output
  - Parent-child hierarchy reflects document structure
- **Comparison**: Run both the simple narrative extractor and this pipeline on the Hard doc, compare coverage and quality
- **Edge cases**: PDF with only tables, PDF with no headings (flat structure), PDF with deeply nested sections (5+ levels)
