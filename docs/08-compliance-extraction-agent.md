# Component 8: Compliance Extraction Agent (Long Documents)

**Files**: `backend/agents/compliance_extractor.py`, `backend/tools/document_segmenter.py`, `backend/tools/term_extractor.py`
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
│ 3. Term Definition      │  ← Extract acronyms + glossary entries
│    Extraction            │     (e.g. ECM, POF, MCP, SUD) → vector DB
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 4. Obligation Filtering │  ← Rule-based: keep sections with must/shall/required/etc.
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 5. Per-Section LLM      │  ← ToolCallingAgent extracts + normalizes requirements
│    Extraction            │     (parallelized across sections)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 6. Deduplication        │  ← Embedding similarity to merge near-duplicates
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ 7. Hierarchy Assembly   │  ← Link sub-requirements to parent obligations
└───────────┬─────────────┘
            ▼
  list[Requirement]  +  TermDefinition vector store
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

## Step 3: Term Definition Extraction

Regulatory documents rely heavily on program-specific terminology and acronyms (e.g. the CalAIM policy guide defines **ECM** = Enhanced Care Management, **POF** = Population of Focus, **MCP** = Managed Care Plan, **SUD** = Substance Use Disorder, **I/DD** = Intellectual or Developmental Disability, **TBI** = Traumatic Brain Injury, etc.). Downstream agents (the Question Agent in Component 4, the Critic) need these definitions to correctly interpret policy passages and user-facing requirement text. We extract them here, once, and persist them to a dedicated vector collection for cheap lookup.

**File**: `backend/tools/term_extractor.py`

### Term Definition Model
A new Pydantic model lives alongside `Requirement` / `ComplianceRequirement` in `backend/models/schemas.py`:

```python
class TermDefinition(BaseModel):
    term: str                        # Full term, e.g. "Enhanced Care Management"
    abbreviation: str | None = None  # Acronym/short form, e.g. "ECM"
    definition: str                  # Verbatim or lightly cleaned definition text
    source_file: str                 # Path to the source PDF
    page_number: int                 # Page where the definition appears
    section_heading: str | None = None  # e.g. "Appendix C: Definitions for..."
    source: str                      # "glossary" | "appendix" | "inline_acronym" | "definition_table"
```

### Sources of Definitions
The extractor looks in three places, in priority order — later sources do **not** overwrite earlier ones for the same `(term, abbreviation)` key:

1. **Glossary / Definitions / Abbreviations sections** — headings matching `r"(?i)^(?:appendix [a-z]:\s*)?(definitions?|glossary|abbreviations?|acronyms?|key terms?)\b"`. Parse the body for term/definition pairs in any of these formats:
   - `Term \u2014 definition.` or `Term: definition.` line-based patterns
   - Bullet or dash lists: `- Term: definition`
   - Two-column tables with headers like `Criteria | Definition` or `Term | Definition` (common in the `Example Input Doc - Hard.pdf` — Appendix C of the CalAIM guide uses this format for Mental Illness, SUD, Chronic Condition, I/DD, TBI, HIV/AIDS, Pregnant or Postpartum)
2. **Inline first-use acronyms** — regex over the full document body: `r"([A-Z][A-Za-z0-9&/\- ]{2,80}?)\s*\(([A-Z][A-Z0-9/&\-]{1,10})\)"`. For each match, validate that the initials of the phrase roughly match the acronym (allowing stop-words like "of", "and", "the"). This catches patterns like `Enhanced Care Management (ECM)`, `Population of Focus (POF)`, `Managed Care Plans (MCPs)`, `Substance Use Disorder (SUD)`.
3. **Definition tables within regular sections** — any `[TABLE]` block whose header row contains `definition`, `criteria`, or `meaning` is parsed as a definition table.

### Extractor API

```python
def extract_term_definitions(
    structured_text: str,
    sections: list[DocumentSection],
    source_file: str,
) -> list[TermDefinition]:
    """
    Scan the structured PDF text and its segmented sections for term definitions.

    Returns a deduplicated list of TermDefinition objects. Deduplication prefers
    explicit glossary entries over inline acronym expansions for the same key.
    """
```

**Determinism**: this step is purely rule-based (regex + table parsing). No LLM call is required, keeping it cheap enough to run on every ingested document. If a glossary section exceeds 10,000 chars and rule-based parsing yields fewer than 5 entries, fall back to a single LLM call that extracts term/definition pairs from the section as JSON.

### Persistence: the `document_terms` Vector Collection
Term definitions live in a **separate ChromaDB collection** from the policy corpus so term lookups never collide with policy retrieval:

- Collection name: `document_terms`
- Persistence directory: `chroma_db/` (same directory as `policy_documents`, different collection)
- Embedding function: `sentence-transformers/all-MiniLM-L6-v2` (same as ingestion — keeps a single model loaded)
- Document ID: `f"{source_file}::{abbreviation or term}"` so re-extracting the same file upserts instead of duplicating
- Embedded text: `f"{term} ({abbreviation}): {definition}"` — concatenating the term, abbreviation, and definition gives the best retrieval for both acronym-style ("What is POF?") and concept-style ("population with the highest needs") queries
- Metadata per entry: `term`, `abbreviation`, `definition`, `source_file`, `page_number`, `section_heading`, `source`

```python
def upsert_term_definitions(terms: list[TermDefinition]) -> None:
    """
    Upsert TermDefinitions into the `document_terms` ChromaDB collection.

    Idempotent: re-running on the same document overwrites existing entries
    for that source_file.
    """
    collection = get_chroma_client().get_or_create_collection(
        name="document_terms",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        ),
    )
    ids = [f"{t.source_file}::{t.abbreviation or t.term}" for t in terms]
    documents = [f"{t.term} ({t.abbreviation}): {t.definition}" if t.abbreviation
                 else f"{t.term}: {t.definition}" for t in terms]
    metadatas = [t.model_dump(exclude={"definition"}) | {"definition": t.definition}
                 for t in terms]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
```

**Why a separate collection**: policies and term definitions have very different retrieval characteristics. A query like "POF" against the `policy_documents` collection would return hundreds of policy passages that happen to mention the acronym; against `document_terms` it returns a handful of canonical definitions. Splitting them also lets the Question Agent expose two distinct tools (see Component 4's `define_term`).

## Step 4: Obligation Language Filtering

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

## Step 5: Per-Section LLM Extraction

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

## Step 6: Deduplication

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

## Step 7: Hierarchy Assembly

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

## Step 8: ID Assignment and Final Output

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

Two entry points: a simple one that returns the final list, and a streaming one that yields progress events for the frontend.

```python
async def run_compliance_extractor(pdf_path: str) -> list[Requirement]:
    """Simple entry point — returns final requirements list."""
    results = []
    async for event in run_compliance_extractor_with_progress(pdf_path):
        if event["type"] == "complete":
            results = event["requirements"]
    return results


async def run_compliance_extractor_with_progress(
    pdf_path: str,
) -> AsyncGenerator[dict, None]:
    """
    Streaming entry point — yields progress dicts for each pipeline step.

    Each yielded dict has {"type": "progress", "step": ..., "step_number": ..., ...}
    or {"type": "complete", "requirements": [...], "total_requirements": N}.

    The API server forwards these as SSE events to the frontend.
    """
    # Step 1: Structured parse
    yield {"type": "progress", "step": "parsing", "step_number": 1, "total_steps": 8,
           "detail": "Parsing PDF structure..."}
    structured_text = parse_pdf_with_structure(pdf_path)

    # Step 2: Segment
    yield {"type": "progress", "step": "segmenting", "step_number": 2, "total_steps": 8,
           "detail": "Segmenting document..."}
    sections = segment_document(structured_text)
    yield {"type": "progress", "step": "segmenting", "step_number": 2, "total_steps": 8,
           "detail": f"Found {len(sections)} sections"}

    # Step 3: Extract term definitions (rule-based; persist to document_terms collection)
    yield {"type": "progress", "step": "term_extraction", "step_number": 3, "total_steps": 8,
           "detail": "Extracting term definitions and acronyms..."}
    terms = extract_term_definitions(structured_text, sections, source_file=pdf_path)
    upsert_term_definitions(terms)
    yield {"type": "progress", "step": "term_extraction", "step_number": 3, "total_steps": 8,
           "detail": f"Indexed {len(terms)} term definitions (e.g. ECM, POF, MCP)"}

    # Step 4: Filter
    yield {"type": "progress", "step": "filtering", "step_number": 4, "total_steps": 8,
           "detail": "Filtering for obligation language..."}
    obligation_sections, skipped = filter_obligation_sections(sections)
    yield {"type": "progress", "step": "filtering", "step_number": 4, "total_steps": 8,
           "detail": f"{len(obligation_sections)} of {len(sections)} sections contain obligation language"}

    # Step 5: Parallel per-section extraction (with sub-progress)
    yield {"type": "progress", "step": "extracting", "step_number": 5, "total_steps": 8,
           "sections_completed": 0, "sections_total": len(obligation_sections),
           "detail": "Extracting requirements from sections..."}

    raw_requirements = []
    completed = 0
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)

    async def extract_one(section):
        async with semaphore:
            return await asyncio.to_thread(run_section_extractor, section)

    tasks = [asyncio.create_task(extract_one(s)) for s in obligation_sections]
    for coro in asyncio.as_completed(tasks):
        section_reqs = await coro
        raw_requirements.extend(section_reqs)
        completed += 1
        yield {"type": "progress", "step": "extracting", "step_number": 5, "total_steps": 8,
               "sections_completed": completed, "sections_total": len(obligation_sections),
               "detail": "Extracting requirements from sections..."}

    # Step 6: Deduplicate
    yield {"type": "progress", "step": "deduplicating", "step_number": 6, "total_steps": 8,
           "detail": f"Deduplicating {len(raw_requirements)} requirements..."}
    deduped = deduplicate_requirements(raw_requirements)
    yield {"type": "progress", "step": "deduplicating", "step_number": 6, "total_steps": 8,
           "detail": f"Deduplicated {len(raw_requirements)} → {len(deduped)} requirements"}

    # Step 7: Hierarchy
    yield {"type": "progress", "step": "hierarchy", "step_number": 7, "total_steps": 8,
           "detail": "Assembling requirement hierarchy"}
    with_hierarchy = assemble_hierarchy(deduped, sections)

    # Step 8: Finalize
    yield {"type": "progress", "step": "finalizing", "step_number": 8, "total_steps": 8,
           "detail": "Assigning IDs"}
    final = finalize_requirements(with_hierarchy)

    yield {"type": "complete", "requirements": [r.model_dump() for r in final],
           "total_requirements": len(final),
           "term_count": len(terms)}
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

# Term definition extraction
TERM_COLLECTION_NAME = "document_terms"
TERM_DEFINITION_MIN_LEN = 10      # Minimum chars for a definition body to be kept
TERM_ACRONYM_MAX_LEN = 10         # Maximum chars in an acronym (skips long all-caps phrases)
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
- **Unit**: `extract_term_definitions` on synthetic glossary-style input and inline-acronym text:
  - Input: `"Enhanced Care Management (ECM) is a whole-person approach..."` → asserts a `TermDefinition(term="Enhanced Care Management", abbreviation="ECM", ...)` is produced
  - Input: a `Criteria | Definition` two-column table → asserts one `TermDefinition` per row with `source="definition_table"`
  - Dedup precedence: glossary entry wins over inline acronym for the same `(term, abbreviation)` key
- **Unit**: `upsert_term_definitions` round-trips — upsert then query `document_terms` and verify the canonical definition is returned
- **Integration**: Run full pipeline on `data/Example Input Doc - Hard.pdf` (145 pages), verify:
  - Reasonable requirement count (expect 30–100 for a 145-page regulatory doc)
  - Each requirement has `exact_quote`, `reference`, and `obligation_type` populated
  - No obvious duplicates in output
  - Parent-child hierarchy reflects document structure
  - **Term extraction**: `document_terms` collection contains canonical entries for at least `ECM`, `POF`, `MCP`, `DHCS`, `CalAIM`, `SUD`, `I/DD`, `TBI`, `HIV/AIDS`, and the Appendix C criteria (Mental Illness, Chronic Condition, Pregnant or Postpartum), each with non-empty `definition` and the correct `page_number` from the Hard PDF
- **Comparison**: Run both the simple narrative extractor and this pipeline on the Hard doc, compare coverage and quality
- **Edge cases**: PDF with only tables, PDF with no headings (flat structure), PDF with deeply nested sections (5+ levels), PDF with no definitions or glossary (term extractor returns empty list without error)
