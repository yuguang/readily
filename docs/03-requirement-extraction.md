# Component 3: Requirement Extraction

**Files**: `backend/agents/extractor.py`, `backend/tools/review_form_parser.py`, `backend/tools/narrative_extractor.py`
**Dependencies**: Data Models (Component 1)
**Can be built in parallel with**: Components 2, 4, 7

## Purpose
Parse an uploaded regulatory PDF and extract a list of `Requirement` objects. This runs once per upload and feeds into the parallel dispatcher (Component 5).

## Routing Pattern
The extractor uses a **three-way routing** strategy based on document type and length:

### Structured Forms (Easy)
Documents like the DHCS Submission Review Form with explicit numbered questions.

**Detection heuristic**: Text contains patterns like `"1. Does the P&P state"` and `"Yes No"` checkboxes.

**Extraction method**: Deterministic regex — no LLM needed.

```python
def parse_review_form(pdf_text: str) -> list[Requirement]:
    """
    Regex pattern: captures numbered questions (1-64) with their
    full text and APL references.

    Pattern:
      ^(\d+)\.\s+(Does the P&P state.+?)
      \(Reference:\s*(APL\s+[\d-]+,\s*pages?\s*[\d-]+)\)
    """
```

**Key parsing challenges**:
- Questions span multiple lines (multi-line regex with `re.DOTALL`)
- Some questions have sub-parts separated by "Also," or "And that"
- The `(Reference: APL 25-008, page X)` always follows the question text
- Questions 1-64 in the Easy example; count may vary for other APLs

### Short Narrative Documents (≤20 pages)
Short regulatory documents written as prose where requirements are implicit but the full text fits in a single LLM context window.

**Detection heuristic**: No numbered question pattern found AND page count ≤ `LONG_DOC_PAGE_THRESHOLD` (default 20).

**Extraction method**: Single-pass LLM via smolagents `ToolCallingAgent`.

```python
import os
from smolagents import ToolCallingAgent, OpenAIModel

model = OpenAIModel(
    model_id="gemini-2.5-pro",
    api_key=os.environ["GEMINI_API_KEY"],
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0.2,
)

narrative_extractor = ToolCallingAgent(
    tools=[parse_pdf_tool],
    model=model,
    name="narrative_extractor",
    description="Extracts compliance requirements from narrative regulatory text.",
    instructions="""
    Read the regulatory document and identify each distinct compliance requirement.
    A requirement is any statement that a Managed Care Plan (MCP) MUST, SHALL,
    or is REQUIRED to do. Also include SHOULD and EXPECTED TO statements.

    For each requirement, output:
    - id: sequential number
    - text: the requirement phrased as a yes/no question ("Does the P&P state that...")
    - reference: section/page reference in the source document
    - category: topic grouping (e.g., "Eligibility", "Payment", "Provider Network")
    """
)
```

### Long Narrative Documents (>20 pages) — Compliance Extraction Agent
Long regulatory PDFs (e.g. 145-page DHCS policy guides) where a single LLM call loses detail, can't handle tables, and misses conditional logic buried deep in the document.

**Detection heuristic**: No numbered question pattern found AND page count > `LONG_DOC_PAGE_THRESHOLD`.

**Extraction method**: Multi-step pipeline (Component 8) that segments the document by section, filters for obligation language, extracts per-section in parallel, deduplicates, and assembles a requirement hierarchy. Produces enriched `ComplianceRequirement` objects with fields for obligation type, actor, condition, timeframe, and evidence needed.

See [08-compliance-extraction-agent.md](08-compliance-extraction-agent.md) for full design.

## Router Logic
```python
from backend.config import LONG_DOC_PAGE_THRESHOLD

def classify_and_extract(pdf_path: str) -> tuple[str, list[Requirement]]:
    pages = parse_pdf(pdf_path)
    full_text = "\n".join(p["text"] for p in pages)
    page_count = len(pages)

    # Route 1: Try structured extraction first
    requirements = parse_review_form(full_text)
    if len(requirements) >= 3:
        return "structured", requirements

    # Route 2: Long docs → compliance extraction agent (section-by-section pipeline)
    if page_count > LONG_DOC_PAGE_THRESHOLD:
        requirements = await run_compliance_extractor(pdf_path)
        return "compliance", requirements

    # Route 3: Short narrative → single-pass LLM extraction
    requirements = run_narrative_extractor(full_text)
    return "narrative", requirements
```

## Output Format
A list of `Requirement` objects (see Component 1) returned to the API server, which stores them in the `ReviewSession` and sends them to the frontend for user confirmation.

## Testing
- **Structured**: Parse `data/Example Input Doc - Easy.pdf` (14 pages) → expect exactly 64 requirements, spot-check questions 1, 18, 40, 64
- **Short narrative**: A ≤20-page narrative PDF → verify the simple extractor is used, returns requirements with text + reference
- **Long narrative**: Parse `data/Example Input Doc - Hard.pdf` (145 pages) → verify compliance extraction agent is used, returns enriched `ComplianceRequirement` objects
- **Router**: Verify correct three-way routing: structured (regex hits), short narrative (≤20 pages), long narrative (>20 pages)
- **Edge cases**: PDF with no extractable text, PDF with mixed structured/narrative sections, PDF at exactly 20 pages (boundary)
