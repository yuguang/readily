# Component 3: Requirement Extraction

**Files**: `backend/agents/extractor.py`, `backend/tools/review_form_parser.py`, `backend/tools/narrative_extractor.py`
**Dependencies**: Data Models (Component 1)
**Can be built in parallel with**: Components 2, 4, 7

## Purpose
Parse an uploaded regulatory PDF and extract a list of `Requirement` objects. This runs once per upload and feeds into the parallel dispatcher (Component 5).

## Routing Pattern
The extractor uses a **routing** strategy to handle two document types:

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

### Narrative Documents (Hard)
Regulatory documents written as prose where requirements are implicit.

**Detection heuristic**: No numbered question pattern found. Text reads as continuous paragraphs.

**Extraction method**: LLM-based via smolagents `ToolCallingAgent`.

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

## Router Logic
```python
def classify_and_extract(pdf_path: str) -> tuple[str, list[Requirement]]:
    pages = parse_pdf(pdf_path)
    full_text = "\n".join(p["text"] for p in pages)

    # Try structured extraction first
    requirements = parse_review_form(full_text)
    if len(requirements) >= 3:  # At least 3 numbered questions found
        return "structured", requirements

    # Fall back to narrative extraction
    requirements = run_narrative_extractor(full_text)
    return "narrative", requirements
```

## Output Format
A list of `Requirement` objects (see Component 1) returned to the API server, which stores them in the `ReviewSession` and sends them to the frontend for user confirmation.

## Testing
- **Structured**: Parse `data/Example Input Doc - Easy.pdf` → expect exactly 64 requirements, spot-check questions 1, 18, 40, 64
- **Narrative**: Parse `data/Example Input Doc - Hard.pdf` → verify reasonable requirement count, each has text + reference
- **Router**: Verify correct routing for both doc types
- **Edge cases**: PDF with no extractable text, PDF with mixed structured/narrative sections
