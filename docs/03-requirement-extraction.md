# Component 3: Requirement Extraction

**Files**: `backend/agents/extractor.py`, `backend/tools/review_form_parser.py`, `backend/tools/narrative_extractor.py`
**Dependencies**: Data Models (Component 1)
**Can be built in parallel with**: Components 2, 4, 7

## Purpose
Parse an uploaded regulatory PDF and extract a list of `Requirement` objects. This runs once per upload and feeds into the parallel dispatcher (Component 5).

## Routing Pattern
The extractor uses a **two-way routing** strategy based on document length:

### Short Narrative Documents (≤20 pages)
Short regulatory documents written as prose where requirements are implicit but the full text fits in a single LLM context window.

**Detection heuristic**: Page count ≤ `LONG_DOC_PAGE_THRESHOLD` (default 20).

**Extraction method**: Single chat completion call to Gemini with the full document text in the user message. No agent framework or tool calls — the document is passed directly and the model returns a JSON array.

```python
import openai

client = openai.OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

response = client.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": full_text},
    ],
    temperature=0.2,
)
```

The system prompt instructs the model to return a JSON array of requirement objects with fields `id`, `text`, `reference`, and `category`. Output parsing handles plain JSON, markdown-fenced blocks, and JSON embedded in surrounding prose.

### Long Narrative Documents (>20 pages) — Compliance Extraction Agent
Long regulatory PDFs (e.g. 145-page DHCS policy guides) where a single LLM call loses detail, can't handle tables, and misses conditional logic buried deep in the document.

**Detection heuristic**: Page count > `LONG_DOC_PAGE_THRESHOLD`.

**Extraction method**: Multi-step pipeline (Component 8) that segments the document by section, filters for obligation language, extracts per-section in parallel, deduplicates, and assembles a requirement hierarchy. Produces enriched `ComplianceRequirement` objects with fields for obligation type, actor, condition, timeframe, and evidence needed.

See [08-compliance-extraction-agent.md](08-compliance-extraction-agent.md) for full design.

## Router Logic
```python
from backend.config import LONG_DOC_PAGE_THRESHOLD

async def classify_and_extract(pdf_path: str) -> tuple[str, list[Requirement]]:
    pages = parse_pdf(pdf_path)
    full_text = "\n".join(p["text"] for p in pages)
    page_count = len(pages)

    # Route 1: Long docs → compliance extraction agent (section-by-section pipeline)
    if page_count > LONG_DOC_PAGE_THRESHOLD:
        requirements = await run_compliance_extractor(pdf_path)
        return "compliance", requirements

    # Route 2: Short narrative → single-pass LLM chat completion
    requirements = run_narrative_extractor(full_text)
    return "narrative", requirements
```

## Output Format
A list of `Requirement` objects (see Component 1) returned to the API server, which stores them in the `ReviewSession` and sends them to the frontend for user confirmation.

## Testing
- **Short narrative**: A ≤20-page narrative PDF → verify `run_narrative_extractor` is called and returns requirements with text + reference
- **Long narrative**: Parse `data/Example Input Doc - Hard.pdf` (145 pages) → verify compliance extraction agent is used, returns enriched `ComplianceRequirement` objects
- **Router**: Verify correct two-way routing: short narrative (≤20 pages), long narrative (>20 pages)
- **Edge cases**: PDF with no extractable text, PDF at exactly 20 pages (boundary)
