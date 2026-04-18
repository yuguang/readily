# Component 4: Question Agent + RAG

**Files**: `backend/agents/question_agent.py`, `backend/tools/policy_search.py`
**Dependencies**: Data Models (Component 1), Ingestion Pipeline (Component 2 — ChromaDB must be populated)
**Can be built in parallel with**: Components 3, 7

## Purpose
A self-contained `ToolCallingAgent` (smolagents) that answers **one** compliance requirement by searching the policy corpus and evaluating the best match. One instance is created per parallel worker — there is no shared state between instances.

## Model Configuration

All agents use Gemini 2.5 Pro via smolagents' `OpenAIModel` pointing at Gemini's OpenAI-compatible endpoint:

```python
import os
from smolagents import OpenAIModel

def get_model() -> OpenAIModel:
    return OpenAIModel(
        model_id="gemini-2.5-pro",
        api_key=os.environ["GEMINI_API_KEY"],
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.2,
    )
```

## Agent Definition

```python
from smolagents import ToolCallingAgent

def create_question_agent(model: OpenAIModel) -> ToolCallingAgent:
    return ToolCallingAgent(
        tools=[search_policies_tool, define_term_tool, evaluate_citation_tool],
        model=model,
        max_steps=8,
        name="question_reviewer",
        description="Reviews a single compliance requirement against the policy corpus.",
        instructions=QUESTION_AGENT_PROMPT,
    )
```

## System Prompt (`QUESTION_AGENT_PROMPT`)

```
You are a healthcare compliance reviewer. You are given ONE compliance requirement
(a yes/no question about a Managed Care Plan's Policy & Procedure document).

Your job:
1. Read the requirement carefully. Identify the key regulatory concepts.
2. If the requirement text contains an acronym or program term you are not 100%
   sure about (e.g. "ECM", "POF", "MCP", "SUD", "I/DD", "TBI"), call
   `define_term` with that term BEFORE searching. Use the returned definition
   to formulate better, concept-level search queries.
3. Generate 2-3 search queries that would find the relevant policy section.
   - Use specific regulatory terms from the requirement.
   - Try variations: acronyms, full names (from the definition you looked up),
     and related concepts.
4. Call `search_policies` with each query.
5. From the returned passages, identify the one that best answers the requirement.
6. Determine: does this passage satisfy the requirement? Answer YES, NO, or PARTIAL.
7. Return your answer using `final_answer` with this exact JSON structure:
   {
     "answer": "yes" | "no" | "partial",
     "citation_text": "exact quoted text from the policy",
     "source_file": "path from the passage metadata",
     "page_number": <int>,
     "confidence": <float 0-1>,
     "reasoning": "brief explanation"
   }

IMPORTANT:
- Only cite text that ACTUALLY appears in the retrieved passages. Never fabricate.
- If no passage adequately addresses the requirement, answer "no" with low confidence.
- A "partial" answer means the policy addresses the topic but misses specific details.
- Term definitions from `define_term` are reference material ONLY — never cite them
  as the `citation_text` for a YES/NO/PARTIAL answer. Citations must come from
  `search_policies` results.
```

## Tools

### `search_policies` — ChromaDB Vector Search
The primary retrieval tool exposed to the agent.

```python
from smolagents import tool

@tool
def search_policies(query: str, top_k: int = 10) -> str:
    """
    Search the policy document corpus for passages relevant to a compliance query.

    Args:
        query: Natural language search query about a compliance requirement.
        top_k: Number of results to return (default 10).

    Returns:
        A formatted string of search results, each with the passage text,
        source file, and page number.
    """
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    formatted = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - dist  # ChromaDB returns L2 distance; convert to similarity
        formatted.append(
            f"[Result {i+1}] Score: {score:.3f}\n"
            f"Source: {meta['source_file']}, Page {meta['page_number']}\n"
            f"Text: {doc}\n"
        )
    return "\n---\n".join(formatted)
```

**Design notes**:
- Returns formatted text (not structured data) because smolagents agents work with string tool outputs.
- The agent parses the results in its reasoning to pick the best passage.
- `top_k=10` gives the agent enough options to find a good match.

### `define_term` — Term Definition Lookup
Looks up a term or acronym (e.g. `ECM`, `POF`, `MCP`) in the `document_terms` ChromaDB collection populated by the Compliance Extraction Agent (Component 8). This gives the agent authoritative, in-document definitions so it can expand acronyms and find the right policy passages.

```python
from smolagents import tool

@tool
def define_term(term: str, top_k: int = 3) -> str:
    """
    Look up the definition of a term or acronym extracted from the source document.

    Use this when the compliance requirement contains an unfamiliar acronym
    (e.g. "ECM", "POF") or program-specific term (e.g. "Population of Focus").
    The returned definition can be used to formulate better policy search queries.

    Args:
        term: The term or acronym to define (case-insensitive).
        top_k: Maximum number of candidate definitions to return (default 3).

    Returns:
        A formatted string of matching definitions, each with the term,
        abbreviation, definition text, source file, page number, and section
        heading. Returns "No definition found for '<term>'." if nothing matches.
    """
    collection = get_chroma_client().get_or_create_collection(
        name="document_terms",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        ),
    )

    # 1. Exact metadata match on abbreviation (case-insensitive).
    exact = collection.get(
        where={"$or": [
            {"abbreviation": term.upper()},
            {"abbreviation": term},
            {"term": term},
        ]},
    )
    if exact["ids"]:
        return _format_definitions(exact["metadatas"])

    # 2. Fall back to embedding similarity search.
    results = collection.query(query_texts=[term], n_results=top_k)
    if not results["ids"] or not results["ids"][0]:
        return f"No definition found for '{term}'."
    return _format_definitions(results["metadatas"][0])


def _format_definitions(metadatas: list[dict]) -> str:
    lines = []
    for meta in metadatas:
        abbr = f" ({meta['abbreviation']})" if meta.get("abbreviation") else ""
        section = meta.get("section_heading") or ""
        source = f"{meta.get('source_file', '')}, page {meta.get('page_number', '?')}"
        if section:
            source = f"{section} — {source}"
        lines.append(
            f"[{meta['term']}{abbr}]\n"
            f"Definition: {meta['definition']}\n"
            f"Source: {source}"
        )
    return "\n---\n".join(lines)
```

**Design notes**:
- **Two-stage lookup**: first an exact metadata match on `abbreviation` or `term` (cheap, no embedding roundtrip, and avoids the false positives that pure vector search gives for very short queries like `"POF"`). Only if that misses do we fall back to embedding similarity.
- **Distinct collection**: queries hit `document_terms`, not `policy_documents`, so acronym lookups never compete with or drown in policy passages that happen to mention the acronym.
- **String output**: like `search_policies`, the tool returns a formatted string because `ToolCallingAgent` works with string tool outputs.
- **Empty result is not an error**: the tool returns a human-readable "No definition found" string so the agent can decide how to proceed (typically: fall back to treating the term as a literal search query in `search_policies`).

**Example agent trace**:
```
Requirement: "Does the P&P describe the MCP's approach to identifying ECM POF members?"
→ define_term("ECM")
  ← "[Enhanced Care Management (ECM)]
       Definition: A whole-person, interdisciplinary approach to care that
       addresses the clinical and non-clinical needs of Members with the most
       complex medical and social needs...
       Source: Section II — data/Example Input Doc - Hard.pdf, page 5"
→ define_term("POF")
  ← "[Population of Focus (POF)]
       Definition: The groups of Medi-Cal Members eligible for ECM, such as
       adults and youth experiencing homelessness, justice-involved individuals,
       adults at risk for long-term care institutionalization, etc.
       Source: Section IV — data/Example Input Doc - Hard.pdf, page 10"
→ search_policies("Enhanced Care Management Population of Focus identification process")
→ search_policies("ECM member identification outreach")
→ final_answer({...})
```

### `evaluate_citation` — Citation Evaluation (Optional)
A secondary tool the agent can call for borderline cases. Makes an LLM call to check whether a specific passage satisfies a specific requirement.

```python
@tool
def evaluate_citation(requirement: str, passage: str) -> str:
    """
    Evaluate whether a policy passage satisfies a compliance requirement.

    Args:
        requirement: The compliance question to evaluate.
        passage: The candidate policy text to check against the requirement.

    Returns:
        JSON with: answer (yes/no/partial), confidence (0-1), reasoning.
    """
```

This is optional — the agent can do this evaluation in its own reasoning. But having it as a tool gives the agent a "second opinion" mechanism for difficult cases.

## Running the Agent

```python
async def run_question_agent(requirement: Requirement, model: LiteLLMModel) -> Evaluation:
    agent = create_question_agent(model)
    task = f"""
    Evaluate this compliance requirement against the policy corpus:

    Requirement #{requirement.id}: {requirement.text}
    Reference: {requirement.reference or 'N/A'}
    """
    result = agent.run(task)
    # Parse the agent's final_answer JSON into an Evaluation
    return parse_agent_result(result, requirement.id)
```

**Key**: Each call creates a **new** agent instance. This ensures no memory leakage between questions and makes each worker fully independent.

## Error Handling
- If the agent exceeds `max_steps` (8), catch the error and return a low-confidence "no" evaluation with `needs_human_review=True`.
- If ChromaDB returns no results, the agent should recognize this and answer "no".
- If the agent's output doesn't parse as valid JSON, wrap in a fallback Evaluation.
- If the `document_terms` collection does not exist yet (e.g. the compliance extractor has not populated it), `define_term` returns `"No definition found for '<term>'."` — the agent treats this as a non-fatal signal and proceeds with `search_policies` using the raw term.

## Testing
- **Unit**: Mock ChromaDB, run agent on requirement #1 from the Easy example, verify it returns valid Evaluation JSON.
- **Unit**: `define_term` with a seeded `document_terms` collection:
  - Exact abbreviation hit: `define_term("ECM")` returns the Enhanced Care Management definition.
  - Case variant: `define_term("ecm")` returns the same result.
  - Full-term hit: `define_term("Population of Focus")` returns the POF definition.
  - Miss: `define_term("XYZZY")` returns the `"No definition found"` string.
- **Integration**: With a populated ChromaDB (after ingestion) and a populated `document_terms` collection (after running the compliance extractor on `data/Example Input Doc - Hard.pdf`), run the question agent on a requirement that references an acronym and verify from the trace that:
  1. The agent calls `define_term` for the acronym
  2. Subsequent `search_policies` calls include the expanded full term
  3. The returned Evaluation cites a `policy_documents` passage, not a `document_terms` entry
- **Integration**: With a populated ChromaDB (after ingestion), run on 3-5 known requirements and manually verify answers.
- **Edge cases**: Requirement with very specific jargon, requirement that spans multiple policy areas, requirement whose acronym is not present in `document_terms` (agent should still produce a valid Evaluation via `search_policies`).
