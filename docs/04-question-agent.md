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
        tools=[search_policies_tool, evaluate_citation_tool],
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
2. Generate 2-3 search queries that would find the relevant policy section.
   - Use specific regulatory terms from the requirement.
   - Try variations: acronyms, full names, related concepts.
3. Call `search_policies` with each query.
4. From the returned passages, identify the one that best answers the requirement.
5. Determine: does this passage satisfy the requirement? Answer YES, NO, or PARTIAL.
6. Return your answer using `final_answer` with this exact JSON structure:
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

## Testing
- **Unit**: Mock ChromaDB, run agent on requirement #1 from the Easy example, verify it returns valid Evaluation JSON.
- **Integration**: With a populated ChromaDB (after ingestion), run on 3-5 known requirements and manually verify answers.
- **Edge cases**: Requirement with very specific jargon, requirement that spans multiple policy areas.
