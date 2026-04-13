# Component 5: Parallel Dispatcher + Batch Critic

**Files**: `backend/agents/dispatcher.py`, `backend/agents/critic.py`
**Dependencies**: Data Models (Component 1), Question Agent (Component 4)
**Can be built in parallel with**: Components 2, 3, 7 (stub the Question Agent)

## Purpose
Orchestrates the parallel execution of N Question Agents (one per requirement) using `asyncio`, streams results via SSE, and runs a batch critic pass on low-confidence results.

## Parallelization Pattern

### Why Parallelize
Each of the 64 requirements is fully independent — no shared state, no cross-question dependencies. This makes it an ideal parallelization target. Without parallelization, 64 sequential agent runs (each ~3-5 LLM calls) would take ~10+ minutes. With 8 concurrent workers, this drops to ~1-2 minutes.

### Why Application-Level asyncio (Not smolagents `managed_agents`)
smolagents' `managed_agents` run sequentially within a single agent's ReAct loop. They're designed for delegation, not parallel execution. We need true concurrency, so we use Python's `asyncio` with a semaphore to bound concurrent LLM calls.

## Dispatcher Implementation

```python
import asyncio
from collections.abc import AsyncGenerator

MAX_CONCURRENT_WORKERS = 8

async def dispatch_review(
    requirements: list[Requirement],
    on_result: Callable[[Evaluation], None] | None = None,
) -> list[Evaluation]:
    """
    Fan out requirements to parallel workers. Calls on_result callback
    as each worker completes (for SSE streaming).
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)
    results: list[Evaluation] = []
    errors: list[tuple[int, str]] = []

    async def worker(req: Requirement) -> Evaluation | None:
        async with semaphore:
            try:
                evaluation = await asyncio.to_thread(
                    run_question_agent_sync, req
                )
                return evaluation
            except Exception as e:
                errors.append((req.id, str(e)))
                return make_error_evaluation(req.id, str(e))

    tasks = [asyncio.create_task(worker(r)) for r in requirements]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
            if on_result:
                on_result(result)

    return results
```

### SSE Streaming Generator
For the FastAPI SSE endpoint:

```python
async def stream_review(requirements: list[Requirement]) -> AsyncGenerator[SSEEvent, None]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_WORKERS)
    completed = 0
    total = len(requirements)

    async def worker(req: Requirement) -> Evaluation:
        async with semaphore:
            return await asyncio.to_thread(run_question_agent_sync, req)

    tasks = [asyncio.create_task(worker(r)) for r in requirements]

    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            completed += 1
            yield SSEEvent(
                event="evaluation",
                data={"evaluation": result.model_dump(), "progress": completed, "total": total},
            )
        except Exception as e:
            completed += 1
            yield SSEEvent(event="error", data={"error": str(e), "progress": completed, "total": total})

    # Trigger critic pass
    yield SSEEvent(event="progress", data={"status": "running_critic"})
```

### `asyncio.to_thread` for smolagents
smolagents agents are synchronous (they use blocking HTTP calls). We wrap them in `asyncio.to_thread` to run them in a thread pool, allowing true parallel execution within the async event loop.

## Batch Critic (Reflection Pattern)

### Why Batch Instead of Per-Question
Running a separate critic for each of 64 questions doubles the LLM calls (128 total). Instead, we only review the ~10-15% of results with `confidence < 0.7`, in a single batched LLM call.

### Implementation

```python
async def run_batch_critic(evaluations: list[Evaluation], requirements: list[Requirement]) -> list[Evaluation]:
    """
    Review low-confidence evaluations and update their needs_human_review flag.
    """
    low_conf = [e for e in evaluations if e.confidence < CONFIDENCE_THRESHOLD]

    if not low_conf:
        return evaluations

    # Build a batch prompt
    review_items = []
    for e in low_conf:
        req = next(r for r in requirements if r.id == e.requirement_id)
        review_items.append({
            "requirement_id": e.requirement_id,
            "requirement_text": req.text,
            "agent_answer": e.answer,
            "citation": e.citation_text,
            "source": e.source_file,
            "confidence": e.confidence,
        })

    prompt = f"""
    You are a quality reviewer for a compliance audit. Review these {len(review_items)}
    low-confidence findings and determine if each should be flagged for human review.

    For each item, respond with:
    - requirement_id
    - verified_answer: yes/no/partial (your assessment)
    - needs_human_review: true/false
    - reason: brief explanation

    Items to review:
    {json.dumps(review_items, indent=2)}
    """

    # Single LLM call for the whole batch (using same Gemini model)
    from smolagents import OpenAIModel
    model = OpenAIModel(
        model_id="gemini-2.5-pro",
        api_key=os.environ["GEMINI_API_KEY"],
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.2,
    )
    response = model([{"role": "user", "content": prompt}])
    critic_results = parse_critic_response(response.content)

    # Update evaluations
    for cr in critic_results:
        eval_obj = next(e for e in evaluations if e.requirement_id == cr["requirement_id"])
        eval_obj.needs_human_review = cr["needs_human_review"]
        if cr["verified_answer"] != eval_obj.answer:
            eval_obj.answer = cr["verified_answer"]
            eval_obj.reasoning += f" [Critic: {cr['reason']}]"

    return evaluations
```

## Configuration
In `backend/config.py`:

```python
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "8"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Model (shared config — each worker creates its own instance)
LLM_MODEL_ID = "gemini-2.5-pro"  # via OpenAIModel + Gemini OpenAI-compatible endpoint
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
```

## Error Handling
- **Worker timeout**: If a single worker takes > 60s, cancel it and return an error evaluation.
- **Rate limiting**: The semaphore (default 8) prevents hitting LLM API rate limits. Can be tuned via env var.
- **Partial failure**: If some workers fail, the rest still complete. Failed items get `needs_human_review=True`.

## Testing
- **Unit**: Mock `run_question_agent_sync`, verify dispatcher runs N workers with correct semaphore bounds.
- **Unit**: Test batch critic with mock LLM response, verify `needs_human_review` flags are set correctly.
- **Integration**: Run dispatcher on 5 requirements with real ChromaDB, verify SSE events are emitted in completion order (not sequential).
- **Load test**: Verify 64 concurrent tasks with semaphore=8 completes without deadlock.
