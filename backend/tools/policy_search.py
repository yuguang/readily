"""
Policy search tools for the Question Agent.

Provides ChromaDB vector search and LLM-based citation evaluation tools,
exposed to the ToolCallingAgent via the smolagents @tool decorator.
"""

from __future__ import annotations

import functools

import chromadb
import litellm
from smolagents import tool

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    GEMINI_API_KEY,
    LITELLM_MODEL_ID,
)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_chroma_collection() -> chromadb.Collection:
    """Return the ChromaDB collection containing ingested policy documents."""
    return _get_chroma_client().get_collection(name=COLLECTION_NAME)


# ---------------------------------------------------------------------------
# smolagents tools
# ---------------------------------------------------------------------------


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
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        score = 1 - dist  # ChromaDB L2 distance → similarity
        formatted.append(
            f"[Result {i + 1}] Score: {score:.3f}\n"
            f"Source: {meta['source_file']}, Page {meta['page_number']}\n"
            f"Text: {doc}\n"
        )

    if not formatted:
        return "No relevant passages found in the policy corpus."

    return "\n---\n".join(formatted)


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
    prompt = (
        "You are a healthcare compliance reviewer.\n\n"
        f"Compliance requirement:\n{requirement}\n\n"
        f"Policy passage:\n{passage}\n\n"
        "Does this passage satisfy the requirement? "
        "Respond with JSON only — no markdown fences:\n"
        '{"answer": "yes"|"no"|"partial", "confidence": <float 0-1>, '
        '"reasoning": "<brief explanation>"}'
    )

    response = litellm.completion(
        model=LITELLM_MODEL_ID,
        api_key=GEMINI_API_KEY,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()
