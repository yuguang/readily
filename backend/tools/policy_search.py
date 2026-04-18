"""
Policy search tools for the Question Agent.

Provides ChromaDB vector search and LLM-based citation evaluation tools,
exposed to the ToolCallingAgent via the smolagents @tool decorator.
"""

from __future__ import annotations

import functools

import chromadb
import openai
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from smolagents import tool

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_BASE,
    GEMINI_API_KEY,
    LLM_MODEL_ID,
    TERM_COLLECTION_NAME,
)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@functools.lru_cache(maxsize=1)
def _get_openai_client() -> openai.OpenAI:
    """Return a cached OpenAI client pointed at the Gemini-compatible endpoint."""
    return openai.OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_API_BASE)


def get_chroma_collection() -> chromadb.Collection:
    """Return the ChromaDB collection containing ingested policy documents."""
    return _get_chroma_client().get_collection(name=COLLECTION_NAME)


def _get_term_collection() -> chromadb.Collection:
    """Return (or lazily create) the ChromaDB collection for term definitions."""
    return _get_chroma_client().get_or_create_collection(
        name=TERM_COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        ),
    )


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
    collection = _get_term_collection()

    # 1. Exact metadata match on abbreviation (case-insensitive) or full term.
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
    """Format a list of term-definition metadata dicts into a human-readable string."""
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

    client = _get_openai_client()
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()
