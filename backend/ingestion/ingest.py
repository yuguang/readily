"""Bulk ingestion orchestrator: parse → chunk → upsert into ChromaDB.

PDF parsing and chunking run concurrently via a ThreadPoolExecutor
(controlled by INGEST_MAX_WORKERS).  ChromaDB upserts remain sequential
to avoid write contention on the underlying SQLite store.
"""

import argparse
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Allow running as  python -m backend.ingestion.ingest  from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    INGEST_MAX_WORKERS,
    POLICIES_DIR,
)
from backend.ingestion.chunker import chunk_document
from backend.tools.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)

# ChromaDB upsert batch size — keeps memory usage predictable
_UPSERT_BATCH = 100

# Serialise console output from concurrent workers
_print_lock = threading.Lock()


def _get_collection(chroma_dir: str) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_dir)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _parse_and_chunk(
    folder_name: str, pdf_path: Path
) -> tuple[str, list[dict] | None, str | None]:
    """Parse one PDF and return its chunks.

    Returns ``(source_file, chunks, error_message)``.
    *chunks* is ``None`` when an error or empty-document condition occurs.
    Thread-safe: reads only from the filesystem, no shared mutable state.
    """
    source_file = f"{folder_name}/{pdf_path.name}"
    with _print_lock:
        print(f"  [{folder_name}] {pdf_path.name}", flush=True)

    try:
        pages = parse_pdf(str(pdf_path))
        if not pages:
            return source_file, None, f"No extractable text in {source_file} (skipped)"

        chunks = chunk_document(
            pages,
            source_file=source_file,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        if not chunks:
            return source_file, None, f"No chunks generated from {source_file} (skipped)"

        return source_file, chunks, None

    except Exception as exc:  # pragma: no cover
        msg = f"Error processing {source_file}: {exc}"
        logger.error(msg, exc_info=True)
        return source_file, None, msg


def ingest_all_policies(policies_dir: str, chroma_dir: str) -> Dict:
    """
    Walk *policies_dir*, parse every PDF, chunk it, and upsert into ChromaDB.

    PDF parsing and chunking run concurrently (``INGEST_MAX_WORKERS`` threads).
    ChromaDB upserts remain sequential to avoid write contention.

    Returns a stats dict::

        {
            "total_files":  int,        # successfully processed PDFs
            "total_chunks": int,        # total chunks upserted
            "errors":       list[str],  # per-file warning / error messages
        }
    """
    collection = _get_collection(chroma_dir)
    stats: Dict = {"total_files": 0, "total_chunks": 0, "errors": []}

    policies_path = Path(policies_dir)
    if not policies_path.is_dir():
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    # Collect all (folder_name, pdf_path) tasks up front
    tasks: list[tuple[str, Path]] = [
        (folder_path.name, pdf_path)
        for folder_path in sorted(p for p in policies_path.iterdir() if p.is_dir())
        for pdf_path in sorted(folder_path.glob("*.pdf"))
    ]

    if not tasks:
        return stats

    # ── Parallel parse + chunk ──────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=INGEST_MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(_parse_and_chunk, folder_name, pdf_path): (folder_name, pdf_path)
            for folder_name, pdf_path in tasks
        }

        for future in as_completed(future_to_task):
            source_file, chunks, error = future.result()

            if error:
                is_hard_error = error.startswith("Error processing")
                (logger.error if is_hard_error else logger.warning)(error)
                stats["errors"].append(error)
                continue

            # ── Sequential upsert (thread-safe w.r.t. ChromaDB) ────────────
            for batch_start in range(0, len(chunks), _UPSERT_BATCH):
                batch = chunks[batch_start : batch_start + _UPSERT_BATCH]
                collection.upsert(
                    ids=[
                        f"{c['source_file']}::{c['chunk_index']}"
                        for c in batch
                    ],
                    documents=[c["text"] for c in batch],
                    metadatas=[
                        {
                            "source_file": c["source_file"],
                            "folder": c["folder"],
                            "page_number": c["page_number"],
                            "chunk_index": c["chunk_index"],
                        }
                        for c in batch
                    ],
                )

            stats["total_files"] += 1
            stats["total_chunks"] += len(chunks)

    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingest policy PDFs into ChromaDB vector store."
    )
    parser.add_argument(
        "--policies-dir",
        default=str(POLICIES_DIR),
        help=f"Path to Public Policies folder (default: {POLICIES_DIR})",
    )
    parser.add_argument(
        "--chroma-dir",
        default=str(CHROMA_DIR),
        help=f"Path for ChromaDB persistence (default: {CHROMA_DIR})",
    )
    args = parser.parse_args()

    print(f"Policies dir : {args.policies_dir}")
    print(f"ChromaDB dir : {args.chroma_dir}")
    print("Starting ingestion...\n")

    result = ingest_all_policies(args.policies_dir, args.chroma_dir)

    print()
    print("=" * 50)
    print("Ingestion complete")
    print(f"  Files processed : {result['total_files']}")
    print(f"  Chunks upserted : {result['total_chunks']}")
    if result["errors"]:
        print(f"  Warnings/errors : {len(result['errors'])}")
        for err in result["errors"]:
            print(f"    - {err}")
    print("=" * 50)
