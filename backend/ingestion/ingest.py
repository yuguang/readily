"""Bulk ingestion orchestrator: parse → chunk → upsert into ChromaDB."""

import argparse
import logging
import sys
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
    POLICIES_DIR,
)
from backend.ingestion.chunker import chunk_document
from backend.tools.pdf_parser import parse_pdf

logger = logging.getLogger(__name__)

# ChromaDB upsert batch size — keeps memory usage predictable
_UPSERT_BATCH = 100


def _get_collection(chroma_dir: str) -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=chroma_dir)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_all_policies(policies_dir: str, chroma_dir: str) -> Dict:
    """
    Walk *policies_dir*, parse every PDF, chunk it, and upsert into ChromaDB.

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

    folder_dirs = sorted(p for p in policies_path.iterdir() if p.is_dir())

    for folder_path in folder_dirs:
        folder_name = folder_path.name
        pdf_files = sorted(folder_path.glob("*.pdf"))

        for pdf_path in pdf_files:
            source_file = f"{folder_name}/{pdf_path.name}"
            print(f"  [{folder_name}] {pdf_path.name}", flush=True)

            try:
                pages = parse_pdf(str(pdf_path))
                if not pages:
                    msg = f"No extractable text in {source_file} (skipped)"
                    logger.warning(msg)
                    stats["errors"].append(msg)
                    continue

                chunks = chunk_document(
                    pages,
                    source_file=source_file,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )
                if not chunks:
                    msg = f"No chunks generated from {source_file} (skipped)"
                    logger.warning(msg)
                    stats["errors"].append(msg)
                    continue

                # Upsert in batches (ID = source_file::chunk_index for idempotency)
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

            except Exception as exc:
                msg = f"Error processing {source_file}: {exc}"
                logger.error(msg, exc_info=True)
                stats["errors"].append(msg)

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
