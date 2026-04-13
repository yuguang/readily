# Component 2: Ingestion Pipeline

**Files**: `backend/ingestion/ingest.py`, `backend/ingestion/chunker.py`, `backend/tools/pdf_parser.py`
**Dependencies**: Data Models (Component 1)
**Can be built in parallel with**: Components 3, 4, 7

## Purpose
One-time batch process that parses all 373 policy PDFs from `data/Public Policies/`, chunks them, generates embeddings, and stores them in a ChromaDB vector store. This must run before any review can happen.

## Input
```
data/Public Policies/
├── AA/   (19 PDFs)
├── CMC/  (4 PDFs)
├── DD/   (11 PDFs)
├── EE/   (12 PDFs)
├── FF/   (24 PDFs)
├── GA/   (5 PDFs)
├── GG/   (144 PDFs)   ← largest folder
├── HH/   (47 PDFs)
├── MA/   (69 PDFs)
└── PA/   (38 PDFs)
```

## Output
A ChromaDB collection (`policy_documents`) with embeddings and metadata for each chunk.

## Subcomponents

### `pdf_parser.py` — PDF Text Extraction
Uses PyMuPDF (`fitz`) to extract text from PDFs.

```python
def parse_pdf(pdf_path: str) -> list[dict]:
    """Returns list of { page_number: int, text: str } for each page."""
```

**Edge cases**:
- Scanned PDFs with no text layer → log warning, skip (or future: OCR)
- PDFs with tables → extract as best-effort text
- Multi-column layouts → PyMuPDF handles these reasonably

### `chunker.py` — Semantic-Aware Chunking
Splits extracted page text into chunks suitable for embedding.

**Strategy**: Paragraph-boundary chunking with fallback to character-level splitting.
1. Split text on double-newlines (paragraph boundaries)
2. Greedily merge adjacent paragraphs until chunk reaches ~2000 chars (~500 tokens)
3. If a single paragraph exceeds the limit, split on sentence boundaries
4. Overlap: include the last 100 chars of the previous chunk as prefix

```python
def chunk_document(pages: list[dict], chunk_size: int = 2000, overlap: int = 100) -> list[dict]:
    """Returns list of { text, source_file, page_number, chunk_index }."""
```

**Metadata per chunk**:
- `source_file`: Relative path like `"GG/GG.1508_CEO20250129_v20241231.pdf"`
- `folder`: Top-level folder name like `"GG"`
- `page_number`: Which page(s) this chunk spans (use the first page)
- `chunk_index`: Sequential index within the document

### `ingest.py` — Bulk Ingestion Orchestrator
Walks `data/Public Policies/`, parses each PDF, chunks it, and upserts into ChromaDB.

```python
def ingest_all_policies(policies_dir: str, chroma_dir: str) -> dict:
    """
    Returns stats: { total_files, total_chunks, errors: list[str] }
    """
```

**ChromaDB Setup**:
- Collection name: `policy_documents`
- Embedding function: `sentence-transformers/all-MiniLM-L6-v2` (via ChromaDB's built-in `SentenceTransformerEmbeddingFunction`)
- Persistence directory: `chroma_db/`

**Performance**: 373 PDFs can be processed sequentially (no need for parallel ingestion — it's a one-time operation). Expect ~5-10 minutes for the full corpus.

**Idempotency**: Use the chunk's `source_file + chunk_index` as the ChromaDB document ID. Re-running ingestion overwrites existing chunks.

## CLI Entry Point
```bash
python -m backend.ingestion.ingest
```

Should print progress and final stats.

## Testing
- Unit test: `parse_pdf` on a known PDF, assert page count and text content
- Unit test: `chunk_document` on a known text, assert chunk count, overlap, and metadata
- Integration test: ingest a small subset (e.g., `GA/` with 5 PDFs), verify ChromaDB collection count
- Edge case: empty PDF, corrupted PDF, PDF with only images
