"""Semantic-aware chunking of extracted PDF pages."""

import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries using punctuation heuristics."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    pages: List[Dict],
    source_file: str = "",
    chunk_size: int = 2000,
    overlap: int = 100,
) -> List[Dict]:
    """
    Split document pages into overlapping chunks suitable for embedding.

    Strategy:
    1. Split each page on double-newlines to obtain paragraph units.
    2. Greedily merge adjacent paragraphs into a chunk until *chunk_size* chars
       would be exceeded.
    3. If a single paragraph itself exceeds *chunk_size*, fall back to
       sentence-boundary splitting.
    4. Each chunk is prefixed with the last *overlap* characters of the
       previous chunk to preserve context across boundaries.

    Args:
        pages:       Output of ``parse_pdf`` — list of {page_number, text}.
        source_file: Relative path used to populate chunk metadata, e.g.
                     ``"GG/GG.1508_CEO20250129_v20241231.pdf"``.
        chunk_size:  Target maximum character count per chunk (before overlap).
        overlap:     Number of characters from the previous chunk to prepend.

    Returns:
        List of dicts with keys:
            text, source_file, folder, page_number, chunk_index
    """
    folder = source_file.split("/")[0] if source_file and "/" in source_file else ""

    # ── Step 1: collect (page_number, paragraph_text) pairs ──────────────────
    para_units: List[Tuple[int, str]] = []
    for page in pages:
        pnum = page["page_number"]
        for para in re.split(r"\n\n+", page.get("text", "")):
            para = para.strip()
            if para:
                para_units.append((pnum, para))

    if not para_units:
        return []

    chunks: List[Dict] = []
    chunk_idx = 0
    prev_tail = ""  # last `overlap` chars of the previously emitted chunk

    def _emit(text: str, page_num: int) -> str:
        """Append a completed chunk and return the new prev_tail."""
        nonlocal chunk_idx
        full = (prev_tail + "\n\n" + text).strip() if prev_tail else text.strip()
        if full:
            chunks.append(
                {
                    "text": full,
                    "source_file": source_file,
                    "folder": folder,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                }
            )
            chunk_idx += 1
        return full[-overlap:] if overlap and len(full) >= overlap else full

    # ── Step 2: greedy paragraph merging ─────────────────────────────────────
    i = 0
    while i < len(para_units):
        page_start = para_units[i][0]
        buf: List[str] = []   # paragraph texts accumulated in current chunk
        buf_len = 0            # total char length (separators included)

        while i < len(para_units):
            pnum, para = para_units[i]
            sep_len = 2 if buf else 0  # length of "\n\n" separator

            if len(para) > chunk_size:
                # ── Step 3a: paragraph too big — flush buf, sentence-split ──
                if buf:
                    prev_tail = _emit("\n\n".join(buf), page_start)
                    buf = []
                    buf_len = 0
                    page_start = pnum

                for sent in _split_sentences(para):
                    s_sep = 1 if buf else 0  # space between sentences
                    if buf_len + s_sep + len(sent) <= chunk_size:
                        buf.append(sent)
                        buf_len += s_sep + len(sent)
                    else:
                        if buf:
                            prev_tail = _emit(" ".join(buf), pnum)
                            buf = []
                            buf_len = 0
                        buf = [sent]
                        buf_len = len(sent)

                i += 1
                break  # restart outer loop so prev_tail is honoured

            elif buf_len + sep_len + len(para) <= chunk_size:
                # ── Step 3b: paragraph fits — add to current chunk ───────────
                buf.append(para)
                buf_len += sep_len + len(para)
                i += 1

            else:
                # ── Step 3c: paragraph doesn't fit — flush and start fresh ───
                break

        # Flush whatever remains in the buffer
        if buf:
            prev_tail = _emit("\n\n".join(buf), page_start)

    return chunks
