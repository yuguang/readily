"""
On-disk artifact cache for the Compliance Extraction pipeline (Component 8).

Stable per-PDF artifacts that are deterministic functions of the input bytes
are cached under ``EXTRACTION_CACHE_DIR/<sha256>/`` so that re-uploads of the
same document skip the expensive PyMuPDF pass and the header/footer scan.

Cached artifacts:

- ``structured_text.txt``   — output of ``parse_pdf_with_structure``
- ``hf_signatures.json``    — repeated header/footer signatures
- ``sections.pickle``       — segmented ``DocumentSection`` list
- ``requirements/<hash>.json`` — per-section extraction results keyed by
                                  a hash of ``(section_text, prompt_version)``

The cache is a best-effort optimisation: any read/write error is logged and
the caller falls back to regenerating the artifact.  Cache invalidation is
handled by bumping ``CACHE_SCHEMA_VERSION`` or the ``prompt_version`` passed
in by callers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.config import EXTRACTION_CACHE_DIR, EXTRACTION_CACHE_ENABLED

if TYPE_CHECKING:
    from backend.tools.document_segmenter import DocumentSection
    from backend.tools.header_footer_stripper import HeaderFooterFilter

logger = logging.getLogger(__name__)


# Bump this to invalidate every cached artifact across the project (e.g. when
# the structured-text format or segmentation algorithm changes).
CACHE_SCHEMA_VERSION = "v1"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def hash_pdf_bytes(pdf_path: str | Path) -> str:
    """Return a stable SHA-256 hex digest of the raw bytes of *pdf_path*."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str, *extra: str) -> str:
    """Return a stable SHA-256 hex digest for a text blob + optional tags."""
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="replace"))
    for tag in extra:
        h.update(b"\x00")
        h.update(tag.encode("utf-8", errors="replace"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache directory helpers
# ---------------------------------------------------------------------------


def _cache_dir_for(pdf_hash: str) -> Path:
    base = Path(EXTRACTION_CACHE_DIR) / CACHE_SCHEMA_VERSION / pdf_hash
    base.mkdir(parents=True, exist_ok=True)
    return base


def _enabled() -> bool:
    return bool(EXTRACTION_CACHE_ENABLED)


# ---------------------------------------------------------------------------
# Structured text
# ---------------------------------------------------------------------------


def get_structured_text(pdf_hash: str) -> str | None:
    """Return cached ``structured_text`` for *pdf_hash*, or ``None`` on miss."""
    if not _enabled():
        return None
    path = _cache_dir_for(pdf_hash) / "structured_text.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cache read failed for structured_text %s: %s", path, exc)
        return None


def put_structured_text(pdf_hash: str, structured_text: str) -> None:
    """Persist ``structured_text`` for later reuse."""
    if not _enabled():
        return
    path = _cache_dir_for(pdf_hash) / "structured_text.txt"
    try:
        path.write_text(structured_text, encoding="utf-8")
    except OSError as exc:
        logger.warning("Cache write failed for structured_text %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Header/footer signatures
# ---------------------------------------------------------------------------


def get_hf_signatures(pdf_hash: str) -> frozenset[str] | None:
    """Return cached header/footer signatures, or ``None`` on miss."""
    if not _enabled():
        return None
    path = _cache_dir_for(pdf_hash) / "hf_signatures.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return frozenset(str(s) for s in data)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cache read failed for hf_signatures %s: %s", path, exc)
    return None


def put_hf_signatures(pdf_hash: str, signatures: frozenset[str]) -> None:
    """Persist header/footer signatures keyed by *pdf_hash*."""
    if not _enabled():
        return
    path = _cache_dir_for(pdf_hash) / "hf_signatures.json"
    try:
        path.write_text(json.dumps(sorted(signatures)), encoding="utf-8")
    except OSError as exc:
        logger.warning("Cache write failed for hf_signatures %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Segmented sections
# ---------------------------------------------------------------------------


def get_sections(pdf_hash: str) -> list[DocumentSection] | None:
    """Return cached segmented sections, or ``None`` on miss."""
    if not _enabled():
        return None
    path = _cache_dir_for(pdf_hash) / "sections.pickle"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError, AttributeError) as exc:
        # AttributeError covers the case where DocumentSection's schema moved.
        logger.warning("Cache read failed for sections %s: %s", path, exc)
        return None


def put_sections(pdf_hash: str, sections: list[DocumentSection]) -> None:
    """Persist segmented sections keyed by *pdf_hash*."""
    if not _enabled():
        return
    path = _cache_dir_for(pdf_hash) / "sections.pickle"
    try:
        with open(path, "wb") as f:
            pickle.dump(sections, f)
    except (OSError, pickle.PickleError) as exc:
        logger.warning("Cache write failed for sections %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Per-section extraction results
# ---------------------------------------------------------------------------


def _requirements_dir(pdf_hash: str) -> Path:
    d = _cache_dir_for(pdf_hash) / "requirements"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_section_requirements(
    pdf_hash: str,
    section_hash: str,
) -> list[dict[str, Any]] | None:
    """Return cached requirement dicts for a specific section, or ``None``."""
    if not _enabled():
        return None
    path = _requirements_dir(pdf_hash) / f"{section_hash}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cache read failed for requirements %s: %s", path, exc)
    return None


def put_section_requirements(
    pdf_hash: str,
    section_hash: str,
    requirements: list[dict[str, Any]],
) -> None:
    """Persist requirement dicts for a specific section."""
    if not _enabled():
        return
    path = _requirements_dir(pdf_hash) / f"{section_hash}.json"
    try:
        path.write_text(json.dumps(requirements), encoding="utf-8")
    except OSError as exc:
        logger.warning("Cache write failed for requirements %s: %s", path, exc)
