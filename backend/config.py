import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
POLICIES_DIR = DATA_DIR / "Public Policies"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

UPLOADS_DIR.mkdir(exist_ok=True)

# LLM
# Model ID for smolagents OpenAIModel (OpenAI-compatible endpoint, no provider prefix)
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "gemini-2.5-pro")
# LiteLLM-style model ID for direct litellm.completion() calls (e.g. evaluate_citation tool)
LITELLM_MODEL_ID = os.getenv("LITELLM_MODEL_ID", "gemini/gemini-2.5-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Compliance extraction model — defaults to gpt-4.1-mini (OpenAI).
# Override via env vars to use a different model or provider.
# Set COMPLIANCE_LLM_API_BASE to a non-empty string to route through a
# custom OpenAI-compatible endpoint (e.g. the Gemini base URL above).
COMPLIANCE_LLM_MODEL_ID = os.getenv("COMPLIANCE_LLM_MODEL_ID", "gemini/gemini-flash-latest")  # override with e.g. gpt-4.1, gpt-4.1-mini, gpt-4o-mini
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COMPLIANCE_LLM_API_BASE = os.getenv("COMPLIANCE_LLM_API_BASE", "")  # empty = OpenAI default

# Routing
LONG_DOC_PAGE_THRESHOLD = int(os.getenv("LONG_DOC_PAGE_THRESHOLD", "20"))

# Parallelization
MAX_CONCURRENT_WORKERS = int(os.getenv("MAX_CONCURRENT_WORKERS", "8"))
COMPLIANCE_MAX_CONCURRENT_WORKERS = int(os.getenv("COMPLIANCE_MAX_CONCURRENT_WORKERS", "20"))
WORKER_TIMEOUT_SECONDS = int(os.getenv("WORKER_TIMEOUT_SECONDS", "300"))
INGEST_MAX_WORKERS = int(os.getenv("INGEST_MAX_WORKERS", "4"))

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking — smaller chunks improve retrieval precision and reduce LLM context size.
# Re-run ingestion after changing these values.
CHUNK_SIZE = 300  # chars per chunk (~75 tokens)
CHUNK_OVERLAP = 75  # chars of overlap between adjacent chunks

# RAG
TOP_K = 6  # fewer results → smaller LLM context per step → faster + cheaper
CONFIDENCE_THRESHOLD = 0.7

# ChromaDB
COLLECTION_NAME = "policy_documents"

# Compliance extraction (Component 8)
SECTION_MAX_CHARS = 5000          # Max chars per section before sub-splitting
DEDUP_SIMILARITY_THRESHOLD = 0.90

# Term definition extraction
TERM_COLLECTION_NAME = "document_terms"
TERM_DEFINITION_MIN_LEN = 10      # Minimum chars for a definition body to be kept
TERM_ACRONYM_MAX_LEN = 10         # Maximum chars in an acronym (skips long all-caps phrases)

# Term definition extraction
TERM_COLLECTION_NAME = "document_terms"
TERM_DEFINITION_MIN_LEN = 10      # Minimum chars for a definition body to be kept
TERM_ACRONYM_MAX_LEN = 10         # Maximum chars in an acronym (skips long all-caps phrases)
