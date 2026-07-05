"""
Centralized configuration for the Code RAG Agent.
Change model names, chunk settings, and paths here — all other modules import from this file.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_index")
QUERY_LOG_FILE = str(DATA_DIR / "query_log.jsonl")

# Default codebase to index (override via CLI or UI)
DEFAULT_CODEBASE_PATH = str(PROJECT_ROOT / "sample_codebase")

# ──────────────────────────────────────────────
# Ollama Models
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"                 # alternatives: "codellama:13b", "deepseek-coder:6.7b"
LLM_REQUEST_TIMEOUT = 300.0          # seconds (higher for CPU-only inference)

# ──────────────────────────────────────────────
# Retrieval Settings
# ──────────────────────────────────────────────
TOP_K = 8                            # number of chunks to retrieve per query
CHROMA_COLLECTION_NAME = "codebase"

# If the best retrieved chunk scores below this, the question is likely
# not about the indexed codebase and we fall back to general-knowledge mode.
RELEVANCE_THRESHOLD = 0.35

# Phrases in the RAG answer that signal "no relevant code found" --
# if the answer contains any of these AND scores are low, fall back.
NOT_FOUND_PHRASES = [
    "could not find the answer",
    "not present in the provided code",
    "not found in the provided code",
    "no information available",
    "not available in the code context",
    "cannot determine from the provided",
]

# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

# Used when the question IS about the indexed codebase (RAG mode)
SYSTEM_PROMPT = """\
You are a senior software engineer answering questions about a Python codebase.
You MUST answer ONLY based on the code context provided below.
If the answer is not present in the provided code, say:
"I could not find the answer in the provided code context."

For every claim you make, cite the source file and function/class name.
Be precise, technical, and concise.\
"""

# Used when the question is general knowledge (direct LLM mode)
GENERAL_SYSTEM_PROMPT = """\
You are a knowledgeable AI assistant. Answer the user's question clearly,
accurately, and concisely. You can answer questions on any topic including
programming, science, math, history, and general knowledge.
If you are unsure, say so rather than guessing.\
"""
