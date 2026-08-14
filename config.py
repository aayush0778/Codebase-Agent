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
LLM_MODEL = "devstral-small-2:24b"         # Production model
LLM_REQUEST_TIMEOUT = 300.0          # seconds (higher for CPU-only inference)
LLM_CONTEXT_WINDOW = 8192            # Ollama context window size
MAX_HISTORY_TURNS = 4                # Number of past Q&A exchanges to include in history

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

# ──────────────────────────────────────────────
# Style Profiles
# ──────────────────────────────────────────────
DEFAULT_STYLE_PROFILE = "concise"

STYLE_PROFILES = {
    "concise": {
        "name": "⚡ Concise",
        "description": "Short, direct answers with minimal explanation.",
        "prompt_modifier": (
            "\nKeep your answer concise and to the point. "
            "Use bullet points where appropriate. Avoid unnecessary elaboration."
        ),
    },
    "detailed": {
        "name": "📖 Detailed",
        "description": "Comprehensive explanations with examples.",
        "prompt_modifier": (
            "\nProvide a thorough, detailed explanation. "
            "Include code examples where helpful. Explain the reasoning "
            "behind design decisions and patterns you identify."
        ),
    },
    "tutorial": {
        "name": "🎓 Tutorial",
        "description": "Step-by-step explanations for learning.",
        "prompt_modifier": (
            "\nExplain step-by-step as if teaching a junior developer. "
            "Define technical terms. Use numbered steps and simple language. "
            "Include 'why' explanations, not just 'what'."
        ),
    },
    "review": {
        "name": "🔍 Code Review",
        "description": "Focus on quality, bugs, and improvements.",
        "prompt_modifier": (
            "\nAnalyze the code like a senior code reviewer. "
            "Point out potential bugs, anti-patterns, performance issues, "
            "and suggest improvements. Be constructive but thorough."
        ),
    },
}
