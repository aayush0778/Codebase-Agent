"""
Query Engine -- Phase 3 (Hybrid Mode with Heuristic Quality Analysis)
Loads the persisted ChromaDB index, retrieves relevant code chunks for a question,
and generates a grounded answer using the local LLM via Ollama.

If the question is not about the indexed codebase (low relevance scores or
the RAG answer says "not found"), falls back to a direct general-knowledge
LLM call so the user always gets a useful answer.

Quality evaluation calculates:
  - Retrieval Relevance (bounded [0, 100])
  - Grounding (evidence from citations & symbol overlap)
  - Coverage (volume + file diversity across Top-K)
  - Specificity (deterministic query clarity evaluation)
  - Overall Heuristic Quality & Metadata-Driven Suggestions
"""

import logging
import json
import re
import datetime
from pathlib import Path
import urllib.request
import urllib.error

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    TOP_K,
    CHROMA_COLLECTION_NAME,
    SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    QUERY_LOG_FILE,
    RELEVANCE_THRESHOLD,
    NOT_FOUND_PHRASES,
    LLM_CONTEXT_WINDOW,
    MAX_HISTORY_TURNS,
    STYLE_PROFILES,
    DEFAULT_STYLE_PROFILE,
)

logger = logging.getLogger(__name__)

# Custom prompt template that forces the model to answer only from context
QA_PROMPT_TEMPLATE = PromptTemplate(
    """\
{system_prompt}

--- CODE CONTEXT START ---
{context_str}
--- CODE CONTEXT END ---

Question: {query_str}

Answer (cite file and function/class names):\
"""
)


# ──────────────────────────────────────────────
# Shared LLM instance (reused across RAG + general)
# ──────────────────────────────────────────────
_llm_instance = None


def _get_llm(llm_model=None):
    """Return a cached Ollama LLM instance."""
    global _llm_instance
    if llm_model is None:
        llm_model = LLM_MODEL
    if _llm_instance is None or _llm_instance.model != llm_model:
        _llm_instance = Ollama(
            model=llm_model, 
            request_timeout=LLM_REQUEST_TIMEOUT,
            context_window=LLM_CONTEXT_WINDOW,
        )
    return _llm_instance


def load_query_engine(persist_dir=None, top_k=None, llm_model=None):
    """Load the persisted Chroma index and return a query engine.

    Args:
        persist_dir: Path to the persisted Chroma index directory.
        top_k: Number of chunks to retrieve per query.
        llm_model: Ollama model name for generation.

    Returns:
        A LlamaIndex query engine ready to answer questions.
    """
    if persist_dir is None:
        persist_dir = CHROMA_PERSIST_DIR
    if top_k is None:
        top_k = TOP_K
    if llm_model is None:
        llm_model = LLM_MODEL

    logger.info("Loading index from %s (top_k=%d, model=%s)", persist_dir, top_k, llm_model)

    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    llm = _get_llm(llm_model)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        text_qa_template=QA_PROMPT_TEMPLATE.partial_format(system_prompt=SYSTEM_PROMPT),
    )

    logger.info("Query engine loaded successfully.")
    return query_engine


# ──────────────────────────────────────────────
# Relevance & Quality Detection Helpers
# ──────────────────────────────────────────────

def _get_best_score(source_nodes):
    """Return the highest similarity score among retrieved nodes."""
    scores = [n.score for n in source_nodes if n.score is not None]
    return max(scores) if scores else 0.0


def _answer_says_not_found(answer_text):
    """Check if the RAG answer contains a 'not found' phrase."""
    lower = answer_text.lower()
    return any(phrase in lower for phrase in NOT_FOUND_PHRASES)


def _estimate_tokens(text):
    """Rough token estimation: ~4 chars per token for English/code."""
    return len(text) // 4


# ──────────────────────────────────────────────
# Intelligent Context Compression
# ──────────────────────────────────────────────

def _compress_context(history_text, question, style_modifier, max_tokens):
    """Intelligently compress conversation context when exceeding budget.

    Preserves high-value code structures:
      1. imports
      2. class declarations & signatures
      3. function definitions & signatures
      4. decorators & docstrings
      5. recent conversational turns

    Returns:
        (compressed_text, metadata_dict)
    """
    question_tokens = _estimate_tokens(question)
    style_tokens = _estimate_tokens(style_modifier)
    available_tokens = int(max_tokens * 0.7) - question_tokens - style_tokens

    if available_tokens <= 0:
        return "", {
            "was_compressed": True,
            "original_chars": len(history_text),
            "compressed_chars": 0,
            "preserved_symbols": [],
        }

    original_chars = len(history_text)
    history_tokens = _estimate_tokens(history_text)

    if history_tokens <= available_tokens:
        return history_text, {
            "was_compressed": False,
            "original_chars": original_chars,
            "compressed_chars": original_chars,
            "preserved_symbols": [],
        }

    # Extract high-value preserved symbols from history code snippets
    preserved_symbols = []
    symbol_matches = re.findall(r'(?:def|class)\s+([A-Za-z0-9_]+)', history_text)
    if symbol_matches:
        preserved_symbols = list(dict.fromkeys(symbol_matches))[:8]

    # Compress turns structurally: keep newest turns intact, compress older turns
    lines = history_text.split("\n")
    compressed_lines = []
    
    # Process from newest to oldest
    for line in reversed(lines):
        line_tokens = _estimate_tokens("\n".join(compressed_lines + [line]))
        if line_tokens <= available_tokens:
            compressed_lines.insert(0, line)
        else:
            # Preserve import or signature lines if possible
            if line.strip().startswith(("import ", "from ", "def ", "class ", "@", "class:")):
                if _estimate_tokens("\n".join(compressed_lines + [line])) <= available_tokens:
                    compressed_lines.insert(0, line)

    compressed_text = "\n".join(compressed_lines)
    compressed_chars = len(compressed_text)

    return compressed_text, {
        "was_compressed": True,
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "preserved_symbols": preserved_symbols,
    }


# ──────────────────────────────────────────────
# Deterministic Response Quality Metrics
# ──────────────────────────────────────────────

def _compute_specificity(question: str) -> float:
    """Evaluate prompt specificity deterministically (0-100).
    
    Higher score: mentions specific identifiers, filenames, functions, classes, code tokens.
    Lower score: short, vague keywords ('it', 'this', 'that', 'something', 'explain').
    """
    if not question or not question.strip():
        return 0.0

    text = question.strip()
    words = re.findall(r'\b[A-Za-z0-9_.]+\b', text)
    word_count = len(words)

    score = 40.0  # Base score

    # Length signals
    if word_count >= 5:
        score += min((word_count - 4) * 3.5, 20.0)
    elif word_count <= 2:
        score -= 20.0

    # Code identifiers (snake_case, camelCase, dot.notation, file extensions)
    has_identifier = bool(re.search(r'\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b', text))
    has_file_ext = bool(re.search(r'\b[A-Za-z0-9_-]+\.(?:py|js|ts|json|md|html|css|cpp|h|rs|go)\b', text, re.IGNORECASE))
    has_method_call = bool(re.search(r'\b[A-Za-z0-9_]+\.[A-Za-z0-9_]+(?:\(\))?|\b[A-Za-z0-9_]+\(\)', text))
    has_code_keywords = bool(re.search(r'\b(class|def|function|method|module|variable|import|return|parameter|attribute|decorator)\b', text, re.IGNORECASE))

    if has_file_ext:
        score += 20.0
    if has_method_call or has_identifier:
        score += 20.0
    if has_code_keywords:
        score += 10.0

    # Vague words penalty
    vague_words = {"it", "this", "that", "something", "everything", "stuff", "thing", "code", "app"}
    vague_count = sum(1 for w in words if w.lower() in vague_words)
    if vague_count >= 2 or (word_count <= 3 and vague_count >= 1):
        score -= 25.0

    # Generic one-word prompts like "Explain", "Help", "Run"
    if word_count <= 2 and not (has_identifier or has_file_ext or has_method_call):
        score = min(score, 25.0)

    return float(max(0.0, min(100.0, score)))


def _compute_coverage(source_nodes, top_k: int) -> float:
    """Evaluate retrieval coverage and diversity (0-100)."""
    if not source_nodes:
        return 0.0

    source_count = len(source_nodes)
    # Ratio against configured TOP_K
    base_ratio = min(source_count / max(top_k, 1), 1.0) * 80.0

    # File diversity factor (avoid over-crediting 8 identical chunks from one line)
    unique_files = set()
    for n in source_nodes:
        fname = n.metadata.get("file", "unknown") if hasattr(n, "metadata") else "unknown"
        unique_files.add(fname)

    diversity_bonus = min(len(unique_files) * 6.0, 20.0)
    total_coverage = base_ratio + diversity_bonus

    return float(max(0.0, min(100.0, total_coverage)))


def _compute_grounding(rag_answer: str, sources: list, best_score: float, mode: str) -> float:
    """Evaluate grounding evidence of the answer in retrieved code context (0-100)."""
    if mode == "general" or not sources:
        return 0.0

    # Base from clamped retrieval similarity
    base_grounding = min(max(best_score * 100.0, 0.0), 100.0) * 0.65

    answer_lower = rag_answer.lower()
    citation_bonus = 0.0

    for s in sources:
        fname = Path(s.get("file", "")).name.lower()
        symbol_name = s.get("name", "").lower()
        if fname and fname in answer_lower:
            citation_bonus += 10.0
        if symbol_name and symbol_name != "module" and symbol_name in answer_lower:
            citation_bonus += 8.0

    citation_bonus = min(citation_bonus, 35.0)
    grounding = base_grounding + citation_bonus

    # Penalty if answer expresses uncertainty or missing symbols
    if _answer_says_not_found(rag_answer):
        grounding = max(0.0, grounding - 40.0)

    return float(max(0.0, min(100.0, grounding)))


def _compute_quality_info(question: str, rag_answer: str, sources: list, mode: str, best_score: float, source_nodes: list) -> dict:
    """Compute the complete Response Confidence Indicators quality object."""
    # 1. Retrieval relevance (0-100 bounded)
    retrieval_relevance = float(max(0.0, min(100.0, round(best_score * 100.0, 1)))) if mode == "code" else 0.0

    # 2. Dimensions
    specificity = round(_compute_specificity(question), 1)
    coverage = round(_compute_coverage(source_nodes, TOP_K), 1) if mode == "code" else 0.0
    grounding = round(_compute_grounding(rag_answer, sources, best_score, mode), 1) if mode == "code" else 0.0

    # 3. Overall Heuristic Score: 0.45 * grounding + 0.30 * coverage + 0.25 * specificity
    if mode == "code":
        overall_score = round(0.45 * grounding + 0.30 * coverage + 0.25 * specificity, 1)
    else:
        # General knowledge mode
        overall_score = round(0.40 * specificity, 1)

    overall_score = max(0.0, min(100.0, overall_score))

    # 4. Overall Level Categorization
    if overall_score >= 75.0:
        overall = "High"
    elif overall_score >= 55.0:
        overall = "Good"
    elif overall_score >= 35.0:
        overall = "Fair"
    else:
        overall = "Low"

    # 5. Context-aware, metadata-driven suggestions
    suggestions = []
    if mode == "general":
        suggestions.append("Query did not match indexed codebase entities — mention specific files (.py), classes, or functions to query local code.")
    else:
        if overall_score >= 75.0 and grounding >= 70.0:
            # High-confidence grounded responses do not need boilerplate warnings
            suggestions = []
        else:
            candidate_files = list(dict.fromkeys([Path(s.get("file", "")).name for s in sources if s.get("file")]))[:3]
            candidate_symbols = list(dict.fromkeys([s.get("name") for s in sources if s.get("name") and s.get("name") not in ("module", "unknown")]))[:3]

            if specificity < 55.0:
                if candidate_symbols:
                    suggestions.append(f"Target specific symbols such as `{', '.join(candidate_symbols)}` in your next question.")
                elif candidate_files:
                    suggestions.append(f"Specify target files like `{', '.join(candidate_files)}` to focus the search.")
                else:
                    suggestions.append("Clarify the exact component, class, or method behavior you want to understand.")

            if grounding < 55.0 and sources:
                answer_lower = rag_answer.lower()
                unreferenced_symbols = [s for s in candidate_symbols if s.lower() not in answer_lower]
                if unreferenced_symbols:
                    suggestions.append(f"Ask specifically about `{', '.join(unreferenced_symbols[:2])}` to pull deeper implementation logic.")
                else:
                    suggestions.append("Ask for specific code snippets, method signatures, or input/output flows to strengthen grounding.")

            if coverage < 55.0:
                if len(candidate_files) == 1:
                    suggestions.append(f"Only `{candidate_files[0]}` was retrieved — mention related modules if this feature spans multiple files.")
                else:
                    suggestions.append("Ask about a narrower submodule or check that all related files were indexed.")

            if retrieval_relevance < 55.0:
                suggestions.append("Vector match confidence is modest — use exact function names or class identifiers.")

    return {
        "grounding": grounding,
        "coverage": coverage,
        "specificity": specificity,
        "overall_score": overall_score,
        "overall": overall,
        "source_count": len(sources),
        "suggestions": suggestions,
        "retrieval_relevance": retrieval_relevance,
    }


# ──────────────────────────────────────────────
# General-knowledge fallback (direct LLM call)
# ──────────────────────────────────────────────

def _ask_general(question, history_text=""):
    """Send the question directly to the LLM without any code context.

    Returns:
        The answer string from the LLM.
    """
    llm = _get_llm()
    context_block = f"\nRecent conversation:\n{history_text}\n" if history_text else ""
    prompt = f"""{GENERAL_SYSTEM_PROMPT}
{context_block}
Question: {question}

Answer:"""

    logger.info("Falling back to general-knowledge mode for: %s", question)
    response = llm.complete(prompt)
    return str(response)


# ──────────────────────────────────────────────
# Main ask function (hybrid: RAG + general)
# ──────────────────────────────────────────────

def _format_history(history):
    if not history:
        return ""
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def ask(query_engine, question, history=None, progress_fn=None, style_profile=None):
    """Ask a question with automatic fallback to general knowledge and quality analysis.

    Flow:
      1. Try the RAG pipeline (retrieve code chunks + LLM answer).
      2. If the top retrieved chunk scores below RELEVANCE_THRESHOLD,
         OR the RAG answer says "could not find", fall back to a
         direct LLM call for general knowledge.
      3. Compute deterministic Response Confidence Indicators quality_info.
      4. Return answer, sources, mode, best_score, context_truncated, quality_info.

    Args:
        query_engine: A LlamaIndex query engine (for code questions).
        question: The natural-language question to ask.
        progress_fn: Optional callback function(step_message: str) to
                     report progress to the UI in real time.
        style_profile: Optional style profile key (e.g. 'concise', 'detailed').

    Returns:
        A tuple of:
          (answer_text, sources_list, mode, best_score, context_truncated, quality_info)
    """
    def _progress(msg):
        if progress_fn:
            progress_fn(msg)

    logger.info("Question: %s", question)
    _progress("Reading question...")

    # Resolve style modifier
    style_key = style_profile or DEFAULT_STYLE_PROFILE
    style_modifier = ""
    if style_key in STYLE_PROFILES:
        style_modifier = STYLE_PROFILES[style_key].get("prompt_modifier", "")

    history_text = _format_history(history)

    # Long-context handling: intelligent compression
    compression_info = {
        "was_compressed": False,
        "original_chars": len(history_text),
        "compressed_chars": len(history_text),
        "preserved_symbols": [],
    }
    if history_text:
        history_text, compression_info = _compress_context(
            history_text, question, style_modifier, LLM_CONTEXT_WINDOW
        )
        if compression_info["was_compressed"]:
            _progress("Context compressed to preserve key code structures...")
            logger.info("History compressed: %d -> %d chars", compression_info["original_chars"], compression_info["compressed_chars"])

    effective_query = f"{history_text}\nUser: {question}" if history_text else question
    if style_modifier:
        effective_query = f"[Style: {style_key}]{style_modifier}\n\n{effective_query}"

    # Step 1: Retrieve relevant code chunks
    _progress("Searching codebase for relevant code...")
    response = query_engine.query(effective_query)

    source_nodes = getattr(response, "source_nodes", [])
    best_score = _get_best_score(source_nodes)
    num_chunks = len(source_nodes)
    rag_answer = str(response)

    _progress(f"Retrieved {num_chunks} chunks (best score: {best_score:.2f})")
    logger.info("RAG best_score=%.4f, answer_preview=%s", best_score, rag_answer[:100])

    # Step 2: Decide if we should fall back to general mode
    low_relevance = best_score < RELEVANCE_THRESHOLD
    says_not_found = _answer_says_not_found(rag_answer)

    if low_relevance or says_not_found:
        _progress("Low relevance -- switching to general knowledge mode...")
        logger.info(
            "Falling back to general mode (low_relevance=%s, says_not_found=%s)",
            low_relevance, says_not_found,
        )
        try:
            _progress("Generating answer from general knowledge...")
            general_answer = _ask_general(question, history_text)
            mode = "general"
            answer_text = general_answer
            sources = []
        except Exception as e:
            logger.warning("General-knowledge fallback failed: %s", e)
            mode = "code"
            answer_text = rag_answer
            sources = _extract_sources(response)
    else:
        _progress("Generating answer from codebase context...")
        mode = "code"
        answer_text = rag_answer
        sources = _extract_sources(response)

    # Step 3: Calculate Response Confidence Indicators
    _progress("Calculating response quality...")
    quality_info = _compute_quality_info(
        question=question,
        rag_answer=answer_text,
        sources=sources,
        mode=mode,
        best_score=best_score,
        source_nodes=source_nodes,
    )

    _progress("Generation complete")

    # Log the query locally for debugging/audit
    _log_query(question, answer_text, sources, mode, best_score)

    logger.info("Answer generated in '%s' mode with %d source(s), overall quality=%s (%.1f%%).", 
                mode, len(sources), quality_info["overall"], quality_info["overall_score"])

    return answer_text, sources, mode, best_score, compression_info, quality_info


def _extract_sources(response):
    """Extract deduplicated source information from a RAG response."""
    sources = []
    seen = set()
    for node in getattr(response, "source_nodes", []):
        meta = node.metadata
        key = (meta.get("file", "unknown"), meta.get("name", "unknown"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": meta.get("file", "unknown"),
                "name": meta.get("name", "unknown"),
                "type": meta.get("type", "unknown"),
                "line": meta.get("line", 0),
                "score": round(node.score, 4) if node.score else None,
            })
    return sources


def _log_query(question, answer, sources, mode, best_score):
    """Append a query record to the local log file (JSONL format)."""
    try:
        log_path = Path(QUERY_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer[:500],
            "mode": mode,
            "best_score": round(best_score, 4),
            "source_count": len(sources),
            "sources": [s["file"] + "::" + s["name"] for s in sources],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Failed to write query log: %s", e)


def get_index_stats():
    """Read and return stats from data/index_stats.json, or None if it doesn't exist."""
    stats_file = Path("data/index_stats.json")
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read index stats: %s", e)
    return None


def check_ollama_status():
    """Check Ollama connection and list loaded models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return {"running": True, "models": models}
    except Exception as e:
        logger.warning("Ollama status check failed: %s", e)
        return {"running": False, "models": []}
