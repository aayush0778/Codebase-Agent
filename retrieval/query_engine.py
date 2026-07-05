"""
Query Engine -- Phase 3 (Hybrid Mode)
Loads the persisted ChromaDB index, retrieves relevant code chunks for a question,
and generates a grounded answer using the local LLM via Ollama.

If the question is not about the indexed codebase (low relevance scores or
the RAG answer says "not found"), falls back to a direct general-knowledge
LLM call so the user always gets a useful answer.
"""

import logging
import json
import datetime
from pathlib import Path

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
        _llm_instance = Ollama(model=llm_model, request_timeout=LLM_REQUEST_TIMEOUT)
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

    # Embedding model (same one used at indexing time)
    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

    # Local LLM
    llm = _get_llm(llm_model)

    # Reconnect to persisted ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Rebuild the index from the vector store (no re-embedding needed)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # Build query engine with custom prompt
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        text_qa_template=QA_PROMPT_TEMPLATE.partial_format(system_prompt=SYSTEM_PROMPT),
    )

    logger.info("Query engine loaded successfully.")
    return query_engine


# ──────────────────────────────────────────────
# Relevance detection helpers
# ──────────────────────────────────────────────

def _get_best_score(source_nodes):
    """Return the highest similarity score among retrieved nodes."""
    scores = [n.score for n in source_nodes if n.score is not None]
    return max(scores) if scores else 0.0


def _answer_says_not_found(answer_text):
    """Check if the RAG answer contains a 'not found' phrase."""
    lower = answer_text.lower()
    return any(phrase in lower for phrase in NOT_FOUND_PHRASES)


# ──────────────────────────────────────────────
# General-knowledge fallback (direct LLM call)
# ──────────────────────────────────────────────

def _ask_general(question):
    """Send the question directly to the LLM without any code context.

    Returns:
        The answer string from the LLM.
    """
    llm = _get_llm()
    prompt = f"""{GENERAL_SYSTEM_PROMPT}

Question: {question}

Answer:"""

    logger.info("Falling back to general-knowledge mode for: %s", question)
    response = llm.complete(prompt)
    return str(response)


# ──────────────────────────────────────────────
# Main ask function (hybrid: RAG + general)
# ──────────────────────────────────────────────

def ask(query_engine, question):
    """Ask a question with automatic fallback to general knowledge.

    Flow:
      1. Try the RAG pipeline (retrieve code chunks + LLM answer).
      2. If the top retrieved chunk scores below RELEVANCE_THRESHOLD,
         OR the RAG answer says "could not find", fall back to a
         direct LLM call for general knowledge.
      3. Return the answer and metadata indicating which mode was used.

    Args:
        query_engine: A LlamaIndex query engine (for code questions).
        question: The natural-language question to ask.

    Returns:
        A tuple of (answer_text, sources_list).
        sources_list contains dicts with file, name, type, line, score, mode.
        If mode == "general", sources_list will be empty.
    """
    logger.info("Question: %s", question)

    # Step 1: Try RAG pipeline
    response = query_engine.query(question)
    best_score = _get_best_score(response.source_nodes)
    rag_answer = str(response)

    logger.info("RAG best_score=%.4f, answer_preview=%s", best_score, rag_answer[:100])

    # Step 2: Decide if we should fall back to general mode
    low_relevance = best_score < RELEVANCE_THRESHOLD
    says_not_found = _answer_says_not_found(rag_answer)

    if low_relevance or says_not_found:
        # Fall back to general-knowledge LLM
        logger.info(
            "Falling back to general mode (low_relevance=%s, says_not_found=%s)",
            low_relevance, says_not_found,
        )
        try:
            general_answer = _ask_general(question)
            mode = "general"
            answer_text = general_answer
            sources = []
        except Exception as e:
            logger.warning("General-knowledge fallback failed: %s", e)
            # If fallback also fails, return the original RAG answer
            mode = "code"
            answer_text = rag_answer
            sources = _extract_sources(response)
    else:
        # RAG answer is relevant -- use it
        mode = "code"
        answer_text = rag_answer
        sources = _extract_sources(response)

    # Log the query locally for debugging/audit
    _log_query(question, answer_text, sources, mode, best_score)

    logger.info("Answer generated in '%s' mode with %d source(s).", mode, len(sources))
    return answer_text, sources, mode


def _extract_sources(response):
    """Extract deduplicated source information from a RAG response."""
    sources = []
    seen = set()
    for node in response.source_nodes:
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
            "answer": answer[:500],  # truncate for log readability
            "mode": mode,
            "best_score": round(best_score, 4),
            "source_count": len(sources),
            "sources": [s["file"] + "::" + s["name"] for s in sources],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("Failed to write query log: %s", e)
