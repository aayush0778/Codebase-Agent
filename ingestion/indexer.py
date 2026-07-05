"""
Indexing Pipeline — Phase 2
Embeds AST chunks using Ollama's nomic-embed-text model and persists them
to a local ChromaDB vector store. No data ever leaves the machine.
"""

import logging
import time

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ingestion.ast_chunker import index_codebase

# Import from central config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)


def build_index(root_dir, persist_dir=None):
    """Embed all code chunks and persist them to a local Chroma vector store.

    Args:
        root_dir: Path to the codebase directory to index.
        persist_dir: Directory to persist the Chroma index. Defaults to config value.

    Returns:
        The VectorStoreIndex object.
    """
    if persist_dir is None:
        persist_dir = CHROMA_PERSIST_DIR

    start_time = time.time()

    # Phase 1: Extract chunks via AST parser
    chunks = index_codebase(root_dir)
    if not chunks:
        logger.warning("No chunks extracted — check that the path contains .py files.")
        return None

    logger.info("Extracted %d chunks from %s", len(chunks), root_dir)

    # Convert chunks to LlamaIndex Documents with metadata
    documents = []
    for c in chunks:
        content = f"# File: {c['file']}\n# {c['type']}: {c['name']}\n\n{c['text']}"
        documents.append(Document(
            text=content,
            metadata={
                "file": c["file"],
                "name": c["name"],
                "type": c["type"],
                "line": c["line"],
            }
        ))

    logger.info("Created %d documents, starting embedding...", len(documents))

    # Set up embedding model (local Ollama)
    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

    # Set up ChromaDB (persisted to local disk)
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build and persist the index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    elapsed = time.time() - start_time
    logger.info("Index built and persisted to %s (%.1f seconds)", persist_dir, elapsed)
    print(f"\n[OK] Indexed {len(chunks)} chunks in {elapsed:.1f}s -> {persist_dir}")

    return index
