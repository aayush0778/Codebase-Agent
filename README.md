# Local AST-Aware Codebase Assistant

> **Codebase Assistant** — Intelligent local RAG system for querying a Python codebase.  
> **Fully offline / air-gapped.** No cloud APIs, no data leaving the machine at any point.

## Architecture

```
Python Codebase (.py files)
        │
        ▼
[1] AST Parser (ast_chunker.py)
    Extracts functions, classes, async functions
        │
        ▼
[2] Embedding Model (Ollama: nomic-embed-text)
        │
        ▼
[3] Vector Store (ChromaDB → local disk)
        │
        ▼
   User Question (typed in Streamlit UI)
        │
        ▼
[4] Similarity Search → top-k chunks retrieved
        │
        ▼
[5] Local LLM (Ollama: devstral:24b)
        │
        ▼
[6] Grounded Answer + Source Citations
```

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Ollama** installed and running

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull devstral:24b
ollama pull nomic-embed-text

# Verify
ollama list
```

### 2. Install Dependencies

```bash
cd code-rag-agent
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Index a Codebase

```bash
# Via CLI script
python scripts/build_index.py --path /path/to/your/python/codebase

# Or use the Streamlit sidebar (see step 4)
```

### 4. Launch the Chat UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```
code-rag-agent/
├── config.py                 # Centralized configuration
├── app.py                    # Streamlit chat UI (entry point)
├── requirements.txt          # Python dependencies
├── ingestion/
│   ├── __init__.py
│   ├── ast_chunker.py        # AST parsing → chunk extraction
│   └── indexer.py            # Embed chunks, build/persist Chroma index
├── retrieval/
│   ├── __init__.py
│   └── query_engine.py       # Retrieval + prompt + LLM call
├── scripts/
│   └── build_index.py        # CLI script to (re)index a codebase
├── data/
│   └── chroma_index/         # Persisted vector store (gitignored)
└── sample_codebase/          # Sample code for testing
```

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `devstral:24b` | Ollama LLM model for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `TOP_K` | `8` | Number of chunks retrieved per query |
| `LLM_REQUEST_TIMEOUT` | `300.0` | LLM request timeout in seconds |

### Hardware Recommendations

| RAM | GPU VRAM | Recommended Model |
|-----|----------|-------------------|
| 16 GB | 8 GB | `deepseek-coder:6.7b` |
| 32 GB | 16 GB | `devstral:24b` |
| 64 GB+ | 24 GB+ | `codellama:34b` |
| CPU only | — | `deepseek-coder:6.7b` (slower) |

## Security Notes

- **All data stays local.** No network calls at runtime.
- The `data/chroma_index/` folder contains embedded representations of your code — **treat it with the same classification level as the source code.**
- Query logs are stored in `data/query_log.jsonl` — also treat as classified if applicable.

## Key Files to Understand

1. **`ast_chunker.py`** — Why function/class-level chunking beats fixed-size text splitting for code
2. **`indexer.py`** — How chunks become vectors with metadata attached at indexing time
3. **`query_engine.py`** — The retrieval-then-generate flow; try changing `top_k` in `config.py`
4. **The prompt template** in `config.py` — Small wording changes have outsized effects on hallucination
5. **`app.py`** — The Streamlit UI (least conceptually important, most visible in demos)
