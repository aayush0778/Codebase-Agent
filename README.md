# 📘 CodebookLM v1.0
Offline AI Codebase Assistant

> Understand Any Codebase. Instantly.

CodebookLM is an offline AI-powered software repository assistant that enables developers to explore, understand and query software repositories using Retrieval-Augmented Generation (RAG), semantic search and local Large Language Models while ensuring complete source-code privacy.

---

## Overview

CodebookLM is a state-of-the-art, fully offline AI-powered software repository assistant. It empowers developers, researchers, and engineers to interact with their codebases using natural language, much like ChatGPT, but explicitly tailored to complex software architecture—without a single byte of code ever leaving your machine. 

By building a local semantic index of your repository using Advanced AST parsing and utilizing a local Large Language Model (defaulting to the highly capable `devstral-small-2:24b`), CodebookLM can answer intricate architectural questions, trace implementation logic across multiple files, and summarize extensive codebases instantly.

It prioritizes:
1. **Privacy & Security**: Perfect for enterprise environments, proprietary code, or defense applications where cloud-based LLMs are prohibited.
2. **Transparency**: Every answer generated from your codebase includes explicit source attribution (File path, Line numbers, and Relevance scores).
3. **Speed & UX**: Designed with a beautiful, minimal, dark-mode desktop UI inspired by developer-first tools like Linear and Cursor.

## Features

- **100% Offline & Private:** No internet connection required. Your code never leaves your machine.
- **Source Attribution:** Every codebase-grounded response includes collapsible citations with file names, line numbers, and relevance scores.
- **Repository Dashboard:** Instant overview of your indexed repository including chunk count, build time, and status.
- **System Diagnostics:** Built-in monitoring for Ollama status, active LLM, Embedding model, and GPU VRAM usage.
- **Export Conversations:** Export your entire chat session as Markdown or Text for documentation or sharing.
- **Premium Desktop Experience:** Clean, minimal, professional interface inspired by tools like Linear, Cursor, and Notion.
- **Hybrid Retrieval:** Automatically falls back to general knowledge if the answer isn't in your codebase.

## Architecture

```mermaid
graph TD
    A[Upload Repository / .zip] --> B(AST Parsing & Chunking)
    B --> C[Ollama Local Embeddings]
    C --> D[(ChromaDB Vector Store)]
    
    E[User Question] --> F(Query Engine)
    F --> |Retrieve Top-K Chunks| D
    
    D --> G[Local LLM Generation]
    G --> H[CodebookLM v1.0 UI]
    H --> |Sources & Confidence| I(Developer)
```

## Technology Stack

- **Frontend:** Streamlit (Custom UI/UX styling)
- **RAG & Indexing:** LlamaIndex
- **Vector Database:** ChromaDB
- **Local LLM Engine:** Ollama
- **Default LLM:** Devstral Small 2 (`devstral-small-2:24b`)
- **Default Embeddings:** `nomic-embed-text`

## Installation

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running.

### 2. Install Dependencies
```bash
git clone https://github.com/aayush0778/Codebase-Agent.git
cd Codebase-Agent
pip install -r requirements.txt
```

### 3. Pull Required Models
Ensure Ollama is running, then download the necessary models:
```bash
ollama pull devstral-small-2:24b
ollama pull nomic-embed-text
```

## Usage

### Launching the Application
You can use the provided desktop launchers for a quick start:
- **Windows:** Double-click `CodebookLM.bat`
- **PowerShell:** Run `.\Launch-CodebookLM.ps1`

Or run manually via terminal:
```bash
python -m streamlit run app.py
```

### Workflow
1. **Upload Repository:** Drag and drop your `.py` files or a `.zip` archive containing your repository using the sidebar.
2. **Build Semantic Index:** Click "Build Index" to parse and embed your codebase into ChromaDB.
3. **Ask Questions:** Use the chat interface to ask architectural, stylistic, or implementation questions.
4. **Inspect Sources:** Expand the "📄 Sources" panel after any codebase response to see exactly where the LLM got its information.
5. **Export:** Click the "Export" buttons in the sidebar to save your session.

## Screenshots

### 1. Main Dashboard & Chat Interface
![CodebookLM Dashboard](assets/screenshot1.png)
*The clean, minimal, Linear-inspired dark interface with system diagnostics and repository status on the sidebar.*

### 2. Retrieval in Action
![CodebookLM Query Response](assets/screenshot2.png)
*Detailed streaming responses with confidence metrics and general knowledge fallbacks when code isn't relevant.*

## Troubleshooting

- **No index loaded?** Ensure you upload files and click "Build Index" in the sidebar before asking code-related questions.
- **Model not found error?** Check that Ollama is running and you have pulled the required models (`ollama list`). You can change the default model in `config.py`.
- **GPU Not Detected?** Ensure `nvidia-smi` is accessible in your system PATH. The application will fall back to CPU if a GPU is unavailable.

## Future Enhancements
- Multi-language AST parsing (JS/TS, Go, Rust, Java).
- Advanced chunking strategies (e.g., semantic boundaries).
- Direct GitHub repository ingestion via URL.
- Persistent multiple repository indexing.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Credits

Designed and built for developers who care about code privacy and AI accessibility.
