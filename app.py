"""
Codebase Assistant — Streamlit UI
A local RAG-powered assistant for querying Python codebases.
Supports both codebase-specific and general knowledge questions.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import os
import shutil
import time
import zipfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from ingestion.indexer import build_index
from retrieval.query_engine import (
    load_query_engine, ask, get_index_stats, check_ollama_status,
)
from config import (
    DEFAULT_CODEBASE_PATH, LLM_MODEL, TOP_K, EMBEDDING_MODEL,
    DATA_DIR, RELEVANCE_THRESHOLD,
)

# ──────────────────────────────────────────────
# Constants & Directories
# ──────────────────────────────────────────────
CHAT_HISTORY_DIR = os.path.join(str(DATA_DIR), "chat_history")
UPLOAD_DIR = os.path.join(str(DATA_DIR), "uploads")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

EXAMPLE_QUESTIONS = [
    "Explain the project architecture",
    "Summarize the repository",
    "Where is authentication implemented?",
    "Find API endpoints",
    "Explain the indexing pipeline",
    "Locate database connections",
]

# ──────────────────────────────────────────────
# Chat History Helpers
# ──────────────────────────────────────────────

def _generate_chat_id():
    """Generate a unique chat ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _get_chat_title(messages):
    """Extract a title from the first user message, or return a default."""
    for msg in messages:
        if msg["role"] == "user":
            title = msg["content"][:50]
            if len(msg["content"]) > 50:
                title += "..."
            return title
    return "Empty chat"


def _save_chat(chat_id, messages):
    """Save a chat session to disk as JSON."""
    if not messages:
        return
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    data = {
        "id": chat_id,
        "title": _get_chat_title(messages),
        "updated": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": messages,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_chat(chat_id):
    """Load a chat session from disk."""
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_chats():
    """List all saved chats, sorted by most recent first."""
    chats = []
    for fname in os.listdir(CHAT_HISTORY_DIR):
        if fname.endswith(".json"):
            filepath = os.path.join(CHAT_HISTORY_DIR, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chats.append({
                    "id": data.get("id", fname.replace(".json", "")),
                    "title": data.get("title", "Untitled"),
                    "updated": data.get("updated", ""),
                    "message_count": data.get("message_count", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    chats.sort(key=lambda c: c["updated"], reverse=True)
    return chats


def _delete_chat(chat_id):
    """Delete a saved chat from disk."""
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)


def _handle_uploads(uploaded_files):
    """Save uploaded .py and .zip files to the upload directory.

    Returns the path to the directory and the count of .py files.
    """
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    py_count = 0
    for uf in uploaded_files:
        if uf.name.endswith(".zip"):
            zip_path = os.path.join(UPLOAD_DIR, uf.name)
            with open(zip_path, "wb") as f:
                f.write(uf.getbuffer())
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(UPLOAD_DIR)
            os.remove(zip_path)
            for root, _, files in os.walk(UPLOAD_DIR):
                py_count += sum(1 for f in files if f.endswith(".py"))
        else:
            with open(os.path.join(UPLOAD_DIR, uf.name), "wb") as f:
                f.write(uf.getbuffer())
            py_count += 1

    return UPLOAD_DIR, py_count


def _get_gpu_info():
    """Detect GPU availability and details."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            return {
                "available": True,
                "name": parts[0] if len(parts) > 0 else "Unknown",
                "vram_total": f"{parts[1]} MB" if len(parts) > 1 else "N/A",
                "vram_free": f"{parts[2]} MB" if len(parts) > 2 else "N/A",
            }
    except Exception:
        pass
    return {"available": False, "name": "N/A", "vram_total": "N/A", "vram_free": "N/A"}


def _render_mode_badge(mode, confidence=None):
    """Render an HTML mode badge based on answer mode and confidence."""
    if mode == "code":
        conf_text = f" (Confidence {int(confidence * 100)}%)" if confidence else ""
        return (
            f'<span class="mode-badge mode-code">'
            f'&#x1F7E2; Codebase{conf_text}</span>'
        )
    elif mode == "general":
        return (
            '<span class="mode-badge mode-general">'
            '&#x1F30D; General Knowledge</span>'
        )
    return ""


def _render_sources(sources, mode):
    """Render source attribution section."""
    if mode == "code" and sources:
        with st.expander(f"📄 Sources ({len(sources)} files)", expanded=False):
            for s in sources:
                score_text = f" — score: {s['score']}" if s.get("score") else ""
                st.markdown(
                    f"✓ **{s['file']}** → `{s['name']}` "
                    f"(`{s['type']}`, Line {s['line']}){score_text}"
                )
    elif mode == "general":
        st.markdown(
            '<div class="general-note">'
            '🌍 <strong>General Knowledge</strong> — '
            'No relevant code was found in the indexed repository. '
            'Response generated using the LLM\'s general knowledge.'
            '</div>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Codebase Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS: Premium developer-tool aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base: #0F1115;
    --bg-sidebar: #14161B;
    --bg-card: #1B1D23;
    --bg-input: #242730;
    --bg-hover: #242730;
    --border: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.14);
    --accent: #22C55E;
    --accent-secondary: #4ADE80;
    --accent-dim: rgba(34, 197, 94, 0.10);
    --accent-border: rgba(34, 197, 94, 0.22);
    --text-primary: #F5F5F5;
    --text-secondary: #A1A1AA;
    --text-muted: #71717A;
    --danger-dim: rgba(239, 68, 68, 0.08);
    --purple-dim: rgba(168, 85, 247, 0.08);
    --purple-text: #c4b5fd;
    --yellow-dim: rgba(234, 179, 8, 0.10);
    --yellow-text: #fde047;
    --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.3);
    --shadow-sm: 0 2px 6px rgba(0, 0, 0, 0.2);
    --radius: 16px;
    --radius-sm: 10px;
    --radius-xs: 6px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    line-height: 1.6 !important;
}

.stApp {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.stApp > header { background: transparent !important; }

.stMainBlockContainer {
    max-width: 880px !important;
    padding: 2rem 1.5rem !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] h1 {
    font-size: 1.1rem !important; font-weight: 700 !important;
    letter-spacing: -0.01em !important; margin-bottom: 20px !important;
    background: none !important; -webkit-text-fill-color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] h2 {
    font-size: 0.68rem !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important; color: var(--text-muted) !important;
    margin-top: 20px !important; margin-bottom: 8px !important; font-weight: 600 !important;
}
section[data-testid="stSidebar"] code {
    background: var(--accent-dim) !important; color: var(--accent) !important;
    padding: 2px 7px !important; border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
    border: 1px solid var(--accent-border) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    font-size: 0.74rem !important; padding: 0.35rem 0.7rem !important;
    text-align: left !important; white-space: nowrap !important;
    overflow: hidden !important; text-overflow: ellipsis !important;
}

/* ── TITLE ── */
h1 {
    background: linear-gradient(135deg, #F5F5F5 0%, #d4d4d8 45%, #4ADE80 55%, #d4d4d8 85%, #F5F5F5 100%) !important;
    background-size: 300% 300% !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important; font-weight: 700 !important;
    letter-spacing: -0.025em !important; animation: titleShimmer 14s ease-in-out infinite !important;
}
@keyframes titleShimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* ── STATUS BADGES ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 16px; border-radius: 50px; font-size: 0.78em;
    font-weight: 500; font-family: 'Inter', sans-serif; border: 1px solid;
}
.status-ready { background: var(--accent-dim); color: var(--accent-secondary); border-color: var(--accent-border); }
.status-none { background: var(--danger-dim); color: #fca5a5; border-color: rgba(239,68,68,0.15); }

/* ── MODE BADGES ── */
.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 13px; border-radius: 50px; font-size: 0.7em;
    font-weight: 500; font-family: 'Inter', sans-serif;
    letter-spacing: 0.02em; margin-bottom: 8px; border: 1px solid;
}
.mode-code { background: var(--accent-dim); color: var(--accent-secondary); border-color: var(--accent-border); }
.mode-general { background: var(--purple-dim); color: var(--purple-text); border-color: rgba(168,85,247,0.15); }
.mode-mixed { background: var(--yellow-dim); color: var(--yellow-text); border-color: rgba(234,179,8,0.15); }

/* ── GENERAL NOTE ── */
.general-note {
    background: rgba(168, 85, 247, 0.06); border: 1px solid rgba(168,85,247,0.12);
    border-radius: var(--radius-sm); padding: 10px 16px; margin-top: 8px;
    font-size: 0.82em; color: var(--text-secondary);
}

/* ── INFO CARD (sidebar) ── */
.info-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 14px 16px; margin: 8px 0;
    font-size: 0.78em; line-height: 1.8;
}
.info-card .label { color: var(--text-muted); }
.info-card .value { color: var(--text-primary); font-weight: 500; }
.info-card .accent { color: var(--accent); font-weight: 600; }
.info-card .dot-ok { color: #22C55E; }
.info-card .dot-err { color: #ef4444; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; padding: 22px !important;
    margin-bottom: 16px !important; box-shadow: var(--shadow-card) !important;
    transition: border-color 0.25s ease !important;
    animation: msgFadeIn 0.3s ease-out forwards; color: var(--text-primary) !important;
}
[data-testid="stChatMessage"] * { color: var(--text-primary) !important; }
[data-testid="stChatMessage"]:hover { border-color: var(--border-hover) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { border-left: 2px solid var(--accent) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { border-left: 2px solid rgba(255,255,255,0.08) !important; }
@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── CODE BLOCKS ── */
[data-testid="stChatMessage"] pre {
    background: var(--bg-base) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-xs) !important; padding: 14px 18px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
    line-height: 1.7 !important; overflow-x: auto !important; margin: 10px 0 !important;
    position: relative !important;
}
[data-testid="stChatMessage"] code {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
}
[data-testid="stChatMessage"] p code {
    background: rgba(255,255,255,0.05) !important; padding: 2px 6px !important;
    border-radius: 4px !important; border: 1px solid var(--border) !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] { border-radius: var(--radius) !important; overflow: hidden; }
[data-testid="stChatInput"] textarea {
    background: var(--bg-input) !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: var(--radius) !important; color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important; padding: 14px 18px !important;
    font-size: 0.9rem !important; line-height: 1.6 !important;
    transition: border-color 0.25s ease !important;
}
[data-testid="stChatInput"] textarea:hover { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stChatInput"] textarea:focus { border-color: var(--accent) !important; box-shadow: none !important; outline: none !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: rgba(255,255,255,0.03) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
    font-weight: 500 !important; font-family: 'Inter', sans-serif !important;
    padding: 0.45rem 1rem !important; backdrop-filter: blur(6px) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.06) !important; border-color: var(--border-hover) !important;
    transform: translateY(-1px); box-shadow: var(--shadow-sm) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.98); }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(15,17,21,0.5) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; margin-top: 10px !important;
    transition: border-color 0.25s ease !important;
}
[data-testid="stExpander"]:hover { border-color: var(--border-hover) !important; }
[data-testid="stExpander"] summary { color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.82rem !important; }

/* ── STATUS CONTAINER ── */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] input {
    background: var(--bg-base) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-xs) !important; color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important;
    transition: border-color 0.25s ease !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--accent-border) !important; box-shadow: none !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] { border-radius: var(--radius-sm) !important; }

/* ── EXAMPLE QUESTIONS ── */
.example-btn button {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; font-size: 0.8rem !important;
    padding: 8px 14px !important; text-align: left !important;
    transition: all 0.2s ease !important; width: 100% !important;
}
.example-btn button:hover {
    border-color: var(--accent-border) !important;
    background: var(--bg-hover) !important;
}

/* ── ONBOARDING ── */
.onboarding {
    text-align: center; padding: 60px 20px; color: var(--text-secondary);
}
.onboarding h2 { color: var(--text-primary); font-size: 1.4rem; font-weight: 600; margin-bottom: 8px; }
.onboarding .step {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 10px 20px; margin: 6px;
    font-size: 0.85em; transition: border-color 0.2s ease;
}
.onboarding .arrow { color: var(--text-muted); font-size: 1.2em; margin: 4px 0; }

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 20px 0; margin-top: 40px;
    border-top: 1px solid var(--border); color: var(--text-muted);
    font-size: 0.72em; letter-spacing: 0.04em;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.1); }

/* ── MISC ── */
hr { border-color: var(--border) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; font-size: 0.75rem !important; }
[data-testid="stAlert"] { border-radius: var(--radius-sm) !important; }
[data-testid="stSpinner"] { color: var(--accent) !important; }
#cursor-glow, #particle-canvas, #flash-overlay { display: none; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# JavaScript: auto-scroll + keyboard shortcuts + copy button + subtle glow
# ──────────────────────────────────────────────
components.html("""
<script>
(function() {
    const stDoc = window.parent.document;

    /* ── SUBTLE CURSOR GLOW ── */
    stDoc.querySelectorAll('#cursor-glow-live').forEach(el => el.remove());
    const glowEl = stDoc.createElement('div');
    glowEl.id = 'cursor-glow-live';
    glowEl.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9998; transition: background 0.3s ease;
    `;
    stDoc.body.appendChild(glowEl);
    stDoc.addEventListener('mousemove', function(e) {
        glowEl.style.background =
            `radial-gradient(450px circle at ${e.clientX}px ${e.clientY}px,
             rgba(34, 197, 94, 0.025), transparent 60%)`;
    });

    /* ── AUTO-SCROLL ── */
    const scrollObserver = new MutationObserver(function(mutations) {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                if (node.querySelector &&
                    (node.querySelector('[data-testid="chatAvatarIcon-user"]') ||
                     node.querySelector('[data-testid="chatAvatarIcon-assistant"]'))) {
                    setTimeout(() => {
                        const msgs = stDoc.querySelectorAll('[data-testid="stChatMessage"]');
                        if (msgs.length > 0) {
                            msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
                        }
                    }, 120);
                }
            }
        }
    });
    const mainBlock = stDoc.querySelector('[data-testid="stMainBlockContainer"]');
    if (mainBlock) scrollObserver.observe(mainBlock, { childList: true, subtree: true });

    /* ── KEYBOARD SHORTCUTS ── */
    stDoc.addEventListener('keydown', function(e) {
        // Ctrl+L → Clear chat (triggers New Chat button)
        if (e.ctrlKey && e.key === 'l') {
            e.preventDefault();
            const btns = stDoc.querySelectorAll('button');
            for (const b of btns) {
                if (b.textContent.trim() === 'New Chat') { b.click(); break; }
            }
        }
        // Ctrl+K → Focus input
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            const input = stDoc.querySelector('[data-testid="stChatInput"] textarea');
            if (input) input.focus();
        }
    });

    /* ── COPY BUTTON for code blocks ── */
    function addCopyButtons() {
        const blocks = stDoc.querySelectorAll('[data-testid="stChatMessage"] pre');
        blocks.forEach(function(pre) {
            if (pre.querySelector('.copy-btn')) return;
            const btn = stDoc.createElement('button');
            btn.className = 'copy-btn';
            btn.textContent = 'Copy';
            btn.style.cssText = `
                position: absolute; top: 6px; right: 6px;
                background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
                border-radius: 6px; color: #a1a1aa; font-size: 11px;
                padding: 3px 10px; cursor: pointer; z-index: 10;
                font-family: 'Inter', sans-serif; transition: all 0.2s ease;
            `;
            btn.addEventListener('mouseenter', function() {
                btn.style.background = 'rgba(255,255,255,0.14)';
                btn.style.color = '#f5f5f5';
            });
            btn.addEventListener('mouseleave', function() {
                btn.style.background = 'rgba(255,255,255,0.08)';
                btn.style.color = '#a1a1aa';
            });
            btn.addEventListener('click', function() {
                const code = pre.querySelector('code');
                const text = code ? code.textContent : pre.textContent;
                navigator.clipboard.writeText(text).then(function() {
                    btn.textContent = 'Copied!';
                    btn.style.color = '#22C55E';
                    setTimeout(function() { btn.textContent = 'Copy'; btn.style.color = '#a1a1aa'; }, 1500);
                });
            });
            pre.style.position = 'relative';
            pre.appendChild(btn);
        });
    }

    // Run initially and observe for new messages
    addCopyButtons();
    const copyObserver = new MutationObserver(function() {
        setTimeout(addCopyButtons, 200);
    });
    if (mainBlock) copyObserver.observe(mainBlock, { childList: true, subtree: true });
})();
</script>
""", height=0)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⟐ Codebase Assistant")

    # ── File Upload ──
    st.markdown("---")
    st.subheader("Upload Codebase")

    uploaded_files = st.file_uploader(
        "Upload Python files or a .zip repository",
        accept_multiple_files=True,
        type=["py", "zip"],
        help="Upload .py files directly or a .zip archive containing your project.",
    )

    if uploaded_files:
        upload_path, py_count = _handle_uploads(uploaded_files)
        st.success(f"Uploaded {py_count} Python file(s)")
        st.session_state["upload_path"] = upload_path

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build Index", use_container_width=True):
            target_path = st.session_state.get("upload_path", DEFAULT_CODEBASE_PATH)
            target = Path(target_path)
            if not target.is_dir():
                st.error(f"No valid directory: {target_path}")
            else:
                with st.spinner("Indexing codebase..."):
                    try:
                        result = build_index(str(target))
                        st.session_state.engine = None
                        if isinstance(result, tuple) and len(result) == 2:
                            _, stats = result
                            st.session_state["index_stats"] = stats
                        st.success("Index built successfully!")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
    with col2:
        if st.button("Rebuild", use_container_width=True):
            target_path = st.session_state.get("upload_path", DEFAULT_CODEBASE_PATH)
            target = Path(target_path)
            if not target.is_dir():
                st.error(f"No valid directory: {target_path}")
            else:
                with st.spinner("Rebuilding index..."):
                    try:
                        import chromadb
                        from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
                        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                        try:
                            client.delete_collection(CHROMA_COLLECTION_NAME)
                        except Exception:
                            pass
                        result = build_index(str(target))
                        st.session_state.engine = None
                        if isinstance(result, tuple) and len(result) == 2:
                            _, stats = result
                            st.session_state["index_stats"] = stats
                        st.success("Index rebuilt from scratch.")
                    except Exception as e:
                        st.error(f"Rebuild failed: {e}")

    with st.expander("Advanced: Manual Path"):
        manual_path = st.text_input(
            "Codebase folder path",
            value=DEFAULT_CODEBASE_PATH,
            help="Absolute path to a folder of .py files.",
        )
        if st.button("Use This Path", use_container_width=True):
            st.session_state["upload_path"] = manual_path
            st.success(f"Path set: {manual_path}")

    # ── Repository & Models Info Card (#2, #6) ──
    st.markdown("---")
    st.subheader("Repository & Models")

    stats = st.session_state.get("index_stats") or get_index_stats()
    if stats:
        st.session_state["index_stats"] = stats
        st.markdown(
            f'<div class="info-card">'
            f'<span class="label">Repository</span><br>'
            f'<span class="accent">{stats.get("repo_name", "N/A")}</span><br><br>'
            f'<span class="label">Status</span><br>'
            f'<span class="dot-ok">●</span> <span class="value">Indexed</span><br>'
            f'<span class="label">Files</span> '
            f'<span class="value">{stats.get("files_indexed", "?")}</span> · '
            f'<span class="label">Chunks</span> '
            f'<span class="value">{stats.get("chunks_generated", "?")}</span><br>'
            f'<span class="label">Built in</span> '
            f'<span class="value">{stats.get("elapsed_seconds", 0):.1f}s</span><br>'
            f'<span class="label">Last indexed</span> '
            f'<span class="value">{stats.get("timestamp", "N/A")[:19]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-card">'
            '<span class="label">Repository</span><br>'
            '<span class="dot-err">●</span> <span class="value">Not Indexed</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="info-card">'
        f'<span class="label">LLM</span> <span class="value">`{LLM_MODEL}`</span><br>'
        f'<span class="label">Embeddings</span> <span class="value">`{EMBEDDING_MODEL}`</span><br>'
        f'<span class="label">Top-K</span> <span class="value">{TOP_K}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── System Status (#5) ──
    st.markdown("---")
    st.subheader("System Status")

    gpu_info = _get_gpu_info()
    ollama_status = check_ollama_status()
    ollama_dot = "dot-ok" if ollama_status["running"] else "dot-err"
    ollama_text = "Running" if ollama_status["running"] else "Offline"
    gpu_dot = "dot-ok" if gpu_info["available"] else "dot-err"
    gpu_label = gpu_info["name"] if gpu_info["available"] else "CPU only"

    st.markdown(
        f'<div class="info-card">'
        f'<span class="label">Ollama</span> '
        f'<span class="{ollama_dot}">●</span> '
        f'<span class="value">{ollama_text}</span><br>'
        f'<span class="label">GPU</span> '
        f'<span class="{gpu_dot}">●</span> '
        f'<span class="value">{gpu_label}</span><br>'
        + (f'<span class="label">VRAM</span> '
           f'<span class="value">{gpu_info["vram_free"]} free / {gpu_info["vram_total"]}</span><br>'
           if gpu_info["available"] else '')
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chat History ──
    st.markdown("---")
    st.subheader("Chat History")

    if st.button("New Chat", use_container_width=True):
        if st.session_state.get("messages"):
            _save_chat(st.session_state.chat_id, st.session_state.messages)
        st.session_state.messages = []
        st.session_state.chat_id = _generate_chat_id()
        st.session_state["session_start"] = datetime.now().isoformat()
        st.rerun()

    saved_chats = _list_chats()
    if saved_chats:
        for chat in saved_chats[:15]:
            col_load, col_del = st.columns([5, 1])
            with col_load:
                label = f"{chat['title']}  ({chat['message_count']} msgs)"
                if st.button(label, key=f"load_{chat['id']}", use_container_width=True):
                    if st.session_state.get("messages"):
                        _save_chat(st.session_state.chat_id, st.session_state.messages)
                    loaded = _load_chat(chat["id"])
                    if loaded:
                        st.session_state.messages = loaded["messages"]
                        st.session_state.chat_id = chat["id"]
                        st.rerun()
            with col_del:
                if st.button("X", key=f"del_{chat['id']}"):
                    _delete_chat(chat["id"])
                    if st.session_state.get("chat_id") == chat["id"]:
                        st.session_state.messages = []
                        st.session_state.chat_id = _generate_chat_id()
                    st.rerun()
    else:
        st.caption("No saved chats yet.")

    # ── Session Info (#11) ──
    st.markdown("---")
    st.subheader("Session")
    msg_count = len(st.session_state.get("messages", []))
    session_start = st.session_state.get("session_start", datetime.now().isoformat())
    repo_loaded = "Yes" if st.session_state.get("engine") else "No"
    st.markdown(
        f'<div class="info-card">'
        f'<span class="label">Messages</span> <span class="value">{msg_count}</span><br>'
        f'<span class="label">Repository</span> <span class="value">{repo_loaded}</span><br>'
        f'<span class="label">Started</span> <span class="value">{session_start[:19]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.caption("Ctrl+L Clear Chat · Ctrl+K Focus Input")

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = _generate_chat_id()
if "engine" not in st.session_state:
    st.session_state.engine = None
if "session_start" not in st.session_state:
    st.session_state["session_start"] = datetime.now().isoformat()

# ──────────────────────────────────────────────
# Load query engine
# ──────────────────────────────────────────────
if st.session_state.engine is None:
    try:
        st.session_state.engine = load_query_engine()
    except Exception:
        pass

# ──────────────────────────────────────────────
# Main Panel
# ──────────────────────────────────────────────
st.title("Codebase Assistant")
st.caption("Ask anything — codebase questions use your indexed code, general questions get direct answers.")

# Status badge
if st.session_state.engine:
    st.markdown(
        '<span class="status-badge status-ready">● Index loaded — ready</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-badge status-none">○ No index — upload files and build one</span>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Empty state: Onboarding view (#10) or Example Questions (#7)
# ──────────────────────────────────────────────
if not st.session_state.messages:
    if not st.session_state.engine:
        # Onboarding
        st.markdown(
            '<div class="onboarding">'
            '<h2>Get Started</h2>'
            '<p>Index your Python codebase and ask questions about it.</p>'
            '<div class="step">📁 Upload a Python repository</div>'
            '<div class="arrow">↓</div>'
            '<div class="step">⚡ Build Index</div>'
            '<div class="arrow">↓</div>'
            '<div class="step">💬 Ask questions about your code</div>'
            '<div class="arrow">↓</div>'
            '<div class="step">🎯 Receive grounded answers using RAG</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Example questions
        st.markdown("#### Try an example question")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state["_pending_question"] = q
                    st.rerun()

# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("mode"):
            badge_html = _render_mode_badge(msg["mode"], msg.get("confidence"))
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []), msg.get("mode", ""))

# ──────────────────────────────────────────────
# Chat input with thinking steps + streaming
# ──────────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)
question = pending or st.chat_input("Ask about the codebase or anything else...")

if question:
    if not st.session_state.engine:
        st.warning("Please build an index first using the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer with thinking steps
        with st.chat_message("assistant"):
            with st.status("Processing your question...", expanded=True) as status:
                def on_progress(step_msg):
                    st.write(f"✓ {step_msg}")

                try:
                    on_progress("Reading question...")
                    answer, sources, mode, best_score = ask(
                        st.session_state.engine,
                        question,
                        progress_fn=on_progress,
                    )
                    status.update(label="Complete", state="complete", expanded=False)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
                    sources = []
                    mode = "error"
                    best_score = 0.0
                    status.update(label="Error", state="error", expanded=False)

            # Confidence & badge
            confidence = best_score if mode == "code" else None
            badge_html = _render_mode_badge(mode, confidence)
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)

            # Stream-style rendering (write_stream not available, use markdown)
            st.markdown(answer)

            # Source attribution
            _render_sources(sources, mode)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "mode": mode,
            "confidence": confidence,
        })

        _save_chat(st.session_state.chat_id, st.session_state.messages)

# ──────────────────────────────────────────────
# Footer (#12)
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Offline · Local LLM · Retrieval-Augmented Generation · No Internet Required'
    '</div>',
    unsafe_allow_html=True,
)
