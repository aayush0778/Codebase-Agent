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
import zipfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from ingestion.indexer import build_index
from retrieval.query_engine import load_query_engine, ask
from config import DEFAULT_CODEBASE_PATH, LLM_MODEL, TOP_K, EMBEDDING_MODEL, DATA_DIR

# ──────────────────────────────────────────────
# Chat History Helpers
# ──────────────────────────────────────────────
CHAT_HISTORY_DIR = os.path.join(str(DATA_DIR), "chat_history")
UPLOAD_DIR = os.path.join(str(DATA_DIR), "uploads")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
    # Clear previous uploads
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
# CSS: Premium neutral dark theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════
   DESIGN TOKENS
   ═══════════════════════════════════════════════ */
:root {
    --bg-base: #161618;
    --bg-sidebar: #1C1D20;
    --bg-card: #24262B;
    --bg-input: #2B2D31;
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --accent: #3ECF8E;
    --accent-secondary: #6EE7B7;
    --accent-dim: rgba(62, 207, 142, 0.10);
    --accent-border: rgba(62, 207, 142, 0.22);
    --text-primary: #F5F5F5;
    --text-secondary: #A1A1AA;
    --text-muted: #71717A;
    --danger-dim: rgba(239, 68, 68, 0.08);
    --purple-dim: rgba(168, 85, 247, 0.08);
    --purple-text: #c4b5fd;
    --shadow-card: 0 10px 30px rgba(0, 0, 0, 0.35);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.2);
    --radius: 20px;
    --radius-sm: 12px;
    --radius-xs: 8px;
    --spacing-section: 28px;
    --spacing-chat: 20px;
}

/* ═══════════════════════════════════════════════
   GLOBAL
   ═══════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    line-height: 1.6 !important;
}

.stApp {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.stApp > header {
    background: transparent !important;
}

.stMainBlockContainer {
    max-width: 880px !important;
    padding: 2.5rem 1.5rem !important;
}

/* ═══════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] h1 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    margin-bottom: var(--spacing-section) !important;
}

section[data-testid="stSidebar"] h2 {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    margin-top: var(--spacing-section) !important;
    margin-bottom: 12px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] code {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    border: 1px solid var(--accent-border) !important;
}

section[data-testid="stSidebar"] .stButton > button {
    font-size: 0.76rem !important;
    padding: 0.4rem 0.8rem !important;
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* ═══════════════════════════════════════════════
   TITLE — Slow shimmer, white-to-gray with emerald tint
   ═══════════════════════════════════════════════ */
h1 {
    background: linear-gradient(
        135deg,
        #F5F5F5 0%,
        #d4d4d8 40%,
        #6EE7B7 60%,
        #d4d4d8 80%,
        #F5F5F5 100%
    ) !important;
    background-size: 300% 300% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    animation: titleShimmer 14s ease-in-out infinite !important;
}

@keyframes titleShimmer {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}

/* ═══════════════════════════════════════════════
   STATUS BADGES
   ═══════════════════════════════════════════════ */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 50px;
    font-size: 0.8em;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    border: 1px solid;
    margin-bottom: 8px;
}

.status-ready {
    background: var(--accent-dim);
    color: var(--accent-secondary);
    border-color: var(--accent-border);
}

.status-none {
    background: var(--danger-dim);
    color: #fca5a5;
    border-color: rgba(239, 68, 68, 0.15);
}

/* ═══════════════════════════════════════════════
   MODE BADGES
   ═══════════════════════════════════════════════ */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.7em;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.02em;
    margin-bottom: 8px;
    border: 1px solid;
}

.mode-code {
    background: var(--accent-dim);
    color: var(--accent-secondary);
    border-color: var(--accent-border);
}

.mode-general {
    background: var(--purple-dim);
    color: var(--purple-text);
    border-color: rgba(168, 85, 247, 0.15);
}

/* ═══════════════════════════════════════════════
   CHAT MESSAGES
   ═══════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    margin-bottom: var(--spacing-chat) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.25s ease, transform 0.25s ease !important;
    animation: msgFadeIn 0.3s ease-out forwards;
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"] * {
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--border-hover) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 2px solid var(--accent) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid rgba(255, 255, 255, 0.08) !important;
}

@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════
   CODE BLOCKS
   ═══════════════════════════════════════════════ */
[data-testid="stChatMessage"] pre {
    background: var(--bg-base) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xs) !important;
    padding: 16px 20px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
    overflow-x: auto !important;
    margin: 12px 0 !important;
}

[data-testid="stChatMessage"] code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stChatMessage"] p code {
    background: rgba(255, 255, 255, 0.05) !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    border: 1px solid var(--border) !important;
}

/* ═══════════════════════════════════════════════
   CHAT INPUT — ChatGPT / Claude style
   ═══════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    border-radius: var(--radius) !important;
    overflow: hidden;
}

[data-testid="stChatInput"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 16px 20px !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.25s ease !important;
}

[data-testid="stChatInput"] textarea:hover {
    border-color: rgba(255, 255, 255, 0.12) !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ═══════════════════════════════════════════════
   BUTTONS — Glassmorphism
   ═══════════════════════════════════════════════ */
.stButton > button {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.5rem 1.1rem !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    border-color: var(--border-hover) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm) !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.98);
}

/* ═══════════════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: rgba(22, 22, 24, 0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-top: 12px !important;
    transition: border-color 0.25s ease !important;
}

[data-testid="stExpander"]:hover {
    border-color: var(--border-hover) !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}

/* ═══════════════════════════════════════════════
   STATUS CONTAINER (Thinking steps)
   ═══════════════════════════════════════════════ */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ═══════════════════════════════════════════════
   TEXT INPUT (sidebar)
   ═══════════════════════════════════════════════ */
[data-testid="stTextInput"] input {
    background: var(--bg-base) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xs) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: border-color 0.25s ease !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-border) !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════
   FILE UPLOADER
   ═══════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    border-radius: var(--radius-sm) !important;
}

/* ═══════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.05); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.1); }

/* ═══════════════════════════════════════════════
   MISC
   ═══════════════════════════════════════════════ */
hr { border-color: var(--border) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; font-size: 0.78rem !important; }
[data-testid="stAlert"] { border-radius: var(--radius-sm) !important; }
[data-testid="stSpinner"] { color: var(--accent) !important; }
#cursor-glow, #particle-canvas, #flash-overlay { display: none; }

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# JavaScript: subtle cursor glow + auto-scroll
# ──────────────────────────────────────────────
components.html("""
<script>
(function() {
    const stDoc = window.parent.document;

    /* ── CURSOR GLOW (subtle emerald) ── */
    const existingGlow = stDoc.querySelectorAll('#cursor-glow-live');
    existingGlow.forEach(el => el.remove());

    const glowEl = stDoc.createElement('div');
    glowEl.id = 'cursor-glow-live';
    glowEl.style.cssText = `
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9998;
        transition: background 0.2s ease;
    `;
    stDoc.body.appendChild(glowEl);

    stDoc.addEventListener('mousemove', function(e) {
        const x = e.clientX, y = e.clientY;
        glowEl.style.background =
            `radial-gradient(500px circle at ${x}px ${y}px,
             rgba(62, 207, 142, 0.03),
             rgba(255, 255, 255, 0.008),
             transparent 60%)`;
    });

    /* ── FLOATING PARTICLES (muted white) ── */
    const existingCanvas = stDoc.querySelectorAll('#particles-live');
    existingCanvas.forEach(el => el.remove());

    const canvas = stDoc.createElement('canvas');
    canvas.id = 'particles-live';
    canvas.style.cssText = `
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9997;
        opacity: 0.18;
    `;
    stDoc.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const particles = [];
    const PARTICLE_COUNT = 25;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.15,
            vy: (Math.random() - 0.5) * 0.15,
            r: Math.random() * 1.0 + 0.3,
            alpha: Math.random() * 0.25 + 0.05,
        });
    }

    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (const p of particles) {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${p.alpha})`;
            ctx.fill();
        }

        for (let i = 0; i < particles.length; i++) {
            for (let j = i+1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(255, 255, 255, ${0.03 * (1 - dist/100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animateParticles);
    }
    animateParticles();

    /* ── AUTO-SCROLL on new messages ── */
    const scrollObserver = new MutationObserver(function(mutations) {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                const isChatMsg = node.querySelector &&
                    (node.querySelector('[data-testid="chatAvatarIcon-user"]') ||
                     node.querySelector('[data-testid="chatAvatarIcon-assistant"]'));
                if (isChatMsg) {
                    setTimeout(() => {
                        const main = stDoc.querySelector('[data-testid="stMainBlockContainer"]');
                        if (main) {
                            const msgs = main.querySelectorAll('[data-testid="stChatMessage"]');
                            if (msgs.length > 0) {
                                msgs[msgs.length - 1].scrollIntoView({
                                    behavior: 'smooth',
                                    block: 'end'
                                });
                            }
                        }
                    }, 150);
                }
            }
        }
    });

    const mainBlock = stDoc.querySelector('[data-testid="stMainBlockContainer"]');
    if (mainBlock) {
        scrollObserver.observe(mainBlock, { childList: true, subtree: true });
    }
})();
</script>
""", height=0)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("Codebase Assistant")

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
                with st.spinner("Indexing..."):
                    try:
                        build_index(str(target))
                        st.session_state.engine = None
                        st.success(f"Index built from {target.name}/")
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
                        build_index(str(target))
                        st.session_state.engine = None
                        st.success("Index rebuilt from scratch.")
                    except Exception as e:
                        st.error(f"Rebuild failed: {e}")

    # Advanced: manual path
    with st.expander("Advanced: Manual Path"):
        manual_path = st.text_input(
            "Codebase folder path",
            value=DEFAULT_CODEBASE_PATH,
            help="Absolute path to a folder of .py files.",
        )
        if st.button("Use This Path", use_container_width=True):
            st.session_state["upload_path"] = manual_path
            st.success(f"Path set: {manual_path}")

    # ── Models ──
    st.markdown("---")
    st.subheader("Active Models")
    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Top-K:** `{TOP_K}`")

    # ── Chat History ──
    st.markdown("---")
    st.subheader("Chat History")

    if st.button("New Chat", use_container_width=True):
        if st.session_state.messages:
            _save_chat(st.session_state.chat_id, st.session_state.messages)
        st.session_state.messages = []
        st.session_state.chat_id = _generate_chat_id()
        st.rerun()

    saved_chats = _list_chats()
    if saved_chats:
        for chat in saved_chats[:15]:
            col_load, col_del = st.columns([5, 1])
            with col_load:
                display_title = chat["title"]
                label = f"{display_title}  ({chat['message_count']} msgs)"
                if st.button(label, key=f"load_{chat['id']}", use_container_width=True):
                    if st.session_state.messages:
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

    st.markdown("---")
    st.caption("Fully offline -- no data leaves this machine.")

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = _generate_chat_id()

if "engine" not in st.session_state:
    st.session_state.engine = None

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
st.caption("Ask anything -- codebase questions use your code, general questions get direct answers.")

if st.session_state.engine:
    st.markdown(
        '<span class="status-badge status-ready">&#x25CF; Index loaded -- ready</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-badge status-none">&#x25CB; No index -- upload files and build one</span>',
        unsafe_allow_html=True,
    )
    st.info("Upload Python files in the sidebar and click **Build Index** to get started.")

# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("mode"):
            if msg["mode"] == "code":
                st.markdown(
                    '<span class="mode-badge mode-code">&#x1F4C2; Codebase</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="mode-badge mode-general">&#x1F30D; General</span>',
                    unsafe_allow_html=True,
                )
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} files)"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s['name']}** (`{s['type']}`) -- "
                        f"`{s['file']}` line {s['line']}"
                        + (f" (score: {s['score']})" if s.get('score') else "")
                    )

# ──────────────────────────────────────────────
# Chat input with thinking steps
# ──────────────────────────────────────────────
question = st.chat_input("Ask about the codebase or anything else...")

if question:
    if not st.session_state.engine:
        st.warning("Please build an index first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.status("Processing your question...", expanded=True) as status:
                def on_progress(step_msg):
                    st.write(f"✓ {step_msg}")

                try:
                    on_progress("Reading question...")
                    answer, sources, mode = ask(
                        st.session_state.engine,
                        question,
                        progress_fn=on_progress,
                    )
                    status.update(label="Complete", state="complete", expanded=False)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
                    sources = []
                    mode = "error"
                    status.update(label="Error", state="error", expanded=False)

            if mode == "code":
                st.markdown(
                    '<span class="mode-badge mode-code">&#x1F4C2; Codebase</span>',
                    unsafe_allow_html=True,
                )
            elif mode == "general":
                st.markdown(
                    '<span class="mode-badge mode-general">&#x1F30D; General</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(answer)
            if sources:
                with st.expander(f"Sources ({len(sources)} files)"):
                    for s in sources:
                        st.markdown(
                            f"- **{s['name']}** (`{s['type']}`) -- "
                            f"`{s['file']}` line {s['line']}"
                            + (f" (score: {s['score']})" if s.get('score') else "")
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "mode": mode,
        })

        _save_chat(st.session_state.chat_id, st.session_state.messages)
