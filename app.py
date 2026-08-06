"""
CodebookLM v1.0 — Streamlit UI
Offline AI Codebase Assistant
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
    """Extract a smart title from the first user message."""
    _STRIP_PREFIXES = [
        "explain ", "what is ", "how does ", "how do ", "how to ",
        "what are ", "why is ", "why does ", "can you ", "please ",
        "tell me ", "describe ", "show me ", "give me ",
    ]
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip()
            lower = text.lower()
            for prefix in _STRIP_PREFIXES:
                if lower.startswith(prefix):
                    text = text[len(prefix):]
                    break
            # Capitalize first letter, truncate
            text = text[:1].upper() + text[1:] if text else text
            if len(text) > 40:
                text = text[:37] + "..."
            return text or "New chat"
    return "Empty chat"

def _save_chat(chat_id, messages, custom_title=None, pinned=None):
    """Save a chat session to disk as JSON."""
    if not messages:
        return
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    # Load existing data to preserve extra fields
    existing = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    data = {
        "id": chat_id,
        "title": custom_title or existing.get("custom_title") or _get_chat_title(messages),
        "custom_title": custom_title or existing.get("custom_title"),
        "pinned": pinned if pinned is not None else existing.get("pinned", False),
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
    """List all saved chats, pinned first then sorted by most recent."""
    chats = []
    for fname in os.listdir(CHAT_HISTORY_DIR):
        if fname.endswith(".json"):
            filepath = os.path.join(CHAT_HISTORY_DIR, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                chats.append({
                    "id": data.get("id", fname.replace(".json", "")),
                    "title": data.get("custom_title") or data.get("title", "Untitled"),
                    "updated": data.get("updated", ""),
                    "message_count": data.get("message_count", 0),
                    "pinned": data.get("pinned", False),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    # Pinned first, then by recency
    chats.sort(key=lambda c: (not c.get("pinned", False), c.get("updated", "")), reverse=False)
    chats.sort(key=lambda c: c.get("pinned", False), reverse=True)
    # Within each group, most recent first
    pinned = [c for c in chats if c.get("pinned")]
    unpinned = [c for c in chats if not c.get("pinned")]
    pinned.sort(key=lambda c: c.get("updated", ""), reverse=True)
    unpinned.sort(key=lambda c: c.get("updated", ""), reverse=True)
    return pinned + unpinned

def _delete_chat(chat_id):
    """Delete a saved chat from disk."""
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

def _format_export_content(messages, format="md"):
    """Format chat messages for export."""
    lines = []
    if format == "md":
        lines.append("# CodebookLM v1.0 Conversation Export\n")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")
        for msg in messages:
            role = "User" if msg["role"] == "user" else "CodebookLM"
            lines.append(f"### {role}\n")
            lines.append(f"{msg['content']}\n")
            if msg.get("sources"):
                lines.append("\n**Sources:**\n")
                for s in msg["sources"]:
                    lines.append(f"- `{s['file']}` -> `{s['name']}` (Line {s['line']})\n")
            lines.append("\n---\n")
    else:
        lines.append("CodebookLM v1.0 Conversation Export\n")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lines.append("=========================================================\n\n")
        for msg in messages:
            role = "User" if msg["role"] == "user" else "CodebookLM"
            lines.append(f"[{role}]\n")
            lines.append(f"{msg['content']}\n")
            if msg.get("sources"):
                lines.append("\nSources:\n")
                for s in msg["sources"]:
                    lines.append(f"  - {s['file']} -> {s['name']} (Line {s['line']})\n")
            lines.append("\n---------------------------------------------------------\n\n")
    return "".join(lines)

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


def _render_mode_badge(mode, confidence=None, num_sources=0):
    """Render an HTML mode badge with optional confidence indicator.
    
    Quality indicators are labeled as heuristics, never as model certainty.
    Title: 'Response Confidence Indicators' with disclaimer.
    """
    if mode == "code":
        if confidence is not None:
            pct = int(confidence * 100)
            if pct >= 70:
                bar_color = "var(--accent, #34D399)"
                level = "High"
            elif pct >= 40:
                bar_color = "var(--yellow-text, #fde047)"
                level = "Medium"
            else:
                bar_color = "#f87171"
                level = "Low"
            conf_html = (
                f'<div class="confidence-panel">'
                f'<div class="confidence-header">'
                f'<span class="mode-badge mode-code">&#x1F7E2; Codebase</span>'
                f'<span class="confidence-label">{level} Relevance · {pct}%</span>'
                f'</div>'
                f'<div class="confidence-bar-track">'
                f'<div class="confidence-bar-fill" style="width:{pct}%; background:{bar_color};"></div>'
                f'</div>'
                f'<div class="confidence-meta">'
                f'{num_sources} source{"s" if num_sources != 1 else ""} referenced'
                f' · <span class="confidence-disclaimer">Heuristic estimate based on retrieval similarity</span>'
                f'</div>'
                f'</div>'
            )
            return conf_html
        else:
            return (
                f'<span class="mode-badge mode-code">'
                f'&#x1F7E2; Codebase</span>'
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

def _render_response_actions(msg_index, first_user_msg=""):
    """Render follow-up action buttons under an assistant response.
    
    Actions prefill the chat input via session_state (never auto-invoke).
    """
    actions = [
        ("💡 Explain more", f"Explain this in more detail: {first_user_msg}"),
        ("📝 Show examples", f"Show code examples for: {first_user_msg}"),
        ("🔗 Related code", f"What related code is connected to: {first_user_msg}"),
    ]
    cols = st.columns(len(actions))
    for col, (label, prompt) in zip(cols, actions):
        with col:
            st.markdown('<div class="response-actions"><div class="action-btn">', unsafe_allow_html=True)
            if st.button(label, key=f"action_{msg_index}_{label[:6]}"):
                st.session_state["_pending_question"] = prompt
                st.rerun()
            st.markdown('</div></div>', unsafe_allow_html=True)

def _get_time_duration(start_iso):
    """Calculate duration string from iso start time."""
    try:
        start_dt = datetime.fromisoformat(start_iso)
        duration = datetime.now() - start_dt
        minutes = int(duration.total_seconds() // 60)
        return f"{minutes} min" if minutes > 0 else "< 1 min"
    except Exception:
        return "Unknown"

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="📘 CodebookLM v1.0",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS: Premium desktop application aesthetic (Linear/Notion inspired)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base: #0C0E14;
    --bg-surface: #12141C;
    --bg-sidebar: #101218;
    --bg-card: #181B25;
    --bg-elevated: #1E212D;
    --bg-overlay: #222638;
    --bg-input: #1A1D28;
    --bg-hover: #222638;
    --border: rgba(255, 255, 255, 0.06);
    --border-subtle: rgba(255, 255, 255, 0.04);
    --border-hover: rgba(255, 255, 255, 0.12);
    --border-active: rgba(110, 231, 183, 0.25);
    --accent: #34D399;
    --accent-secondary: #6EE7B7;
    --accent-dim: rgba(52, 211, 153, 0.08);
    --accent-border: rgba(52, 211, 153, 0.18);
    --cyan: #67E8F9;
    --cyan-dim: rgba(103, 232, 249, 0.06);
    --text-primary: #EAEDF3;
    --text-secondary: #8B95A8;
    --text-muted: #5C6478;
    --danger-dim: rgba(239, 68, 68, 0.08);
    --purple-dim: rgba(168, 85, 247, 0.08);
    --purple-text: #c4b5fd;
    --yellow-dim: rgba(234, 179, 8, 0.10);
    --yellow-text: #fde047;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.2);
    --shadow-card: 0 4px 20px rgba(0,0,0,0.3);
    --shadow-lg: 0 8px 40px rgba(0,0,0,0.4);
    --glass: rgba(255, 255, 255, 0.02);
    --glass-border: rgba(255, 255, 255, 0.05);
    --radius: 12px;
    --radius-sm: 8px;
    --radius-xs: 6px;
    --ease-smooth: cubic-bezier(0.25, 0.1, 0.25, 1);
    --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
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
    backdrop-filter: blur(12px) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] h1 {
    font-size: 1.15rem !important; font-weight: 600 !important;
    letter-spacing: -0.01em !important; margin-bottom: 24px !important;
    background: none !important; -webkit-text-fill-color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.72rem !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important; color: var(--text-muted) !important;
    margin-top: 24px !important; margin-bottom: 12px !important; font-weight: 600 !important;
}
section[data-testid="stSidebar"] code {
    background: var(--accent-dim) !important; color: var(--accent) !important;
    padding: 2px 7px !important; border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important;
    border: 1px solid var(--accent-border) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    font-size: 0.8rem !important; padding: 0.4rem 0.8rem !important;
    text-align: left !important; white-space: nowrap !important;
    overflow: hidden !important; text-overflow: ellipsis !important;
    border-radius: var(--radius-sm) !important;
}

/* ── TITLE ── */
h1 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important; 
}
h1.hero-title {
    font-size: 2.4rem !important; margin-bottom: 4px !important;
}
p.hero-subtitle {
    font-size: 1.1rem !important; color: var(--text-secondary) !important;
    margin-bottom: 16px !important; font-weight: 400 !important;
}
p.hero-desc {
    font-size: 0.95rem !important; color: var(--text-muted) !important;
}

/* ── STATUS BADGES ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 6px; font-size: 0.8em;
    font-weight: 500; font-family: 'Inter', sans-serif; border: 1px solid;
    margin-bottom: 20px;
}
.status-ready { background: var(--accent-dim); color: var(--accent-secondary); border-color: var(--accent-border); }
.status-none { background: var(--danger-dim); color: #fca5a5; border-color: rgba(239,68,68,0.15); }

/* ── MODE BADGES ── */
.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 6px; font-size: 0.72em;
    font-weight: 500; font-family: 'Inter', sans-serif;
    letter-spacing: 0.01em; margin-bottom: 8px; border: 1px solid;
}
.mode-code { background: var(--accent-dim); color: var(--accent-secondary); border-color: var(--accent-border); }
.mode-general { background: var(--purple-dim); color: var(--purple-text); border-color: rgba(168,85,247,0.15); }
.mode-mixed { background: var(--yellow-dim); color: var(--yellow-text); border-color: rgba(234,179,8,0.15); }

/* ── GENERAL NOTE ── */
.general-note {
    background: rgba(168, 85, 247, 0.04); border: 1px solid rgba(168,85,247,0.1);
    border-radius: var(--radius-sm); padding: 12px 16px; margin-top: 12px;
    font-size: 0.85em; color: var(--text-secondary);
}

/* ── INFO CARD (sidebar) ── */
.info-card {
    background: var(--bg-sidebar); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 16px; margin: 8px 0;
    font-size: 0.82em; line-height: 1.8;
}
.info-card .label { color: var(--text-muted); display: inline-block; width: 110px; }
.info-card .value { color: var(--text-primary); font-weight: 500; }
.info-card .accent { color: var(--text-primary); font-weight: 600; font-size: 0.9em; }
.info-card .dot-ok { color: var(--accent); margin-right: 4px; }
.info-card .dot-err { color: #ef4444; margin-right: 4px; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: var(--bg-base) !important; border: 1px solid transparent !important;
    border-radius: 0 !important; padding: 16px 8px !important;
    margin-bottom: 8px !important; box-shadow: none !important;
    color: var(--text-primary) !important; border-bottom: 1px solid var(--border) !important;
    animation: msgRiseIn 350ms var(--ease-smooth) both;
}
[data-testid="stChatMessage"] * { color: var(--text-primary) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { 
    background: rgba(255,255,255,0.015) !important; 
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid rgba(52,211,153,0.15) !important;
    transition: border-color 200ms var(--ease-smooth) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]):hover {
    border-left-color: rgba(52,211,153,0.35) !important;
}
@keyframes msgRiseIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── CODE BLOCKS ── */
[data-testid="stChatMessage"] pre {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; padding: 14px 18px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
    line-height: 1.6 !important; overflow-x: auto !important; margin: 12px 0 !important;
    position: relative !important;
}
[data-testid="stChatMessage"] code {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
}
[data-testid="stChatMessage"] p code {
    background: rgba(255,255,255,0.06) !important; padding: 2px 6px !important;
    border-radius: 4px !important; border: 1px solid transparent !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] { border-radius: var(--radius) !important; overflow: hidden; }
[data-testid="stChatInput"] textarea {
    background: var(--bg-input) !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: var(--radius) !important; color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important; padding: 14px 18px !important;
    font-size: 0.95rem !important; line-height: 1.5 !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stChatInput"] textarea:hover { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stChatInput"] textarea:focus { border-color: var(--text-secondary) !important; box-shadow: none !important; outline: none !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: rgba(255,255,255,0.04) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
    font-weight: 500 !important; font-family: 'Inter', sans-serif !important;
    padding: 0.45rem 1rem !important;
    transition: all 200ms var(--ease-smooth) !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.08) !important; border-color: var(--border-hover) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}
.stButton > button:active {
    transform: scale(0.97) translateY(0) !important;
    transition-duration: 80ms !important;
}
.stButton > button:focus-visible {
    box-shadow: 0 0 0 2px var(--accent-dim), 0 0 0 4px rgba(52,211,153,0.1) !important;
    outline: none !important;
}
.stButton > button:disabled, .stButton > button[disabled] {
    opacity: 0.35 !important; cursor: not-allowed !important;
    pointer-events: none !important; transform: none !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; border-color: transparent !important;
    text-align: left !important; box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: transparent !important; transform: none !important;
    box-shadow: none !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: transparent !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; margin-top: 12px !important;
}
[data-testid="stExpander"] summary { color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.85rem !important; }

/* ── STATUS CONTAINER ── */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] { border-radius: var(--radius-sm) !important; border: 1px dashed var(--border) !important; background: transparent !important; }

/* ── EXAMPLE QUESTIONS ── */
.example-btn button {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; font-size: 0.85rem !important;
    padding: 10px 14px !important; text-align: left !important; color: var(--text-secondary) !important;
    transition: all 200ms var(--ease-smooth) !important; width: 100% !important;
}
.example-btn button:hover {
    border-color: var(--border-hover) !important; color: var(--text-primary) !important;
    background: var(--bg-hover) !important; box-shadow: var(--shadow-sm) !important;
}
.example-btn button:active {
    transform: scale(0.98) !important; transition-duration: 80ms !important;
}
.example-btn button:focus-visible {
    box-shadow: 0 0 0 2px var(--accent-dim) !important; outline: none !important;
}

/* ── RESPONSE ACTIONS ── */
.response-actions {
    display: flex; gap: 8px; margin-top: 12px; padding-top: 10px;
    border-top: 1px solid var(--border-subtle, rgba(255,255,255,0.04));
    flex-wrap: wrap;
}
.response-actions .action-btn button {
    background: var(--glass, rgba(255,255,255,0.02)) !important;
    border: 1px solid var(--border, rgba(255,255,255,0.06)) !important;
    border-radius: 20px !important; padding: 4px 14px !important;
    font-size: 0.76rem !important; color: var(--text-muted, #5C6478) !important;
    font-weight: 500 !important; transition: all 200ms var(--ease-smooth) !important;
    white-space: nowrap !important;
}
.response-actions .action-btn button:hover {
    background: rgba(52,211,153,0.06) !important;
    border-color: var(--accent-border, rgba(52,211,153,0.18)) !important;
    color: var(--accent, #34D399) !important; transform: translateY(-1px) !important;
}
.response-actions .action-btn button:active {
    transform: scale(0.97) !important; transition-duration: 80ms !important;
}

/* ── CONFIDENCE INDICATOR ── */
.confidence-panel {
    padding: 10px 14px; margin-bottom: 10px;
    background: var(--glass, rgba(255,255,255,0.02));
    border: 1px solid var(--border-subtle, rgba(255,255,255,0.04));
    border-radius: var(--radius-sm, 8px);
}
.confidence-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
}
.confidence-label {
    font-size: 0.78rem; font-weight: 500;
    color: var(--text-secondary, #8B95A8);
}
.confidence-bar-track {
    width: 100%; height: 4px; border-radius: 2px;
    background: rgba(255,255,255,0.05);
    overflow: hidden; margin-bottom: 6px;
}
.confidence-bar-fill {
    height: 100%; border-radius: 2px;
    transition: width 600ms var(--ease-smooth, cubic-bezier(0.25,0.1,0.25,1));
}
.confidence-meta {
    font-size: 0.72rem; color: var(--text-muted, #5C6478);
}
.confidence-disclaimer {
    font-style: italic; opacity: 0.8;
}

/* ── ONBOARDING ── */
.onboarding {
    text-align: center; padding: 80px 20px; color: var(--text-secondary);
}
.onboarding h2 { color: var(--text-primary); font-size: 1.8rem; font-weight: 600; margin-bottom: 30px; letter-spacing: -0.01em; }
.onboarding .step-container { display: flex; flex-direction: column; align-items: center; gap: 12px; }
.onboarding .step {
    display: flex; align-items: center; justify-content: flex-start;
    background: transparent; border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 20px; width: 280px;
    font-size: 0.9em; font-weight: 500; color: var(--text-primary);
    transition: all 200ms var(--ease-smooth);
}
.onboarding .step:hover {
    border-color: var(--border-hover); transform: translateY(-2px);
    box-shadow: var(--shadow-sm); background: var(--glass);
}
.onboarding .step span { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-right: 12px; font-size: 0.85em; }

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 24px 0; margin-top: 60px;
    border-top: 1px solid var(--border-subtle); color: var(--text-muted);
    font-size: 0.78em; letter-spacing: 0.02em; font-weight: 400;
    opacity: 0.7;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* ── REDUCED MOTION ── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stSpinner"] { color: var(--text-secondary) !important; }

/* ── SPLASH SCREEN ── */
#cb-splash {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: 99999; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: var(--bg-base, #0F1115);
    animation: splashSequence 2.8s cubic-bezier(0.25, 0.1, 0.25, 1) forwards;
}
#cb-splash .splash-icon {
    font-size: 3.4rem; line-height: 1;
    animation: splashGlow 2.8s ease forwards;
    filter: brightness(0.6);
}
#cb-splash .splash-title {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 1.7rem; font-weight: 650; letter-spacing: -0.03em;
    color: #F5F5F5; margin-top: 20px;
    opacity: 0; animation: splashFadeUp 550ms 280ms ease forwards;
}
#cb-splash .splash-tagline {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 0.88rem; font-weight: 400; letter-spacing: 0.01em;
    color: #71717A; margin-top: 10px;
    opacity: 0; animation: splashFadeUp 550ms 480ms ease forwards;
}
#cb-splash .splash-divider {
    width: 52px; height: 1px;
    background: rgba(255,255,255,0.08); margin-top: 28px;
    opacity: 0; animation: splashFadeUp 450ms 620ms ease forwards;
}
@keyframes splashSequence {
    0%   { opacity: 0; }
    10%  { opacity: 1; }
    70%  { opacity: 1; }
    100% { opacity: 0; pointer-events: none; visibility: hidden; }
}
@keyframes splashGlow {
    0%   { transform: scale(0.90); filter: brightness(0.5); }
    14%  { transform: scale(1.0);  filter: brightness(1.0); }
    28%  { transform: scale(1.04); filter: brightness(1.2) drop-shadow(0 0 24px rgba(34,197,94,0.3)); }
    42%  { transform: scale(1.0);  filter: brightness(1.0) drop-shadow(0 0 0 transparent); }
    100% { transform: scale(1.0);  filter: brightness(1.0); }
}
@keyframes splashFadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
    #cb-splash, #cb-splash .splash-icon,
    #cb-splash .splash-title, #cb-splash .splash-tagline,
    #cb-splash .splash-divider {
        animation-duration: 0.01ms !important;
        animation-delay: 0ms !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Splash screen (first load only)
# ──────────────────────────────────────────────
if not st.session_state.get("_splash_shown", False):
    st.session_state["_splash_shown"] = True
    st.markdown(
        '<div id="cb-splash">'
        '<span class="splash-icon">📘</span>'
        '<span class="splash-title">CodebookLM</span>'
        '<span class="splash-tagline">Understand Any Codebase. Instantly.</span>'
        '<span class="splash-divider"></span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# JavaScript: auto-scroll + keyboard shortcuts + copy button
# ──────────────────────────────────────────────
components.html("""
<script>
(function() {
    const stDoc = window.parent.document;

    /* ── SPLASH CLEANUP ── */
    var splashEl = stDoc.getElementById('cb-splash');
    if (splashEl) {
        setTimeout(function() {
            if (splashEl && splashEl.parentNode) {
                splashEl.parentNode.removeChild(splashEl);
            }
        }, 3200);
    }

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
        if (e.ctrlKey && e.key === 'l') {
            e.preventDefault();
            const btns = stDoc.querySelectorAll('button');
            for (const b of btns) {
                if (b.textContent.trim() === 'New Chat') { b.click(); break; }
            }
        }
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
                position: absolute; top: 8px; right: 8px;
                background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                border-radius: 6px; color: #a1a1aa; font-size: 11px;
                padding: 4px 10px; cursor: pointer; z-index: 10;
                font-family: 'Inter', sans-serif; transition: all 0.15s ease;
            `;
            btn.addEventListener('mouseenter', function() {
                btn.style.background = 'rgba(255,255,255,0.1)';
                btn.style.color = '#f5f5f5';
            });
            btn.addEventListener('mouseleave', function() {
                btn.style.background = 'rgba(255,255,255,0.05)';
                btn.style.color = '#a1a1aa';
            });
            btn.addEventListener('click', function() {
                const code = pre.querySelector('code');
                const text = code ? code.textContent : pre.textContent;
                navigator.clipboard.writeText(text).then(function() {
                    btn.textContent = 'Copied';
                    btn.style.color = '#F5F5F5';
                    btn.style.background = 'rgba(255,255,255,0.2)';
                    setTimeout(function() { btn.textContent = 'Copy'; btn.style.color = '#a1a1aa'; btn.style.background = 'rgba(255,255,255,0.05)'; }, 1500);
                });
            });
            pre.style.position = 'relative';
            pre.appendChild(btn);
        });
    }

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
    st.markdown("<h1>📘 CodebookLM v1.0</h1>", unsafe_allow_html=True)

    # ── Upload ──
    uploaded_files = st.file_uploader(
        "Upload Codebase",
        accept_multiple_files=True,
        type=["py", "zip"],
        label_visibility="collapsed"
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
                st.error(f"Invalid path: {target_path}")
            else:
                with st.spinner("Indexing codebase..."):
                    try:
                        result = build_index(str(target))
                        st.session_state.engine = None
                        if isinstance(result, tuple) and len(result) == 2:
                            _, stats = result
                            st.session_state["index_stats"] = stats
                        st.success("Index ready.")
                    except Exception as e:
                        st.error(f"Failed: {e}")
    with col2:
        if st.button("Rebuild", use_container_width=True):
            target_path = st.session_state.get("upload_path", DEFAULT_CODEBASE_PATH)
            target = Path(target_path)
            if not target.is_dir():
                st.error(f"Invalid path: {target_path}")
            else:
                with st.spinner("Rebuilding..."):
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
                        st.success("Rebuilt.")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # ── Repository Dashboard ──
    st.markdown("---")
    st.markdown("<h3>Repository Dashboard</h3>", unsafe_allow_html=True)
    
    stats = st.session_state.get("index_stats") or get_index_stats()
    if stats:
        st.session_state["index_stats"] = stats
        st.markdown(
            f'<div class="info-card">'
            f'<span class="label">Repository</span> <span class="accent">{stats.get("repo_name", "N/A")}</span><br>'
            f'<span class="label">Language</span> <span class="value">Python</span><br>'
            f'<span class="label">Indexed Files</span> <span class="value">{stats.get("files_indexed", "?")}</span><br>'
            f'<span class="label">Chunks</span> <span class="value">{stats.get("chunks_generated", "?")}</span><br>'
            f'<span class="label">Build Time</span> <span class="value">{stats.get("elapsed_seconds", 0):.1f}s</span><br>'
            f'<span class="label">Last Indexed</span> <span class="value">{stats.get("timestamp", "N/A")[:10]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-card">'
            '<span class="label">Repository</span> <span class="value" style="color:#71717A;">Not Indexed</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── System Status ──
    st.markdown("---")
    st.markdown("<h3>System Status</h3>", unsafe_allow_html=True)

    gpu_info = _get_gpu_info()
    ollama_status = check_ollama_status()
    ollama_dot = "dot-ok" if ollama_status["running"] else "dot-err"
    ollama_text = "Running" if ollama_status["running"] else "Offline"
    gpu_dot = "dot-ok" if gpu_info["available"] else "dot-err"
    gpu_label = gpu_info["name"] if gpu_info["available"] else "CPU only"

    st.markdown(
        f'<div class="info-card">'
        f'<span class="label">Ollama</span> <span class="{ollama_dot}">●</span> <span class="value">{ollama_text}</span><br>'
        f'<span class="label">LLM</span> <span class="value">`{LLM_MODEL}`</span><br>'
        f'<span class="label">Embeddings</span> <span class="value">`{EMBEDDING_MODEL}`</span><br>'
        f'<span class="label">GPU</span> <span class="{gpu_dot}">●</span> <span class="value">{gpu_label}</span><br>'
        + (f'<span class="label">VRAM</span> <span class="value">{gpu_info["vram_free"]}</span><br>'
           if gpu_info["available"] else '')
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ── Session Information ──
    st.markdown("---")
    st.markdown("<h3>Session Information</h3>", unsafe_allow_html=True)
    msg_count = len(st.session_state.get("messages", []))
    q_count = sum(1 for m in st.session_state.get("messages", []) if m["role"] == "user")
    session_start = st.session_state.get("session_start", datetime.now().isoformat())
    
    st.markdown(
        f'<div class="info-card">'
        f'<span class="label">Questions</span> <span class="value">{q_count}</span><br>'
        f'<span class="label">Duration</span> <span class="value">{_get_time_duration(session_start)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    # ── Conversation History ──
    st.markdown("---")
    st.markdown("<h3>Conversations</h3>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("✦ New Chat", use_container_width=True):
            if st.session_state.get("messages"):
                _save_chat(st.session_state.chat_id, st.session_state.messages)
            st.session_state.messages = []
            st.session_state.chat_id = _generate_chat_id()
            st.session_state["session_start"] = datetime.now().isoformat()
            st.rerun()

    # Search filter
    chat_search = st.text_input(
        "Search", placeholder="Filter conversations...",
        label_visibility="collapsed", key="chat_search_input"
    )

    saved_chats = _list_chats()
    if chat_search:
        saved_chats = [c for c in saved_chats if chat_search.lower() in c["title"].lower()]

    if saved_chats:
        for chat in saved_chats[:15]:
            # Relative timestamp
            try:
                updated_dt = datetime.fromisoformat(chat["updated"])
                delta = datetime.now() - updated_dt
                if delta.total_seconds() < 60:
                    time_label = "just now"
                elif delta.total_seconds() < 3600:
                    time_label = f"{int(delta.total_seconds() // 60)}m ago"
                elif delta.total_seconds() < 86400:
                    time_label = f"{int(delta.total_seconds() // 3600)}h ago"
                elif delta.days == 1:
                    time_label = "Yesterday"
                else:
                    time_label = f"{delta.days}d ago"
            except (ValueError, TypeError):
                time_label = ""

            pin_icon = "📌 " if chat.get("pinned") else ""
            is_active = chat["id"] == st.session_state.get("chat_id", "")

            # Chat entry row
            c_load, c_pin, c_del = st.columns([7, 1, 1])
            with c_load:
                label = f"{pin_icon}{chat['title']}"
                if time_label:
                    label += f" · {time_label}"
                if st.button(label, key=f"load_{chat['id']}", use_container_width=True,
                             disabled=is_active):
                    if st.session_state.get("messages"):
                        _save_chat(st.session_state.chat_id, st.session_state.messages)
                    loaded = _load_chat(chat["id"])
                    if loaded:
                        st.session_state.messages = loaded["messages"]
                        st.session_state.chat_id = chat["id"]
                        st.rerun()
            with c_pin:
                pin_label = "📌" if not chat.get("pinned") else "✕"
                if st.button(pin_label, key=f"pin_{chat['id']}"):
                    loaded = _load_chat(chat["id"])
                    if loaded:
                        new_pinned = not chat.get("pinned", False)
                        _save_chat(chat["id"], loaded["messages"], pinned=new_pinned)
                        st.rerun()
            with c_del:
                if st.button("✕", key=f"del_{chat['id']}"):
                    _delete_chat(chat["id"])
                    if chat["id"] == st.session_state.get("chat_id"):
                        st.session_state.messages = []
                        st.session_state.chat_id = _generate_chat_id()
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    # ── Export Conversation ──
    st.markdown("---")
    st.markdown("<h3>Export Conversation</h3>", unsafe_allow_html=True)
    
    if st.session_state.get("messages"):
        date_str = datetime.now().strftime("%Y-%m-%d")
        export_md = _format_export_content(st.session_state.messages, format="md")
        export_txt = _format_export_content(st.session_state.messages, format="txt")
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="Markdown",
                data=export_md,
                file_name=f"CodebookLM_v1.0_Conversation_{date_str}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with c2:
            st.download_button(
                label="Text",
                data=export_txt,
                file_name=f"CodebookLM_v1.0_Conversation_{date_str}.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.caption("Chat is empty.")

    # ── About Panel ──
    st.markdown("---")
    with st.expander("About CodebookLM", expanded=False):
        st.markdown(
            "**📘 CodebookLM v1.0**<br>"
            "Offline AI Codebase Assistant<br><br>"
            "**Mission:**<br>"
            "Understand software repositories while keeping all source code completely private using local AI.<br><br>"
            "**Built With:**<br>"
            "• Streamlit<br>"
            "• Ollama<br>"
            "• LlamaIndex<br>"
            "• ChromaDB<br>"
            "• Devstral Small 2<br>"
            "• Local Embeddings",
            unsafe_allow_html=True
        )

# ──────────────────────────────────────────────
# Session state init
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

# Empty state: Onboarding view or Example Questions
if not st.session_state.messages:
    if not st.session_state.engine:
        # Onboarding
        st.markdown(
            '<div class="onboarding">'
            '<h2>Welcome to CodebookLM</h2>'
            '<div class="step-container">'
            '<div class="step"><span>①</span> Upload Repository</div>'
            '<div class="step"><span>②</span> Build Semantic Index</div>'
            '<div class="step"><span>③</span> Ask Questions</div>'
            '<div class="step"><span>④</span> Inspect Sources</div>'
            '<div class="step"><span>⑤</span> Understand Architecture</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Hero section when ready
        st.markdown('<h1 class="hero-title">📘 CodebookLM v1.0</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Offline AI Codebase Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-desc">Understand • Explore • Explain • Navigate software repositories using Local LLMs.</p>', unsafe_allow_html=True)
        
        st.markdown(
            '<span class="status-badge status-ready">● Index loaded — Ready</span>',
            unsafe_allow_html=True,
        )
        
        st.markdown("<br><h4>Try an example question</h4>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state["_pending_question"] = q
                    st.rerun()

# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("mode"):
            badge_html = _render_mode_badge(msg["mode"], msg.get("confidence"), len(msg.get("sources", [])))
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []), msg.get("mode", ""))
            # Find the user question that preceded this response
            user_q = ""
            if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                user_q = st.session_state.messages[i-1]["content"][:60]
            _render_response_actions(i, user_q)

# ──────────────────────────────────────────────
# Chat input with thinking steps + streaming
# ──────────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)
question = pending or st.chat_input("Ask about the codebase...")

if question:
    if not st.session_state.engine:
        st.warning("Please build an index first using the sidebar.")
    else:
        # Capture recent history for context before appending the new user message
        history_for_ask = [
            {"question": m["content"], "answer": st.session_state.messages[i+1]["content"]}
            for i, m in enumerate(st.session_state.messages[:-1])
            if m["role"] == "user" and st.session_state.messages[i+1]["role"] == "assistant"
        ]

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer with thinking steps
        with st.chat_message("assistant"):
            with st.status("Processing...", expanded=True) as status:
                def on_progress(step_msg):
                    st.write(f"✓ {step_msg}")

                try:
                    on_progress("Reading question...")
                    answer, sources, mode, best_score = ask(
                        st.session_state.engine,
                        question,
                        history=history_for_ask,
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
            badge_html = _render_mode_badge(mode, confidence, len(sources))
            if badge_html:
                st.markdown(badge_html, unsafe_allow_html=True)

            st.markdown(answer)
            _render_sources(sources, mode)
            _render_response_actions(len(st.session_state.messages), question[:60])

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
# Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    '📘 CodebookLM v1.0 &nbsp;&nbsp;|&nbsp;&nbsp; Offline &nbsp;&nbsp;•&nbsp;&nbsp; Private &nbsp;&nbsp;•&nbsp;&nbsp; Local AI &nbsp;&nbsp;•&nbsp;&nbsp; No Internet Required'
    '</div>',
    unsafe_allow_html=True,
)
