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
from retrieval.repo_tree import get_cached_repo_tree, extract_file_structure, extract_dependencies
from retrieval.background_tasks import BackgroundTaskManager
from config import (
    DEFAULT_CODEBASE_PATH, LLM_MODEL, TOP_K, EMBEDDING_MODEL,
    DATA_DIR, RELEVANCE_THRESHOLD,
    STYLE_PROFILES, DEFAULT_STYLE_PROFILE,
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


def _bar_color(val):
    if val >= 75: return "#34D399"
    if val >= 55: return "#6EE7B7"
    if val >= 35: return "#FDE047"
    return "#F87171"


def _render_confidence_panel(quality_info, mode, confidence=None, source_count=0):
    """Render Response Confidence Indicators panel.
    
    Exact Title: Response Confidence Indicators
    Exact Disclaimer: This is a heuristic based on retrieval quality and context coverage. It is NOT the model's confidence.
    """
    if not quality_info:
        # Fallback for historical chats without quality_info
        if mode == "code" and confidence is not None:
            pct = int(min(max(confidence * 100, 0), 100))
            level = "High" if pct >= 75 else ("Good" if pct >= 55 else ("Fair" if pct >= 35 else "Low"))
            color = _bar_color(pct)
            return (
                f'<div class="confidence-panel">'
                f'<div class="confidence-title"><span>Response Confidence Indicators</span> <span class="mode-badge mode-code">● Codebase</span></div>'
                f'<div class="confidence-overall-row">'
                f'<span>Retrieval Relevance</span><strong style="color:{color};">{level} · {pct}%</strong>'
                f'</div>'
                f'<div class="confidence-meta-row"><span>{source_count} source{"s" if source_count != 1 else ""} referenced</span></div>'
                f'<div class="confidence-disclaimer">This is a heuristic based on retrieval quality and context coverage. It is NOT the model\'s confidence.</div>'
                f'</div>'
            )
        elif mode == "general":
            return (
                f'<div class="confidence-panel">'
                f'<div class="confidence-title"><span>Response Confidence Indicators</span> <span class="mode-badge mode-general">● General Knowledge</span></div>'
                f'<div class="general-note" style="margin:0 0 8px 0;">No sufficiently relevant repository context was retrieved for this question.</div>'
                f'<div class="confidence-disclaimer">This is a heuristic based on retrieval quality and context coverage. It is NOT the model\'s confidence.</div>'
                f'</div>'
            )
        return ""

    grounding = quality_info.get("grounding", 0.0)
    coverage = quality_info.get("coverage", 0.0)
    specificity = quality_info.get("specificity", 0.0)
    overall_score = quality_info.get("overall_score", 0.0)
    overall_level = quality_info.get("overall", "Low")
    retrieval_relevance = quality_info.get("retrieval_relevance", 0.0)
    suggestions = quality_info.get("suggestions", [])
    src_count = quality_info.get("source_count", source_count)

    overall_color = _bar_color(overall_score)
    badge_cls = "mode-code" if mode == "code" else "mode-general"
    badge_label = "● Codebase" if mode == "code" else "● General Knowledge"

    suggestions_html = ""
    if suggestions:
        sug_items = "".join(f"<li>{s}</li>" for s in suggestions)
        suggestions_html = (
            f'<div class="confidence-suggestions">'
            f'<strong>Suggestions:</strong>'
            f'<ul>{sug_items}</ul>'
            f'</div>'
        )

    if mode == "general":
        return (
            f'<div class="confidence-panel">'
            f'<div class="confidence-title"><span>Response Confidence Indicators</span> <span class="mode-badge {badge_cls}">{badge_label}</span></div>'
            f'<div class="general-note" style="margin:0 0 8px 0;">'
            f'No sufficiently relevant repository context was retrieved for this question.'
            f'</div>'
            f'<div class="confidence-overall-row">'
            f'<span>Query Specificity</span><strong style="color:{_bar_color(specificity)};">{specificity:.0f}%</strong>'
            f'</div>'
            f'{suggestions_html}'
            f'<div class="confidence-disclaimer">'
            f'This is a heuristic based on retrieval quality and context coverage. It is NOT the model\'s confidence.'
            f'</div>'
            f'</div>'
        )

    return (
        f'<div class="confidence-panel">'
        f'<div class="confidence-title"><span>Response Confidence Indicators</span> <span class="mode-badge {badge_cls}">{badge_label}</span></div>'
        f'<div class="confidence-grid">'
        f'<div class="confidence-metric-card">'
        f'<div class="metric-label"><span>Grounding</span><span class="metric-val">{grounding:.0f}%</span></div>'
        f'<div class="metric-bar-track"><div class="metric-bar-fill" style="width:{grounding}%; background:{_bar_color(grounding)};"></div></div>'
        f'</div>'
        f'<div class="confidence-metric-card">'
        f'<div class="metric-label"><span>Coverage</span><span class="metric-val">{coverage:.0f}%</span></div>'
        f'<div class="metric-bar-track"><div class="metric-bar-fill" style="width:{coverage}%; background:{_bar_color(coverage)};"></div></div>'
        f'</div>'
        f'<div class="confidence-metric-card">'
        f'<div class="metric-label"><span>Specificity</span><span class="metric-val">{specificity:.0f}%</span></div>'
        f'<div class="metric-bar-track"><div class="metric-bar-fill" style="width:{specificity}%; background:{_bar_color(specificity)};"></div></div>'
        f'</div>'
        f'</div>'
        f'<div class="confidence-overall-row">'
        f'<span>Overall Quality</span>'
        f'<strong style="color:{overall_color};">{overall_level} · {overall_score:.0f}%</strong>'
        f'</div>'
        f'<div class="confidence-meta-row">'
        f'<span>Sources: {src_count} retrieved chunks</span>'
        f'<span>Retrieval Relevance: {retrieval_relevance:.0f}%</span>'
        f'</div>'
        f'{suggestions_html}'
        f'<div class="confidence-disclaimer">'
        f'This is a heuristic based on retrieval quality and context coverage. It is NOT the model\'s confidence.'
        f'</div>'
        f'</div>'
    )


def _render_compression_notice(context_truncated):
    """Render structured compression notice or legacy truncation note."""
    if not context_truncated:
        return ""
    if isinstance(context_truncated, dict) and context_truncated.get("was_compressed"):
        orig = context_truncated.get("original_chars", 0)
        comp = context_truncated.get("compressed_chars", 0)
        symbols = context_truncated.get("preserved_symbols", [])
        sym_text = f" ({', '.join(symbols[:3])})" if symbols else ""
        return (
            f'<div class="general-note" style="border-color:rgba(234,179,8,0.2); background:rgba(234,179,8,0.03); color:#fde047;">'
            f'⚠ <strong>Context Compressed</strong> — {orig:,} → {comp:,} characters. '
            f'Important symbols{sym_text}, signatures, imports, docstrings and relevant bodies were preserved.'
            f'</div>'
        )
    elif context_truncated is True:
        return (
            f'<div class="general-note" style="border-color:rgba(234,179,8,0.2); background:rgba(234,179,8,0.03); color:#fde047;">'
            f'⚠ <strong>Context Trimmed</strong> — Conversation history was truncated to fit the model context window.'
            f'</div>'
        )
    return ""


def _render_sources(sources, mode):
    """Render source attribution section."""
    if mode == "code" and sources:
        with st.expander(f"Sources ({len(sources)} files)", expanded=False):
            for s in sources:
                score_text = f" — score: {s['score']}" if s.get("score") else ""
                st.markdown(
                    f"✓ **{s['file']}** → `{s['name']}` "
                    f"(`{s['type']}`, Line {s['line']}){score_text}"
                )
    elif mode == "general":
        st.markdown(
            '<div class="general-note">'
            '<strong>General Knowledge</strong> — '
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
        ("Explain more", f"Explain this in more detail: {first_user_msg}"),
        ("Show examples", f"Show code examples for: {first_user_msg}"),
        ("Related code", f"What related code is connected to: {first_user_msg}"),
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
icon_path = "assets/logo.png" if os.path.exists("assets/logo.png") else None
st.set_page_config(
    page_title="CodebookLM v1.0 — Offline AI Codebase Assistant",
    page_icon=icon_path if icon_path else "assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS: Premium desktop application aesthetic — Black + Emerald
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');


:root {
    color-scheme: dark !important;
}

html, body {
    background-color: #050507 !important;
}

header[data-testid="stHeader"],
div[data-testid="stAppHeader"],
div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottomBlockContainer"] > div,
div[data-testid="stChatInputContainer"],
div[data-testid="stChatFloatingInputContainer"],
.stChatFloatingInputContainer,
footer {
    background-color: transparent !important;
    background: transparent !important;
}
:root {
    --bg-base: #050507;
    --bg-surface: #0A0B0F;
    --bg-sidebar: #08090D;
    --bg-card: #0F1014;
    --bg-elevated: #141519;
    --bg-overlay: #1A1B22;
    --bg-input: #0D0E13;
    --bg-hover: #16171E;
    --border: rgba(255, 255, 255, 0.06);
    --border-subtle: rgba(255, 255, 255, 0.03);
    --border-hover: rgba(52, 211, 153, 0.2);
    --border-active: rgba(52, 211, 153, 0.35);
    --accent: #34D399;
    --accent-bright: #5EEAD4;
    --accent-secondary: #6EE7B7;
    --accent-dim: rgba(52, 211, 153, 0.07);
    --accent-glow: rgba(52, 211, 153, 0.12);
    --accent-border: rgba(52, 211, 153, 0.18);
    --cyan: #67E8F9;
    --cyan-dim: rgba(103, 232, 249, 0.06);
    --text-primary: #F0F2F5;
    --text-secondary: #8B95A8;
    --text-muted: #4E5668;
    --danger-dim: rgba(239, 68, 68, 0.08);
    --purple-dim: rgba(168, 85, 247, 0.08);
    --purple-text: #c4b5fd;
    --yellow-dim: rgba(234, 179, 8, 0.10);
    --yellow-text: #fde047;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.35);
    --shadow-card: 0 4px 24px rgba(0,0,0,0.5);
    --shadow-lg: 0 8px 48px rgba(0,0,0,0.6);
    --shadow-glow: 0 0 20px rgba(52,211,153,0.08);
    --glass: rgba(255, 255, 255, 0.02);
    --glass-border: rgba(255, 255, 255, 0.04);
    --radius: 12px;
    --radius-sm: 8px;
    --radius-xs: 6px;
    --ease-smooth: cubic-bezier(0.25, 0.1, 0.25, 1);
    --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
    --ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

/* ═══════ GLOBAL RESET ═══════ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    line-height: 1.6 !important;
}

/* Force true black background everywhere */
.stApp {
    background: var(--bg-base) !important;
    background-image: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(52,211,153,0.02) 0%, transparent 70%) !important;
    color: var(--text-primary) !important;
}
.stApp > header { background: transparent !important; }

/* Main container */
.stMainBlockContainer {
    max-width: 880px !important;
    padding: 2rem 1.5rem !important;
}

/* Force dark backgrounds on all Streamlit elements */
div[data-testid="stAppViewBlockContainer"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
.main .block-container {
    background: transparent !important;
}

/* ═══════ SIDEBAR ═══════ */
section[data-testid="stSidebar"] {
    background: #08090D !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
}
section[data-testid="stSidebar"] > div { 
    background: #08090D !important;
    padding-top: 1.2rem !important;
}
section[data-testid="stSidebar"] * { 
    color: var(--text-primary) !important; 
}

/* Sidebar Brand Header */
.sidebar-brand {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0 16px 0;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.brand-title-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
}
.brand-gem {
    width: 8px;
    height: 8px;
    background: #34D399;
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(52, 211, 153, 0.7);
    display: inline-block;
}
.brand-title {
    font-size: 1.05rem;
    font-weight: 650;
    letter-spacing: -0.02em;
    color: #F0F2F5;
}
.brand-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    color: #34D399;
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.2);
    padding: 2px 7px;
    border-radius: 12px;
    letter-spacing: 0.02em;
}

/* Sidebar Section Headers */
.sidebar-section-hdr {
    font-size: 0.68rem !important;
    font-weight: 650 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: #5C6478 !important;
    margin: 18px 0 8px 0 !important;
    padding: 0 2px !important;
}

.sidebar-subhdr {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #8B95A8 !important;
    display: block;
    margin: 8px 0 4px 0;
}

section[data-testid="stSidebar"] code {
    background: rgba(52, 211, 153, 0.06) !important; 
    color: #6EE7B7 !important;
    padding: 2px 6px !important; 
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important; 
    font-size: 0.75rem !important;
    border: 1px solid rgba(52, 211, 153, 0.15) !important;
}

/* Sidebar Buttons — Sleek dark card interaction */
section[data-testid="stSidebar"] .stButton > button {
    font-size: 0.78rem !important; 
    font-weight: 500 !important;
    padding: 0.42rem 0.75rem !important;
    text-align: left !important; 
    white-space: nowrap !important;
    overflow: hidden !important; 
    text-overflow: ellipsis !important;
    border-radius: 6px !important;
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
    color: #8B95A8 !important;
    transition: all 200ms cubic-bezier(0.25, 0.1, 0.25, 1) !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(52, 211, 153, 0.05) !important;
    border-color: rgba(52, 211, 153, 0.25) !important; 
    color: #F0F2F5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35), 0 0 12px rgba(52, 211, 153, 0.05) !important;
}

section[data-testid="stSidebar"] .stButton > button:active {
    transform: scale(0.97) translateY(0) !important;
    transition-duration: 80ms !important;
}

/* Tree & Code Structure items */
.tree-dir {
    color: #67E8F9 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    font-weight: 600;
}
.tree-file {
    color: #D1D5DB !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
}
.tree-hdr {
    color: #F0F2F5;
    font-size: 0.8rem;
}
.tree-summary {
    color: #5C6478;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-left: 4px;
}
.tree-loc {
    color: #5C6478;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
}
.tree-methods {
    color: #8B95A8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
}

.tag-class, .tag-func, .tag-pkg, .tag-local, .tag-std {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    padding: 1px 5px;
    border-radius: 4px;
    margin-right: 4px;
    display: inline-block;
}
.tag-class {
    background: rgba(168, 85, 247, 0.1);
    color: #C4B5FD;
    border: 1px solid rgba(168, 85, 247, 0.2);
}
.tag-func {
    background: rgba(52, 211, 153, 0.08);
    color: #6EE7B7;
    border: 1px solid rgba(52, 211, 153, 0.2);
}
.tag-pkg {
    background: rgba(103, 232, 249, 0.08);
    color: #67E8F9;
    border: 1px solid rgba(103, 232, 249, 0.2);
}
.tag-local {
    background: rgba(234, 179, 8, 0.08);
    color: #FDE047;
    border: 1px solid rgba(234, 179, 8, 0.2);
}
.tag-std {
    background: rgba(255, 255, 255, 0.04);
    color: #8B95A8;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Sidebar Creator Credit */
.sidebar-credit {
    text-align: center;
    padding: 20px 0 8px 0;
    font-size: 0.72rem;
    color: #4E5668;
    letter-spacing: 0.02em;
    border-top: 1px solid rgba(255, 255, 255, 0.03);
    margin-top: 24px;
}
.sidebar-credit strong {
    color: #8B95A8;
    font-weight: 500;
}

/* ═══════ TITLE ═══════ */
h1 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

.hero-version {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #34D399 !important;
    background: rgba(52, 211, 153, 0.08) !important;
    border: 1px solid rgba(52, 211, 153, 0.2) !important;
    padding: 3px 10px !important;
    border-radius: 12px !important;
    vertical-align: middle !important;
    letter-spacing: 0.02em !important;
    display: inline-block !important;
    margin-left: 6px !important;
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

/* ═══════ STATUS BADGES ═══════ */
.status-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 6px; font-size: 0.8em;
    font-weight: 500; font-family: 'Inter', sans-serif; border: 1px solid;
    margin-bottom: 20px;
}
.status-ready {
    background: var(--accent-dim); color: var(--accent-secondary);
    border-color: var(--accent-border);
    box-shadow: 0 0 12px rgba(52,211,153,0.06);
}
.status-none { background: var(--danger-dim); color: #fca5a5; border-color: rgba(239,68,68,0.15); }

/* ═══════ MODE BADGES ═══════ */
.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 6px; font-size: 0.72em;
    font-weight: 500; font-family: 'Inter', sans-serif;
    letter-spacing: 0.01em; margin-bottom: 8px; border: 1px solid;
}
.mode-code { background: var(--accent-dim); color: var(--accent-secondary); border-color: var(--accent-border); }
.mode-general { background: var(--purple-dim); color: var(--purple-text); border-color: rgba(168,85,247,0.15); }
.mode-mixed { background: var(--yellow-dim); color: var(--yellow-text); border-color: rgba(234,179,8,0.15); }

/* ═══════ GENERAL NOTE ═══════ */
.general-note {
    background: rgba(168, 85, 247, 0.04); border: 1px solid rgba(168,85,247,0.1);
    border-radius: var(--radius-sm); padding: 12px 16px; margin-top: 12px;
    font-size: 0.85em; color: var(--text-secondary);
}

/* ═══════ INFO CARD (sidebar) ═══════ */
.info-card {
    background: rgba(255, 255, 255, 0.018) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    margin: 8px 0 !important;
    font-size: 0.8rem !important;
    line-height: 1.75 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 200ms ease, box-shadow 200ms ease !important;
}
.info-card:hover {
    border-color: rgba(52, 211, 153, 0.15) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35), 0 0 12px rgba(52, 211, 153, 0.03) !important;
}
.info-card .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2px 0;
}
.info-card .label { 
    color: #5C6478; 
    font-size: 0.78rem; 
}
.info-card .value { 
    color: #EAEDF3; 
    font-weight: 500; 
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.info-card .accent { 
    color: #34D399; 
    font-weight: 600; 
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; 
}
.info-card .dot-ok { 
    color: #34D399; 
    text-shadow: 0 0 8px rgba(52, 211, 153, 0.6); 
    font-size: 0.65rem;
    vertical-align: middle;
    margin-right: 4px;
}
.info-card .dot-err { 
    color: #EF4444; 
    text-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
    font-size: 0.65rem;
    vertical-align: middle;
    margin-right: 4px;
}

/* ═══════ CHAT MESSAGES ═══════ */
[data-testid="stChatMessage"] {
    background: transparent !important; border: 1px solid transparent !important;
    border-radius: 0 !important; padding: 16px 8px !important;
    margin-bottom: 4px !important; box-shadow: none !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
    animation: msgRiseIn 350ms var(--ease-smooth) both;
}
[data-testid="stChatMessage"] * { color: var(--text-primary) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(255,255,255,0.01) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 2px solid rgba(52,211,153,0.12) !important;
    transition: border-color 250ms var(--ease-smooth) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]):hover {
    border-left-color: rgba(52,211,153,0.4) !important;
}
@keyframes msgRiseIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══════ CODE BLOCKS ═══════ */
[data-testid="stChatMessage"] pre {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; padding: 14px 18px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
    line-height: 1.6 !important; overflow-x: auto !important; margin: 12px 0 !important;
}
[data-testid="stChatMessage"] code {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important;
}
[data-testid="stChatMessage"] p code {
    background: rgba(52,211,153,0.06) !important; padding: 2px 6px !important;
    border-radius: 4px !important; border: 1px solid rgba(52,211,153,0.1) !important;
    color: var(--accent-secondary) !important;
}

/* ═══════ CHAT INPUT ═══════ */
[data-testid="stChatInput"] { border-radius: var(--radius) !important; overflow: hidden; }
[data-testid="stChatInput"] textarea {
    background: var(--bg-input) !important; border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: var(--radius) !important; color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important; padding: 14px 18px !important;
    font-size: 0.95rem !important; line-height: 1.5 !important;
    transition: all 0.3s var(--ease-smooth) !important;
}
[data-testid="stChatInput"] textarea:hover {
    border-color: rgba(52,211,153,0.15) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(52,211,153,0.3) !important;
    box-shadow: 0 0 0 3px rgba(52,211,153,0.06), 0 0 20px rgba(52,211,153,0.04) !important;
    outline: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

/* ═══════ BUTTONS — Premium with emerald glow ═══════ */
.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important; font-family: 'Inter', sans-serif !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 250ms var(--ease-smooth) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '' !important;
    position: absolute !important; top: 0 !important; left: 0 !important;
    width: 100% !important; height: 100% !important;
    background: linear-gradient(135deg, rgba(52,211,153,0.08), rgba(94,234,212,0.04)) !important;
    opacity: 0 !important;
    transition: opacity 250ms var(--ease-smooth) !important;
}
.stButton > button:hover {
    border-color: rgba(52,211,153,0.25) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3), 0 0 20px rgba(52,211,153,0.06) !important;
    color: var(--accent-secondary) !important;
}
.stButton > button:hover::before { opacity: 1 !important; }
.stButton > button:active {
    transform: scale(0.96) translateY(0) !important;
    transition-duration: 100ms !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
}
.stButton > button:focus-visible {
    box-shadow: 0 0 0 2px rgba(52,211,153,0.15), 0 0 0 4px rgba(52,211,153,0.06) !important;
    outline: none !important;
}
.stButton > button:disabled, .stButton > button[disabled] {
    opacity: 0.3 !important; cursor: not-allowed !important;
    pointer-events: none !important; transform: none !important;
}

/* ═══════ EXPANDER ═══════ */
[data-testid="stExpander"] {
    background: transparent !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; margin-top: 12px !important;
    transition: border-color 200ms var(--ease-smooth) !important;
}
[data-testid="stExpander"]:hover { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important; font-weight: 500 !important; font-size: 0.85rem !important;
}

/* ═══════ STATUS CONTAINER ═══════ */
[data-testid="stStatusWidget"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ═══════ FILE UPLOADER ═══════ */
[data-testid="stFileUploader"] {
    border-radius: var(--radius-sm) !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px dashed rgba(52, 211, 153, 0.2) !important;
    border-radius: 8px !important;
    transition: all 200ms var(--ease-smooth) !important;
}
[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(52, 211, 153, 0.04) !important;
    border-color: rgba(52, 211, 153, 0.45) !important;
}
[data-testid="stFileUploader"] * {
    color: #8B95A8 !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #EAEDF3 !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(52, 211, 153, 0.08) !important;
    border-color: rgba(52, 211, 153, 0.3) !important;
    color: #34D399 !important;
}

/* ═══════ EXAMPLE QUESTIONS — Emerald interactive cards ═══════ */
.example-btn button {
    background: var(--bg-card) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: var(--radius-sm) !important; font-size: 0.85rem !important;
    padding: 12px 16px !important; text-align: left !important;
    color: var(--text-secondary) !important;
    transition: all 280ms var(--ease-smooth) !important;
    width: 100% !important; position: relative !important; overflow: hidden !important;
}
.example-btn button::after {
    content: '→' !important; position: absolute !important;
    right: 14px !important; top: 50% !important; transform: translateY(-50%) translateX(4px) !important;
    opacity: 0 !important; color: var(--accent) !important;
    transition: all 250ms var(--ease-smooth) !important;
    font-size: 1em !important;
}
.example-btn button:hover {
    border-color: rgba(52,211,153,0.2) !important;
    color: var(--text-primary) !important;
    background: rgba(52,211,153,0.04) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25), 0 0 24px rgba(52,211,153,0.04) !important;
    transform: translateY(-2px) !important;
    padding-right: 32px !important;
}
.example-btn button:hover::after { opacity: 1 !important; transform: translateY(-50%) translateX(0) !important; }
.example-btn button:active {
    transform: scale(0.97) translateY(0) !important;
    transition-duration: 100ms !important;
}
.example-btn button:focus-visible {
    box-shadow: 0 0 0 2px rgba(52,211,153,0.1) !important; outline: none !important;
}

/* ═══════ RESPONSE ACTIONS ═══════ */
.response-actions {
    display: flex; gap: 8px; margin-top: 12px; padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.03);
    flex-wrap: wrap;
}
.response-actions .action-btn button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 20px !important; padding: 5px 14px !important;
    font-size: 0.76rem !important; color: var(--text-muted) !important;
    font-weight: 500 !important;
    transition: all 250ms var(--ease-smooth) !important;
    white-space: nowrap !important;
}
.response-actions .action-btn button:hover {
    background: rgba(52,211,153,0.06) !important;
    border-color: rgba(52,211,153,0.2) !important;
    color: var(--accent) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}
.response-actions .action-btn button:active {
    transform: scale(0.95) !important; transition-duration: 80ms !important;
}

/* ═══════ RESPONSE CONFIDENCE INDICATORS ═══════ */
.confidence-panel {
    padding: 12px 14px;
    margin: 8px 0 14px 0;
    background: rgba(255, 255, 255, 0.018);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.confidence-title {
    font-size: 0.74rem;
    font-weight: 650;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8B95A8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.confidence-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 10px;
}
.confidence-metric-card {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 6px;
    padding: 6px 8px;
}
.metric-label {
    font-size: 0.70rem;
    color: #5C6478;
    margin-bottom: 2px;
    display: flex;
    justify-content: space-between;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    color: #EAEDF3;
}
.metric-bar-track {
    width: 100%;
    height: 3px;
    border-radius: 2px;
    background: rgba(255, 255, 255, 0.06);
    overflow: hidden;
    margin-top: 3px;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 400ms ease;
}
.confidence-overall-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    margin-bottom: 8px;
    font-size: 0.76rem;
}
.confidence-meta-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.70rem;
    color: #5C6478;
    margin-bottom: 6px;
}
.confidence-suggestions {
    background: rgba(52, 211, 153, 0.03);
    border: 1px solid rgba(52, 211, 153, 0.12);
    border-radius: 6px;
    padding: 6px 10px;
    margin: 6px 0;
    font-size: 0.74rem;
    color: #A7F3D0;
}
.confidence-suggestions ul {
    margin: 3px 0 0 0;
    padding-left: 16px;
}
.confidence-disclaimer {
    font-size: 0.68rem;
    color: #5C6478;
    font-style: italic;
    line-height: 1.3;
    margin-top: 6px;
}

/* ═══════ ONBOARDING ═══════ */
.onboarding {
    text-align: center; padding: 80px 20px; color: var(--text-secondary);
}
.onboarding h2 {
    color: var(--text-primary); font-size: 1.8rem; font-weight: 600;
    margin-bottom: 30px; letter-spacing: -0.01em;
}
.onboarding .step-container { display: flex; flex-direction: column; align-items: center; gap: 12px; }
.onboarding .step {
    display: flex; align-items: center; justify-content: flex-start;
    background: transparent; border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 20px; width: 280px;
    font-size: 0.9em; font-weight: 500; color: var(--text-primary);
    transition: all 250ms var(--ease-smooth);
}
.onboarding .step:hover {
    border-color: rgba(52,211,153,0.2); transform: translateY(-2px);
    box-shadow: var(--shadow-card); background: rgba(52,211,153,0.02);
}
.onboarding .step span {
    color: var(--accent); font-family: 'JetBrains Mono', monospace;
    margin-right: 12px; font-size: 0.85em;
}

/* ═══════ FOOTER ═══════ */
.footer {
    text-align: center; padding: 24px 0; margin-top: 60px;
    border-top: 1px solid rgba(255,255,255,0.03); color: var(--text-muted);
    font-size: 0.78em; letter-spacing: 0.02em; font-weight: 400;
    opacity: 0.6;
}

/* ═══════ SCROLLBAR ═══════ */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(52,211,153,0.12); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(52,211,153,0.25); }

/* ═══════ SELECTBOX / DROPDOWN ═══════ */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-input) !important;
    border-color: rgba(255,255,255,0.06) !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-input) !important;
    border-color: rgba(255,255,255,0.06) !important;
}

/* ═══════ TEXT INPUT ═══════ */
input[type="text"], .stTextInput input {
    background: var(--bg-input) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}
input[type="text"]:focus, .stTextInput input:focus {
    border-color: rgba(52,211,153,0.25) !important;
    box-shadow: 0 0 0 2px rgba(52,211,153,0.06) !important;
}

/* ═══════ DOWNLOAD BUTTON ═══════ */
.stDownloadButton > button {
    background: rgba(52,211,153,0.06) !important;
    border: 1px solid rgba(52,211,153,0.15) !important;
    color: var(--accent-secondary) !important;
}
.stDownloadButton > button:hover {
    background: rgba(52,211,153,0.12) !important;
    border-color: rgba(52,211,153,0.3) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3), 0 0 20px rgba(52,211,153,0.08) !important;
    transform: translateY(-2px) !important;
}
.stDownloadButton > button:active {
    transform: scale(0.96) !important;
}

/* ═══════ ALERT/SUCCESS/WARNING/ERROR ═══════ */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ═══════ TABS (if any) ═══════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    transition: color 200ms !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ═══════ PROGRESS BAR ═══════ */
.stProgress > div > div {
    background: rgba(255,255,255,0.04) !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-bright)) !important;
}

/* ═══════ REDUCED MOTION ═══════ */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

hr { border-color: rgba(255,255,255,0.04) !important; margin: 1.5rem 0 !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stSpinner"] { color: var(--text-secondary) !important; }

/* ═══════ SPLASH SCREEN ═══════ */
#cb-splash {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: 99999; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: #030305;
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
    color: #4E5668; margin-top: 10px;
    opacity: 0; animation: splashFadeUp 550ms 480ms ease forwards;
}
#cb-splash .splash-divider {
    width: 52px; height: 1px;
    background: rgba(52,211,153,0.15); margin-top: 28px;
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
    28%  { transform: scale(1.04); filter: brightness(1.2) drop-shadow(0 0 30px rgba(52,211,153,0.35)); }
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
# Splash screen state machine (Idempotent, Exactly Once Per Session)
# ──────────────────────────────────────────────
if "_splash_started" not in st.session_state:
    st.session_state["_splash_started"] = False
if "_splash_shown" not in st.session_state:
    st.session_state["_splash_shown"] = False

# Detect & clear any query-parameter completion signal safely
if st.query_params.get("_splash_done") == "1":
    st.session_state["_splash_started"] = True
    st.session_state["_splash_shown"] = True
    try:
        del st.query_params["_splash_done"]
    except Exception:
        pass

# Render splash markup exactly once during initial session startup
if not st.session_state["_splash_shown"]:
    if not st.session_state["_splash_started"]:
        st.session_state["_splash_started"] = True
        st.markdown(
            '<div id="cb-splash">'
            '<span class="splash-icon">📘</span>'
            '<span class="splash-title">CodebookLM</span>'
            '<span class="splash-tagline">Understand Any Codebase. Instantly.</span>'
            '<span class="splash-divider"></span>'
            '</div>',
            unsafe_allow_html=True,
        )
    # Immediately transition lifecycle to shown to ensure subsequent reruns are strictly idempotent
    st.session_state["_splash_shown"] = True

# ──────────────────────────────────────────────
# JavaScript: auto-scroll + keyboard shortcuts + copy button
# ──────────────────────────────────────────────
components.html("""
<script>
(function() {
    const stDoc = window.parent.document;

    /* ── SPLASH CONTROLLER (Zero-Flicker Single Execution Guarantee) ── */
    try {
        var pStorage = window.parent.sessionStorage;
        var splashEl = stDoc.getElementById('cb-splash');
        var alreadyShown = pStorage && pStorage.getItem('cb_splash_shown') === '1';

        if (splashEl) {
            if (alreadyShown) {
                // Instantly remove duplicate splash if already played in this tab session
                if (splashEl.parentNode) {
                    splashEl.parentNode.removeChild(splashEl);
                }
            } else {
                // Record in tab sessionStorage and remove after 2.6s animation completes
                if (pStorage) pStorage.setItem('cb_splash_shown', '1');
                setTimeout(function() {
                    var el = stDoc.getElementById('cb-splash');
                    if (el && el.parentNode) {
                        el.parentNode.removeChild(el);
                    }
                }, 2600);
            }
        }
    } catch(err) {
        console.warn('Splash handler:', err);
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
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-title-wrap">
                <span class="brand-gem"></span>
                <span class="brand-title">CodebookLM</span>
            </div>
            <span class="brand-badge">v1.0</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown('<div class="sidebar-section-hdr">Repository Dashboard</div>', unsafe_allow_html=True)
    
    stats = st.session_state.get("index_stats") or get_index_stats()
    if stats:
        st.session_state["index_stats"] = stats

        # Get code structure stats from cached repo tree
        repo_root_dash = st.session_state.get("upload_path", DEFAULT_CODEBASE_PATH)
        class_count = 0
        func_count = 0
        if os.path.isdir(repo_root_dash):
            rd = get_cached_repo_tree(st.session_state, repo_root_dash)
            for fpath in rd.get("flat_files", []):
                fst = extract_file_structure(fpath)
                class_count += len(fst.get("classes", []))
                func_count += len(fst.get("functions", []))

        st.markdown(
            f'<div class="info-card">'
            f'<div class="info-row"><span class="label">Repository</span> <span class="accent">{stats.get("repo_name", "N/A")}</span></div>'
            f'<div class="info-row"><span class="label">Language</span> <span class="value">Python</span></div>'
            f'<div class="info-row"><span class="label">Indexed Files</span> <span class="value">{stats.get("files_indexed", "?")}</span></div>'
            f'<div class="info-row"><span class="label">Chunks</span> <span class="value">{stats.get("chunks_generated", "?")}</span></div>'
            f'<div class="info-row"><span class="label">Classes</span> <span class="value">{class_count}</span></div>'
            f'<div class="info-row"><span class="label">Functions</span> <span class="value">{func_count}</span></div>'
            f'<div class="info-row"><span class="label">Build Time</span> <span class="value">{stats.get("elapsed_seconds", 0):.1f}s</span></div>'
            f'<div class="info-row"><span class="label">Last Indexed</span> <span class="value">{stats.get("timestamp", "N/A")[:10]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # AI Insight actions (prefill only, never auto-invoke)
        st.markdown('<div class="sidebar-section-hdr">AI Insights</div>', unsafe_allow_html=True)
        insight_actions = [
            ("Architecture Overview", "Describe the high-level architecture and module structure of this codebase"),
            ("Design Patterns", "What design patterns are used in this codebase? List them with examples"),
            ("Execution & Entry Points", "What are the main entry points and how does execution flow through this codebase?"),
        ]
        for idx, (label, prompt) in enumerate(insight_actions):
            if st.button(label, key=f"insight_btn_{idx}", use_container_width=True):
                st.session_state["_pending_question"] = prompt
                st.rerun()
    else:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-row"><span class="label">Repository</span> <span class="value" style="color:var(--text-muted);">Not Indexed</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── System Status ──
    st.markdown('<div class="sidebar-section-hdr">System Diagnostics</div>', unsafe_allow_html=True)

    gpu_info = _get_gpu_info()
    ollama_status = check_ollama_status()
    ollama_dot = "dot-ok" if ollama_status["running"] else "dot-err"
    ollama_text = "Running" if ollama_status["running"] else "Offline"
    gpu_dot = "dot-ok" if gpu_info["available"] else "dot-err"
    gpu_label = gpu_info["name"] if gpu_info["available"] else "CPU only"

    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-row"><span class="label">Ollama</span> <span class="value"><span class="{ollama_dot}">●</span> {ollama_text}</span></div>'
        f'<div class="info-row"><span class="label">LLM</span> <span class="value"><code>{LLM_MODEL}</code></span></div>'
        f'<div class="info-row"><span class="label">Embeddings</span> <span class="value"><code>{EMBEDDING_MODEL}</code></span></div>'
        f'<div class="info-row"><span class="label">GPU</span> <span class="value"><span class="{gpu_dot}">●</span> {gpu_label}</span></div>'
        + (f'<div class="info-row"><span class="label">VRAM</span> <span class="value">{gpu_info["vram_free"]}</span></div>'
           if gpu_info["available"] else '')
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ── Session Information ──
    st.markdown('<div class="sidebar-section-hdr">Session Metrics</div>', unsafe_allow_html=True)
    msg_count = len(st.session_state.get("messages", []))
    q_count = sum(1 for m in st.session_state.get("messages", []) if m["role"] == "user")
    session_start = st.session_state.get("session_start", datetime.now().isoformat())
    
    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-row"><span class="label">Queries</span> <span class="value">{q_count}</span></div>'
        f'<div class="info-row"><span class="label">Duration</span> <span class="value">{_get_time_duration(session_start)}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Response Style ──
    st.markdown('<div class="sidebar-section-hdr">Response Style</div>', unsafe_allow_html=True)
    style_keys = list(STYLE_PROFILES.keys())
    style_names = [STYLE_PROFILES[k]["name"] for k in style_keys]
    current_idx = style_keys.index(
        st.session_state.get("style_profile", DEFAULT_STYLE_PROFILE)
    ) if st.session_state.get("style_profile", DEFAULT_STYLE_PROFILE) in style_keys else 0
    selected_style_name = st.selectbox(
        "Style", style_names, index=current_idx,
        label_visibility="collapsed", key="style_selectbox"
    )
    selected_key = style_keys[style_names.index(selected_style_name)]
    st.session_state["style_profile"] = selected_key
    st.caption(STYLE_PROFILES[selected_key]["description"])
    
    # ── Conversation History ──
    st.markdown('<div class="sidebar-section-hdr">Conversations</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("+ New Chat", use_container_width=True):
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

            pin_badge = "[PIN] " if chat.get("pinned") else ""
            is_active = chat["id"] == st.session_state.get("chat_id", "")

            # Chat entry row
            c_load, c_pin, c_del = st.columns([7, 1.2, 1.2])
            with c_load:
                label = f"{pin_badge}{chat['title']}"
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
                pin_label = "★" if not chat.get("pinned") else "☆"
                if st.button(pin_label, key=f"pin_{chat['id']}", help="Pin / Unpin"):
                    loaded = _load_chat(chat["id"])
                    if loaded:
                        new_pinned = not chat.get("pinned", False)
                        _save_chat(chat["id"], loaded["messages"], pinned=new_pinned)
                        st.rerun()
            with c_del:
                if st.button("×", key=f"del_{chat['id']}", help="Delete"):
                    _delete_chat(chat["id"])
                    if chat["id"] == st.session_state.get("chat_id"):
                        st.session_state.messages = []
                        st.session_state.chat_id = _generate_chat_id()
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    # ── Export Conversation ──
    st.markdown('<div class="sidebar-section-hdr">Export Session</div>', unsafe_allow_html=True)
    
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

    # ── Architecture Explorer ──
    st.markdown('<div class="sidebar-section-hdr">Architecture Explorer</div>', unsafe_allow_html=True)
    repo_root = st.session_state.get("upload_path", DEFAULT_CODEBASE_PATH)
    if os.path.isdir(repo_root):
        repo_data = get_cached_repo_tree(st.session_state, repo_root)
        tree_stats = repo_data.get("stats", {})
        if tree_stats:
            st.markdown(
                f'<div class="info-card">'
                f'<div class="info-row"><span class="label">Files</span> <span class="value">{tree_stats.get("total_files", 0)}</span></div>'
                f'<div class="info-row"><span class="label">Python</span> <span class="value">{tree_stats.get("py_files", 0)}</span></div>'
                f'<div class="info-row"><span class="label">Lines</span> <span class="value">{tree_stats.get("total_loc", 0):,}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # File tree in expander
        with st.expander("File Explorer", expanded=False):
            tree = repo_data.get("tree", {})
            if tree:
                def _render_tree(node, depth=0):
                    """Recursively render the file tree."""
                    for name in sorted(node.keys()):
                        info = node[name]
                        indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * depth
                        if info.get("is_dir"):
                            st.markdown(f"{indent}<span class='tree-dir'>/ {name}</span>", unsafe_allow_html=True)
                            if info.get("children"):
                                _render_tree(info["children"], depth + 1)
                        else:
                            st.markdown(f"{indent}<span class='tree-file'>{name}</span>", unsafe_allow_html=True)
                _render_tree(tree)
            else:
                st.caption("No files found.")

        # Code structure for .py files
        flat_files = repo_data.get("flat_files", [])
        if flat_files:
            with st.expander(f"Code Structure ({len(flat_files)} files)", expanded=False):
                for fpath in flat_files[:30]:  # Cap to avoid UI overload
                    fname = os.path.basename(fpath)
                    structure = extract_file_structure(fpath)
                    classes = structure.get("classes", [])
                    functions = structure.get("functions", [])
                    loc = structure.get("loc", 0)
                    summary = f"{len(classes)}C {len(functions)}F {loc}L"
                    st.markdown(f"<span class='tree-hdr'><strong>{fname}</strong></span> <span class='tree-summary'>{summary}</span>", unsafe_allow_html=True)
                    for cls in classes:
                        methods_str = ", ".join(cls["methods"][:5])
                        if len(cls["methods"]) > 5:
                            methods_str += f" +{len(cls['methods'])-5}"
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='tag-class'>class</span> <code>{cls['name']}</code> <span class='tree-loc'>L{cls['line']}</span> &rarr; <span class='tree-methods'>{methods_str}</span>", unsafe_allow_html=True)
                    for fn in functions:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='tag-func'>def</span> <code>{fn['name']}</code> <span class='tree-loc'>L{fn['line']}</span>", unsafe_allow_html=True)
                if len(flat_files) > 30:
                    st.caption(f"+ {len(flat_files) - 30} more files...")

            # Dependencies analysis
            if flat_files:
                with st.expander("Dependencies", expanded=False):
                    deps = extract_dependencies(flat_files)
                    if deps.get("third_party"):
                        st.markdown("<span class='sidebar-subhdr'>Third-Party</span>", unsafe_allow_html=True)
                        for pkg in deps["third_party"]:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='tag-pkg'>pkg</span> <code>{pkg}</code>", unsafe_allow_html=True)
                    if deps.get("local"):
                        st.markdown("<span class='sidebar-subhdr'>Local Modules</span>", unsafe_allow_html=True)
                        for mod in deps["local"]:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='tag-local'>local</span> <code>{mod}</code>", unsafe_allow_html=True)
                    if deps.get("stdlib"):
                        st.markdown("<span class='sidebar-subhdr'>Standard Library</span>", unsafe_allow_html=True)
                        for mod in deps["stdlib"]:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span class='tag-std'>std</span> <code>{mod}</code>", unsafe_allow_html=True)
                    if not any(deps.values()):
                        st.caption("No imports detected.")
    else:
        st.caption("Index a repository to explore its architecture.")

    # ── About Panel ──
    with st.expander("About CodebookLM", expanded=False):
        st.markdown(
            "<strong>CodebookLM v1.0</strong><br>"
            "<span style='color:#8B95A8;font-size:0.82rem;'>Offline AI Codebase Assistant</span><br><br>"
            "<span style='color:#5C6478;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:650;'>Mission</span><br>"
            "<span style='color:#8B95A8;font-size:0.78rem;'>Understand software repositories while keeping all source code completely private using local AI.</span><br><br>"
            "<span style='color:#5C6478;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;font-weight:650;'>Core Engine</span><br>"
            "<span style='color:#8B95A8;font-size:0.78rem;line-height:1.7;'>"
            "&bull; Streamlit UI<br>"
            "&bull; Ollama Engine<br>"
            "&bull; LlamaIndex RAG<br>"
            "&bull; ChromaDB Vector Store<br>"
            "&bull; Devstral Small 2 / Llama 3<br>"
            "&bull; Local AST Chunker"
            "</span>",
            unsafe_allow_html=True
        )

    # ── Creator Credit ──
    st.markdown(
        '<div class="sidebar-credit">'
        'Crafted with precision by <strong>Aayush Singh</strong>'
        '</div>',
        unsafe_allow_html=True,
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
        st.markdown('<h1 class="hero-title">CodebookLM <span class="hero-version">v1.0</span></h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Offline AI Codebase Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-desc">Understand &bull; Explore &bull; Explain &bull; Navigate software repositories using Local LLMs.</p>', unsafe_allow_html=True)
        
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
# Background Task State Handling
# ──────────────────────────────────────────────
_bg_manager = BackgroundTaskManager.get_instance()

# 1. Consume completed results
_bg_result = _bg_manager.consume_result(st.session_state.chat_id)
if _bg_result:
    _bg_confidence = _bg_result.get("best_score") if _bg_result.get("mode") == "code" else None
    st.session_state.messages.append({
        "role": "assistant",
        "content": _bg_result.get("answer", ""),
        "sources": _bg_result.get("sources", []),
        "mode": _bg_result.get("mode", "code"),
        "confidence": _bg_confidence,
        "quality_info": _bg_result.get("quality_info"),
        "context_truncated": _bg_result.get("context_truncated", False),
    })
    _save_chat(st.session_state.chat_id, st.session_state.messages)

# 2. Check for failed task errors
_bg_error = _bg_manager.clear_failed(st.session_state.chat_id)
if _bg_error:
    st.error(f"Generation failed: {_bg_error}")

# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Response Confidence Indicators Panel
            conf_panel_html = _render_confidence_panel(
                quality_info=msg.get("quality_info"),
                mode=msg.get("mode", "code"),
                confidence=msg.get("confidence"),
                source_count=len(msg.get("sources", [])),
            )
            if conf_panel_html:
                st.markdown(conf_panel_html, unsafe_allow_html=True)

            _render_sources(msg.get("sources", []), msg.get("mode", ""))

            # Compression notice if any
            comp_notice = _render_compression_notice(msg.get("context_truncated"))
            if comp_notice:
                st.markdown(comp_notice, unsafe_allow_html=True)

            # Follow-up actions
            user_q = ""
            if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                user_q = st.session_state.messages[i-1]["content"][:60]
            _render_response_actions(i, user_q)

# ──────────────────────────────────────────────
# Active Background Generation Indicator & Fragment Polling
# ──────────────────────────────────────────────
@st.fragment(run_every="1s")
def _render_generation_progress_fragment(chat_id):
    """Isolated fragment polling the background task without full page flicker."""
    manager = BackgroundTaskManager.get_instance()
    if manager.is_generating(chat_id):
        steps = manager.get_progress_steps(chat_id)
        current = manager.get_progress(chat_id) or "Processing..."
        
        with st.chat_message("assistant"):
            with st.status("Processing...", expanded=True) as status:
                for step in steps:
                    if step == current and step not in ("Generation complete", "Done!"):
                        st.write(f"● {step}")
                    else:
                        st.write(f"✓ {step}")
                
                col_sp, col_stop = st.columns([4, 1])
                with col_stop:
                    if st.button("Stop", key=f"stop_btn_frag_{chat_id}", use_container_width=True):
                        manager.cancel(chat_id)
                        st.rerun()
    else:
        # Task completed or cancelled — trigger top-level rerun to consume and display result
        st.rerun()

if _bg_manager.is_generating(st.session_state.chat_id):
    _render_generation_progress_fragment(st.session_state.chat_id)

# ──────────────────────────────────────────────
# Chat Input & Background Submission
# ──────────────────────────────────────────────
pending = st.session_state.pop("_pending_question", None)
question = pending or st.chat_input("Ask about the codebase...")

if question:
    if not st.session_state.engine:
        st.warning("Please build an index first using the sidebar.")
    elif _bg_manager.is_generating(st.session_state.chat_id):
        st.warning("A generation is already in progress for this conversation. Please wait or click Stop.")
    else:
        # Capture recent history for context before appending the new user message
        history_for_ask = [
            {"question": m["content"], "answer": st.session_state.messages[i+1]["content"]}
            for i, m in enumerate(st.session_state.messages[:-1])
            if m["role"] == "user" and i+1 < len(st.session_state.messages) and st.session_state.messages[i+1]["role"] == "assistant"
        ]

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        _save_chat(st.session_state.chat_id, st.session_state.messages)

        # Submit background task
        try:
            _bg_manager.submit(
                chat_id=st.session_state.chat_id,
                question=question,
                generate_fn=ask,
                query_engine=st.session_state.engine,
                history=history_for_ask,
                style_profile=st.session_state.get("style_profile", DEFAULT_STYLE_PROFILE),
            )
        except Exception as e:
            st.error(f"Failed to start generation: {e}")

        st.rerun()

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    '<strong>CodebookLM v1.0</strong> &nbsp;&nbsp;|&nbsp;&nbsp; Offline &nbsp;&bull;&nbsp; Private &nbsp;&bull;&nbsp; Local AI &nbsp;&bull;&nbsp; No Internet Required'
    '</div>',
    unsafe_allow_html=True,
)
