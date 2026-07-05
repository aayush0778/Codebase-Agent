"""
Streamlit Chat UI — Phase 4 (Enhanced)
Single-page app for querying a Python codebase using the local RAG pipeline.
No external network calls anywhere in this file.

Features:
  - Cursor-tracking radial glow effect
  - Cyan pulse animation on question submit
  - Green wave animation on answer received
  - Ambient floating particles background
  - Glassmorphism styling throughout

Launch with:
    python -m streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import os
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
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


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
# Massive CSS + JS injection for visual effects
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   IMPORTS
   ═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════
   ROOT VARIABLES & GLOBAL OVERRIDES
   ═══════════════════════════════════════════════ */
:root {
    /* Bind to Streamlit's native theme variables to ensure text/bg match the user's selected mode (Light/Dark) */
    --bg-primary: var(--background-color, #0a0e1a);
    --bg-secondary: var(--secondary-background-color, #0f1629);
    --bg-card: var(--secondary-background-color, rgba(15, 22, 41, 0.65));
    --bg-card-hover: var(--background-color, rgba(20, 30, 55, 0.80));
    --border-glow: rgba(0, 200, 255, 0.15);
    --border-subtle: rgba(128, 128, 128, 0.2);
    --text-primary: var(--text-color, #e8ecf5);
    --text-secondary: var(--text-color, #8892b0);
    --text-muted: var(--text-color, #5a6380);
    
    --accent-cyan: #00d4ff;
    --accent-blue: #3b82f6;
    --accent-purple: #a855f7;
    --accent-green: #22c55e;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --glow-cyan: 0 0 30px rgba(0, 212, 255, 0.3);
    --glow-green: 0 0 30px rgba(34, 197, 94, 0.3);
    --glow-purple: 0 0 30px rgba(168, 85, 247, 0.3);
}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Main app background */
.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp > header {
    background: transparent !important;
}

/* Main content area */
.stMainBlockContainer {
    max-width: 960px !important;
    padding-top: 2rem !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ═══════════════════════════════════════════════
   CURSOR GLOW LAYER (injected via JS below)
   ═══════════════════════════════════════════════ */
#cursor-glow {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    transition: opacity 0.3s ease;
}

/* ═══════════════════════════════════════════════
   ANIMATED PARTICLES BACKGROUND
   ═══════════════════════════════════════════════ */
#particle-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ═══════════════════════════════════════════════
   FLASH OVERLAYS (question & answer pulses)
   ═══════════════════════════════════════════════ */
#flash-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 9999;
    opacity: 0;
}

@keyframes flashCyan {
    0%   { opacity: 0.6; background: radial-gradient(ellipse at center, rgba(0,212,255,0.25) 0%, transparent 70%); }
    100% { opacity: 0;   background: radial-gradient(ellipse at center, rgba(0,212,255,0) 0%, transparent 70%); }
}

@keyframes flashGreen {
    0%   { opacity: 0.6; background: radial-gradient(ellipse at center, rgba(34,197,94,0.30) 0%, transparent 70%); }
    100% { opacity: 0;   background: radial-gradient(ellipse at center, rgba(34,197,94,0) 0%, transparent 70%); }
}

.flash-cyan  { animation: flashCyan 1.2s ease-out forwards; }
.flash-green { animation: flashGreen 1.5s ease-out forwards; }

/* ═══════════════════════════════════════════════
   TITLE STYLING
   ═══════════════════════════════════════════════ */
h1 {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue), var(--accent-purple)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    position: relative;
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
    font-size: 0.85em;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.02em;
    backdrop-filter: blur(12px);
    border: 1px solid;
    transition: all 0.3s ease;
}

.status-ready {
    background: rgba(16, 185, 129, 0.12);
    color: #6ee7b7;
    border-color: rgba(16, 185, 129, 0.25);
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.1),
                inset 0 0 20px rgba(16, 185, 129, 0.05);
    animation: readyPulse 3s ease-in-out infinite;
}

.status-none {
    background: rgba(244, 63, 94, 0.10);
    color: #fca5a5;
    border-color: rgba(244, 63, 94, 0.20);
    box-shadow: 0 0 20px rgba(244, 63, 94, 0.08);
}

@keyframes readyPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.1), inset 0 0 20px rgba(16, 185, 129, 0.05); }
    50%      { box-shadow: 0 0 35px rgba(16, 185, 129, 0.2), inset 0 0 30px rgba(16, 185, 129, 0.08); }
}

/* ═══════════════════════════════════════════════
   CHAT MESSAGES
   ═══════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    margin-bottom: 1rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: messageSlideIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    position: relative;
    overflow: hidden;
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"] * {
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"]:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-glow) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                0 0 0 1px rgba(0, 200, 255, 0.08);
    transform: translateY(-1px);
}

/* User messages: cyan left border */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--accent-cyan) !important;
}

/* Assistant messages: green left border */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--accent-emerald) !important;
}

@keyframes messageSlideIn {
    from {
        opacity: 0;
        transform: translateY(20px) scale(0.98);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

/* ═══════════════════════════════════════════════
   CHAT INPUT
   ═══════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    overflow: hidden;
}

[data-testid="stChatInput"] textarea {
    background: rgba(15, 22, 41, 0.80) !important;
    border: 1px solid rgba(0, 200, 255, 0.15) !important;
    border-radius: 16px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px 18px !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15),
                0 0 40px rgba(0, 212, 255, 0.08) !important;
}

/* ═══════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.12), rgba(59, 130, 246, 0.12)) !important;
    border: 1px solid rgba(0, 212, 255, 0.20) !important;
    border-radius: 12px !important;
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.22), rgba(59, 130, 246, 0.22)) !important;
    border-color: rgba(0, 212, 255, 0.40) !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.15),
                0 0 0 1px rgba(0, 212, 255, 0.1) !important;
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0px) scale(0.98);
}

/* ═══════════════════════════════════════════════
   EXPANDER (Sources)
   ═══════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: rgba(10, 14, 26, 0.5) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    margin-top: 0.8rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stExpander"]:hover {
    border-color: rgba(0, 200, 255, 0.12) !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

/* ═══════════════════════════════════════════════
   SPINNER (Thinking animation override)
   ═══════════════════════════════════════════════ */
[data-testid="stSpinner"] {
    animation: spinnerGlow 2s ease-in-out infinite;
}

@keyframes spinnerGlow {
    0%, 100% { filter: drop-shadow(0 0 4px rgba(0, 212, 255, 0.3)); }
    50%      { filter: drop-shadow(0 0 12px rgba(168, 85, 247, 0.5)); }
}

/* ═══════════════════════════════════════════════
   TEXT INPUT (sidebar)
   ═══════════════════════════════════════════════ */
[data-testid="stTextInput"] input {
    background: rgba(10, 14, 26, 0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: all 0.3s ease !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(0, 212, 255, 0.3) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.08) !important;
}

/* ═══════════════════════════════════════════════
   SIDEBAR SECTION HEADERS
   ═══════════════════════════════════════════════ */
section[data-testid="stSidebar"] h2 {
    font-size: 0.95rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
    margin-top: 1.5rem !important;
}

/* Sidebar model labels — monospace */
section[data-testid="stSidebar"] code {
    background: rgba(0, 212, 255, 0.08) !important;
    color: var(--accent-cyan) !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    border: 1px solid rgba(0, 212, 255, 0.12) !important;
}

/* Chat history buttons in sidebar */
section[data-testid="stSidebar"] .stButton > button {
    font-size: 0.78rem !important;
    padding: 0.35rem 0.8rem !important;
    text-align: left !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
/* ═══════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.15);
}

/* ═══════════════════════════════════════════════
   MISC STREAMLIT OVERRIDES
   ═══════════════════════════════════════════════ */
hr {
    border-color: var(--border-subtle) !important;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}

/* Success / Error / Warning alerts */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
}

/* ═══════════════════════════════════════════════
   MODE BADGES (Code RAG vs General Knowledge)
   ═══════════════════════════════════════════════ */
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.75em;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.03em;
    margin-bottom: 8px;
    backdrop-filter: blur(8px);
    border: 1px solid;
}

.mode-code {
    background: rgba(0, 212, 255, 0.10);
    color: #67e8f9;
    border-color: rgba(0, 212, 255, 0.20);
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.08);
}

.mode-general {
    background: rgba(168, 85, 247, 0.10);
    color: #c4b5fd;
    border-color: rgba(168, 85, 247, 0.20);
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.08);
}

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# JavaScript: cursor glow, particles, flash effects
# ──────────────────────────────────────────────
components.html("""
<div id="cursor-glow"></div>
<div id="particle-canvas"></div>
<div id="flash-overlay"></div>

<script>
(function() {
    /* ── CURSOR GLOW ── */
    const glow = document.getElementById('cursor-glow');
    // Move glow into the parent Streamlit document
    const stDoc = window.parent.document;
    const glowEl = stDoc.createElement('div');
    glowEl.id = 'cursor-glow-live';
    glowEl.style.cssText = `
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9998;
        transition: background 0.15s ease;
    `;
    stDoc.body.appendChild(glowEl);

    stDoc.addEventListener('mousemove', function(e) {
        const x = e.clientX, y = e.clientY;
        glowEl.style.background =
            `radial-gradient(600px circle at ${x}px ${y}px,
             rgba(0, 212, 255, 0.08),
             rgba(59, 130, 246, 0.04),
             transparent 60%)`;
    });

    /* ── FLOATING PARTICLES ── */
    const canvas = stDoc.createElement('canvas');
    canvas.id = 'particles-live';
    canvas.style.cssText = `
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9997;
        opacity: 0.35;
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
    const PARTICLE_COUNT = 45;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            r: Math.random() * 1.5 + 0.5,
            alpha: Math.random() * 0.5 + 0.1,
            color: ['0, 212, 255', '59, 130, 246', '168, 85, 247'][Math.floor(Math.random()*3)]
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
            ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
            ctx.fill();
        }

        // Draw faint connecting lines
        for (let i = 0; i < particles.length; i++) {
            for (let j = i+1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 212, 255, ${0.06 * (1 - dist/120)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animateParticles);
    }
    animateParticles();

    /* ── FLASH EFFECTS ── */
    // Create the overlay in parent doc
    const flashEl = stDoc.createElement('div');
    flashEl.id = 'flash-live';
    flashEl.style.cssText = `
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9999;
        opacity: 0;
    `;
    stDoc.body.appendChild(flashEl);

    // Expose flash functions globally
    window.parent.__flashCyan = function() {
        flashEl.style.opacity = '1';
        flashEl.style.background = 'radial-gradient(ellipse at center, rgba(0,212,255,0.18) 0%, transparent 65%)';
        flashEl.style.transition = 'opacity 1.2s ease-out';
        setTimeout(() => { flashEl.style.opacity = '0'; }, 50);
    };

    window.parent.__flashGreen = function() {
        flashEl.style.opacity = '1';
        flashEl.style.background = 'radial-gradient(ellipse at center, rgba(34,197,94,0.22) 0%, transparent 65%)';
        flashEl.style.transition = 'opacity 1.5s ease-out';
        setTimeout(() => { flashEl.style.opacity = '0'; }, 50);
    };

    /* ── OBSERVE CHAT FOR AUTO-FLASH ── */
    // Watch for new chat messages and fire the appropriate flash
    const observer = new MutationObserver(function(mutations) {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType !== 1) continue;
                // Check if a user message appeared (cyan flash)
                const userIcon = node.querySelector && node.querySelector('[data-testid="chatAvatarIcon-user"]');
                if (userIcon) { window.parent.__flashCyan(); }
                // Check if an assistant message appeared (green flash)
                const botIcon = node.querySelector && node.querySelector('[data-testid="chatAvatarIcon-assistant"]');
                if (botIcon) { window.parent.__flashGreen(); }
            }
        }
    });
    // Observe the main block container
    const mainBlock = stDoc.querySelector('[data-testid="stMainBlockContainer"]');
    if (mainBlock) {
        observer.observe(mainBlock, { childList: true, subtree: true });
    }

    /* ── CLEANUP previous duplicate elements on re-render ── */
    const oldGlow = stDoc.querySelectorAll('#cursor-glow-live');
    if (oldGlow.length > 1) { for (let i = 0; i < oldGlow.length - 1; i++) oldGlow[i].remove(); }
    const oldCanvas = stDoc.querySelectorAll('#particles-live');
    if (oldCanvas.length > 1) { for (let i = 0; i < oldCanvas.length - 1; i++) oldCanvas[i].remove(); }
    const oldFlash = stDoc.querySelectorAll('#flash-live');
    if (oldFlash.length > 1) { for (let i = 0; i < oldFlash.length - 1; i++) oldFlash[i].remove(); }
})();
</script>
""", height=0)

# ──────────────────────────────────────────────
# Sidebar: Indexing controls & config display
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.markdown("---")
    st.subheader("📂 Index a Codebase")
    codebase_path = st.text_input(
        "Path to codebase folder",
        value=DEFAULT_CODEBASE_PATH,
        help="Absolute or relative path to the folder of .py files to index.",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔨 Build Index", use_container_width=True):
            target = Path(codebase_path)
            if not target.is_dir():
                st.error(f"Not a valid directory: {codebase_path}")
            else:
                with st.spinner("Indexing codebase..."):
                    try:
                        build_index(str(target))
                        st.session_state.engine = None  # force reload
                        st.success(f"Index built from {target.name}/")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
    with col2:
        if st.button("🔄 Rebuild", use_container_width=True):
            # Clear the existing collection and re-index
            target = Path(codebase_path)
            if not target.is_dir():
                st.error(f"Not a valid directory: {codebase_path}")
            else:
                with st.spinner("Clearing old index and rebuilding..."):
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

    st.markdown("---")
    st.subheader("🤖 Active Models")
    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Top-K:** `{TOP_K}`")

    # ── Chat History Section ──
    st.markdown("---")
    st.subheader("💬 Chat History")

    # New Chat button
    if st.button("✨ New Chat", use_container_width=True):
        # Save current chat before starting new one
        if st.session_state.messages:
            _save_chat(st.session_state.chat_id, st.session_state.messages)
        st.session_state.messages = []
        st.session_state.chat_id = _generate_chat_id()
        st.rerun()

    # List saved chats
    saved_chats = _list_chats()
    if saved_chats:
        for chat in saved_chats[:15]:  # show last 15 chats
            col_load, col_del = st.columns([5, 1])
            with col_load:
                # Truncate title for display
                display_title = chat["title"]
                label = f"{display_title}  ({chat['message_count']} msgs)"
                if st.button(label, key=f"load_{chat['id']}", use_container_width=True):
                    # Save current chat first
                    if st.session_state.messages:
                        _save_chat(st.session_state.chat_id, st.session_state.messages)
                    # Load selected chat
                    loaded = _load_chat(chat["id"])
                    if loaded:
                        st.session_state.messages = loaded["messages"]
                        st.session_state.chat_id = chat["id"]
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{chat['id']}"):
                    _delete_chat(chat["id"])
                    # If we deleted the active chat, start fresh
                    if st.session_state.get("chat_id") == chat["id"]:
                        st.session_state.messages = []
                        st.session_state.chat_id = _generate_chat_id()
                    st.rerun()
    else:
        st.caption("No saved chats yet.")

    st.markdown("---")
    st.caption("🔒 Fully offline -- no data leaves this machine.")

# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = _generate_chat_id()

if "engine" not in st.session_state:
    st.session_state.engine = None

# ──────────────────────────────────────────────
# Try to load the query engine if not already loaded
# ──────────────────────────────────────────────
if st.session_state.engine is None:
    try:
        st.session_state.engine = load_query_engine()
    except Exception:
        pass  # no index yet -- handled below

# ──────────────────────────────────────────────
# Main Panel
# ──────────────────────────────────────────────
st.title("Codebase Assistant")
st.caption("Ask anything -- codebase questions are answered from your code, general questions get direct answers.")

# Status indicator
if st.session_state.engine:
    st.markdown(
        '<span class="status-badge status-ready">&#x25CF; Index loaded -- ready to answer</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-badge status-none">&#x25CB; No index loaded -- build one from the sidebar</span>',
        unsafe_allow_html=True,
    )
    st.info("Use the sidebar to point at a codebase folder and click **Build Index** to get started.")

# ──────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show mode badge for assistant messages
        if msg["role"] == "assistant" and msg.get("mode"):
            if msg["mode"] == "code":
                st.markdown(
                    '<span class="mode-badge mode-code">&#x1F4C2; Code RAG</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="mode-badge mode-general">&#x1F30D; General Knowledge</span>',
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
# Chat input
# ──────────────────────────────────────────────
question = st.chat_input("Ask about the codebase or anything else...")

if question:
    if not st.session_state.engine:
        st.warning("Please build an index first using the sidebar.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer (hybrid: code RAG + general fallback)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources, mode = ask(st.session_state.engine, question)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
                    sources = []
                    mode = "error"

            # Show mode badge
            if mode == "code":
                st.markdown(
                    '<span class="mode-badge mode-code">&#x1F4C2; Code RAG</span>',
                    unsafe_allow_html=True,
                )
            elif mode == "general":
                st.markdown(
                    '<span class="mode-badge mode-general">&#x1F30D; General Knowledge</span>',
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

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "mode": mode,
        })

        # Auto-save chat after each exchange
        _save_chat(st.session_state.chat_id, st.session_state.messages)
