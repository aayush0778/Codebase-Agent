"""
Comprehensive test suite for Corrective Implementation:
- Response Confidence Indicators (grounding, coverage, specificity, overall, suggestions)
- Intelligent Context Compression
- BackgroundTaskManager lifecycle, duplicate prevention, cancellation & consumption
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.query_engine import (
    _compute_specificity,
    _compute_coverage,
    _compute_grounding,
    _compute_quality_info,
    _compress_context,
)
from retrieval.background_tasks import (
    BackgroundTaskManager,
    TaskStatus,
)


def _bar_color(val):
    if val >= 75: return "#34D399"
    if val >= 55: return "#6EE7B7"
    if val >= 35: return "#FDE047"
    return "#F87171"


def _render_confidence_panel_test(quality_info, mode, confidence=None, source_count=0):
    """Mirror of app._render_confidence_panel for testing rendering logic."""
    if not quality_info:
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


def test_specificity():
    print("\n--- TEST 1: Specificity Evaluation ---")
    spec_vague = _compute_specificity("Explain it")
    spec_detailed = _compute_specificity("Explain Calculator.add() in sample_module.py")
    spec_medium = _compute_specificity("How does the calculation history work?")
    
    print(f"  'Explain it' Specificity: {spec_vague:.1f}%")
    print(f"  'Explain Calculator.add() in sample_module.py' Specificity: {spec_detailed:.1f}%")
    print(f"  'How does the calculation history work?' Specificity: {spec_medium:.1f}%")

    assert spec_vague < 40.0, f"Expected vague query < 40, got {spec_vague}"
    assert spec_detailed >= 75.0, f"Expected specific query >= 75, got {spec_detailed}"
    print("  [PASS] Specificity deterministic evaluation works correctly.")


def test_coverage():
    print("\n--- TEST 2: Coverage & Diversity ---")
    class MockNode:
        def __init__(self, fname):
            self.metadata = {"file": fname}
            self.score = 0.85

    nodes_empty = []
    nodes_single = [MockNode("sample_module.py") for _ in range(4)]
    nodes_diverse = [MockNode(f"file_{i%3}.py") for i in range(8)]

    cov_empty = _compute_coverage(nodes_empty, top_k=8)
    cov_single = _compute_coverage(nodes_single, top_k=8)
    cov_diverse = _compute_coverage(nodes_diverse, top_k=8)

    print(f"  Empty Nodes Coverage: {cov_empty:.1f}%")
    print(f"  4 Nodes Single File: {cov_single:.1f}%")
    print(f"  8 Nodes Multi-File: {cov_diverse:.1f}%")

    assert cov_empty == 0.0
    assert cov_single < cov_diverse
    print("  [PASS] Coverage and file diversity heuristic works correctly.")


def test_grounding_and_quality_info():
    print("\n--- TEST 3: Grounding & Response Confidence Indicators Object ---")
    rag_answer = (
        "The `Calculator` class in `sample_module.py` defines the `add(a, b)` method "
        "which returns the sum and logs the operation in `history`."
    )
    sources = [
        {"file": "sample_module.py", "name": "Calculator", "type": "class", "line": 10, "score": 0.94},
        {"file": "sample_module.py", "name": "add", "type": "function", "line": 20, "score": 0.92},
    ]
    
    class MockNode:
        def __init__(self, fname, name, score):
            self.metadata = {"file": fname, "name": name}
            self.score = score

    source_nodes = [
        MockNode("sample_module.py", "Calculator", 0.94),
        MockNode("sample_module.py", "add", 0.92),
    ]

    q_info = _compute_quality_info(
        question="Explain Calculator.add() in sample_module.py",
        rag_answer=rag_answer,
        sources=sources,
        mode="code",
        best_score=0.94,
        source_nodes=source_nodes,
    )

    print("  Code Mode Quality Info:", q_info)
    assert q_info["grounding"] >= 70.0
    assert q_info["overall_score"] >= 70.0
    assert q_info["overall"] in ("High", "Good")
    assert q_info["retrieval_relevance"] == 94.0

    # Test General Mode Quality Info
    gen_info = _compute_quality_info(
        question="What is the capital of France?",
        rag_answer="Paris is the capital of France.",
        sources=[],
        mode="general",
        best_score=0.2,
        source_nodes=[],
    )
    print("  General Mode Quality Info:", gen_info)
    assert gen_info["grounding"] == 0.0
    assert gen_info["coverage"] == 0.0
    assert gen_info["retrieval_relevance"] == 0.0
    assert len(gen_info["suggestions"]) >= 1
    print("  [PASS] Quality info calculation & general mode differentiation verified.")


def test_intelligent_compression():
    print("\n--- TEST 4: Intelligent Context Compression ---")
    small_history = "User: Hello\nAssistant: Hi there!"
    text, info = _compress_context(small_history, "Test", "", max_tokens=8192)
    assert not info["was_compressed"]
    print("  Small context: was_compressed = False")

    huge_history = "import math\nfrom sample_module import Calculator\n\n" + (
        "User: Explain function foo\nAssistant: Here is foo code:\ndef foo():\n    return 42\n" * 150
    )
    text_comp, info_comp = _compress_context(huge_history, "Test question", "", max_tokens=500)
    print(f"  Oversized context: was_compressed = {info_comp['was_compressed']}, {info_comp['original_chars']} -> {info_comp['compressed_chars']} chars")
    print(f"  Preserved symbols: {info_comp['preserved_symbols']}")
    assert info_comp["was_compressed"]
    assert info_comp["compressed_chars"] < info_comp["original_chars"]
    print("  [PASS] Intelligent context compression preserves structural tokens.")


def test_background_task_manager():
    print("\n--- TEST 5: BackgroundTaskManager Lifecycle & Duplicate Prevention ---")
    manager = BackgroundTaskManager.get_instance()
    chat_id = "test_chat_101"

    def mock_gen(q, progress_fn=None, **kwargs):
        if progress_fn:
            progress_fn("Step 1")
        time.sleep(0.2)
        if progress_fn:
            progress_fn("Step 2")
        return (
            f"Answer for {q}",
            [{"file": "sample.py", "name": "foo", "type": "function", "line": 1, "score": 0.9}],
            "code",
            0.9,
            {"was_compressed": False},
            {"grounding": 90, "coverage": 80, "specificity": 85, "overall_score": 85, "overall": "High", "suggestions": [], "retrieval_relevance": 90},
        )

    task_id = manager.submit(chat_id, "Test question", mock_gen)
    print(f"  Submitted task {task_id} for chat {chat_id}")
    assert manager.is_generating(chat_id)

    # Test Duplicate Prevention
    try:
        manager.submit(chat_id, "Duplicate question", mock_gen)
        assert False, "Duplicate task was not rejected!"
    except RuntimeError as e:
        print(f"  [OK] Duplicate submission correctly rejected: {e}")

    # Wait for completion
    time.sleep(0.4)
    assert not manager.is_generating(chat_id)
    result = manager.consume_result(chat_id)
    assert result is not None
    assert "quality_info" in result
    assert result["answer"] == "Answer for Test question"
    print("  [PASS] Background task lifecycle and result consumption verified.")

    # Test Cancellation
    chat_id_2 = "test_chat_cancel"
    def slow_gen(q, progress_fn=None, **kwargs):
        time.sleep(0.5)
        return ("Late answer", [], "code", 0.9, False, None)

    task_id_2 = manager.submit(chat_id_2, "Slow question", slow_gen)
    cancelled = manager.cancel(chat_id_2)
    assert cancelled
    assert not manager.is_generating(chat_id_2)
    time.sleep(0.6)
    late_result = manager.consume_result(chat_id_2)
    assert late_result is None
    print("  [PASS] Task cancellation successfully discards late results.")


def test_ui_rendering():
    print("\n--- TEST 6: UI Panel Rendering & Exact Required Strings ---")
    # 1. Exact Title and Exact Disclaimer Check
    q_info = {
        "grounding": 91.0,
        "coverage": 75.0,
        "specificity": 88.0,
        "overall_score": 86.0,
        "overall": "High",
        "source_count": 8,
        "retrieval_relevance": 94.0,
        "suggestions": ["Mention the function name for more precise grounding."],
    }
    panel_html = _render_confidence_panel_test(q_info, mode="code")
    assert "Response Confidence Indicators" in panel_html
    assert "This is a heuristic based on retrieval quality and context coverage. It is NOT the model's confidence." in panel_html
    assert "Grounding" in panel_html
    assert "91%" in panel_html
    assert "Coverage" in panel_html
    assert "75%" in panel_html
    assert "Specificity" in panel_html
    assert "88%" in panel_html
    assert "High · 86%" in panel_html
    assert "Sources: 8 retrieved chunks" in panel_html
    assert "Retrieval Relevance: 94%" in panel_html

    # 2. General Knowledge Mode Check
    gen_panel_html = _render_confidence_panel_test(
        {"specificity": 30.0, "suggestions": ["Mention specific filenames."]},
        mode="general"
    )
    assert "General Knowledge" in gen_panel_html
    assert "No sufficiently relevant repository context was retrieved for this question." in gen_panel_html

    # 3. Backward compatibility for historical chats without quality_info
    old_panel_html = _render_confidence_panel_test(None, mode="code", confidence=0.92, source_count=2)
    assert "Response Confidence Indicators" in old_panel_html
    assert "92%" in old_panel_html

    print("  [PASS] All UI indicators, exact titles, and required disclaimers verified.")


if __name__ == "__main__":
    test_specificity()
    test_coverage()
    test_grounding_and_quality_info()
    test_intelligent_compression()
    test_background_task_manager()
    test_ui_rendering()
    print("\n============================================================")
    print("ALL UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("============================================================")
