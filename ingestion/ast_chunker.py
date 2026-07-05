"""
AST Chunker — Phase 1
Parses Python source files into function/class-level chunks using the `ast` module.
This is the conceptually most important file: function/class-level chunking preserves
semantic boundaries that fixed-size text splitting would destroy.
"""

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_chunks(filepath):
    """Parse one .py file into function/class-level chunks.

    For each FunctionDef, AsyncFunctionDef, and ClassDef node found in the AST,
    extracts the full source text, docstring, file path, and starting line number.

    Args:
        filepath: Path to the .py file to parse.

    Returns:
        A list of dicts, each with keys: text, name, type, file, line, docstring.
        Returns an empty list if the file cannot be parsed (e.g. SyntaxError).
    """
    filepath = Path(filepath)
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("[skip] %s: %s", filepath, e)
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        logger.warning("[skip] %s: %s", filepath, e)
        return []

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chunk_text = ast.get_source_segment(source, node)
            if not chunk_text:
                continue
            chunks.append({
                "text": chunk_text,
                "name": node.name,
                "type": type(node).__name__,
                "file": str(filepath),
                "line": node.lineno,
                "docstring": ast.get_docstring(node) or "",
            })

    return chunks


def index_codebase(root_dir):
    """Walk the full codebase and collect all chunks.

    Recursively scans all .py files under `root_dir`, parses each with
    `extract_chunks`, and returns the combined list.

    Args:
        root_dir: Path to the root directory of the codebase.

    Returns:
        A list of chunk dicts from all .py files found.
    """
    root = Path(root_dir)
    if not root.is_dir():
        logger.error("Not a directory: %s", root)
        return []

    all_chunks = []
    py_files = list(root.rglob("*.py"))
    logger.info("Found %d .py files in %s", len(py_files), root)

    for py_file in py_files:
        file_chunks = extract_chunks(py_file)
        all_chunks.extend(file_chunks)
        if file_chunks:
            logger.debug("  %s → %d chunks", py_file.name, len(file_chunks))

    logger.info("Total chunks extracted: %d", len(all_chunks))
    return all_chunks


# ──────────────────────────────────────────────
# Quick CLI test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    chunks = index_codebase(target)

    print(f"\n{'='*60}")
    print(f"Extracted {len(chunks)} chunks from {target}")
    print(f"{'='*60}")

    # Show first 3 chunks as preview
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(json.dumps({k: v for k, v in c.items() if k != "text"}, indent=2))
        preview = c["text"][:200] + ("..." if len(c["text"]) > 200 else "")
        print(f"Code preview:\n{preview}")
