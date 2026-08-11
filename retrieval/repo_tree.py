"""
Repository Tree Builder — Architecture Tab Backend

Builds a hierarchical tree of the indexed repository's file structure and
extracts per-file code structure (classes, functions, imports) using AST.

Results are designed to be cached in st.session_state to avoid re-computation
on every Streamlit rerun.
"""

import ast
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def build_repo_tree(root_path):
    """Build a file/directory tree structure for the given root path.

    Args:
        root_path: Absolute path to the repository root.

    Returns:
        A dict with keys:
          - tree: nested dict of {name: {children: {}, is_dir: bool, path: str}}
          - stats: {total_files, total_dirs, py_files, total_loc}
          - flat_files: list of absolute paths to all .py files
    """
    root = Path(root_path)
    if not root.is_dir():
        return {"tree": {}, "stats": {}, "flat_files": []}

    tree = {}
    stats = {"total_files": 0, "total_dirs": 0, "py_files": 0, "total_loc": 0}
    flat_files = []

    # Directories to skip
    skip_dirs = {
        "__pycache__", ".git", ".svn", "node_modules", ".tox",
        ".eggs", "*.egg-info", ".venv", "venv", "env",
        ".mypy_cache", ".pytest_cache", "dist", "build",
    }

    for dirpath, dirnames, filenames in os.walk(str(root)):
        # Filter out ignored directories in-place
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.endswith(".egg-info")]

        rel_dir = os.path.relpath(dirpath, str(root))
        stats["total_dirs"] += 1

        for fname in sorted(filenames):
            stats["total_files"] += 1
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.join(rel_dir, fname) if rel_dir != "." else fname

            if fname.endswith(".py"):
                stats["py_files"] += 1
                flat_files.append(full_path)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        stats["total_loc"] += sum(1 for _ in f)
                except OSError:
                    pass

            # Insert into tree
            parts = Path(rel_path).parts
            node = tree
            for i, part in enumerate(parts):
                is_last = (i == len(parts) - 1)
                if part not in node:
                    node[part] = {
                        "children": {},
                        "is_dir": not is_last,
                        "path": os.path.join(str(root), *parts[:i+1]),
                    }
                node = node[part]["children"]

    return {"tree": tree, "stats": stats, "flat_files": flat_files}


def extract_file_structure(filepath):
    """Extract code structure from a single Python file using AST.

    Returns:
        A dict with keys:
          - classes: list of {name, line, methods: list of str, docstring}
          - functions: list of {name, line, docstring}
          - imports: list of str
          - loc: int (lines of code)
          - error: str or None
    """
    result = {
        "classes": [],
        "functions": [],
        "imports": [],
        "loc": 0,
        "error": None,
    }

    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        result["error"] = str(e)
        return result

    result["loc"] = source.count("\n") + 1

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
        return result

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            docstring = ast.get_docstring(node) or ""
            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "docstring": docstring[:120] if docstring else "",
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node) or ""
            result["functions"].append({
                "name": node.name,
                "line": node.lineno,
                "docstring": docstring[:120] if docstring else "",
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                result["imports"].append(f"{module}.{alias.name}")

    return result


def get_cached_repo_tree(session_state, root_path):
    """Return the repo tree, building it if not cached.

    Uses session_state to cache. Only rebuilds if root_path changes.
    """
    cached = session_state.get("_repo_tree_cache")
    if cached and cached.get("root") == str(root_path):
        return cached["data"]

    data = build_repo_tree(root_path)
    session_state["_repo_tree_cache"] = {"root": str(root_path), "data": data}
    return data
