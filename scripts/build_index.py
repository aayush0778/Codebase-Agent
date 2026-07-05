"""
CLI script to (re)index a codebase folder.

Usage:
    python scripts/build_index.py --path /path/to/codebase
    python scripts/build_index.py --path ./sample_codebase --persist ./data/chroma_index
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.indexer import build_index
from config import CHROMA_PERSIST_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Index a Python codebase for the local RAG agent."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the codebase folder to index",
    )
    parser.add_argument(
        "--persist",
        default=CHROMA_PERSIST_DIR,
        help=f"Directory to persist the Chroma index (default: {CHROMA_PERSIST_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    codebase_path = Path(args.path)
    if not codebase_path.is_dir():
        print(f"Error: '{args.path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Indexing codebase at: {codebase_path}")
    print(f"Persisting index to: {args.persist}")
    print("=" * 60)

    index = build_index(str(codebase_path), persist_dir=args.persist)

    if index is None:
        print("No index was built - check logs for errors.", file=sys.stderr)
        sys.exit(1)

    print("\n[OK] Indexing complete. You can now run the Streamlit app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
