"""Quick smoke test for all imports."""
import sys
sys.path.insert(0, ".")

try:
    from ingestion.ast_chunker import index_codebase
    print("[OK] ingestion.ast_chunker")
except Exception as e:
    print(f"[FAIL] ingestion.ast_chunker: {e}")

try:
    from ingestion.indexer import build_index
    print("[OK] ingestion.indexer")
except Exception as e:
    print(f"[FAIL] ingestion.indexer: {e}")

try:
    from retrieval.query_engine import load_query_engine, ask
    print("[OK] retrieval.query_engine")
except Exception as e:
    print(f"[FAIL] retrieval.query_engine: {e}")

try:
    import config
    print(f"[OK] config (LLM={config.LLM_MODEL}, Embed={config.EMBEDDING_MODEL})")
except Exception as e:
    print(f"[FAIL] config: {e}")

print("\nAll imports tested.")
