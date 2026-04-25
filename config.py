"""
config.py — Central configuration loaded from environment variables / .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=False)

# ─── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ─── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "")         # blank → in-memory
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "policy_docs")

# ─── LangSmith ────────────────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "rag-policy-advisor")
LANGCHAIN_ENDPOINT: str = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)

# ─── Corpus ───────────────────────────────────────────────────────────────────
CORPUS_DIR: Path = Path(os.getenv("CORPUS_DIR", "./policy_corpus"))
METADATA_FILE: Path = Path(os.getenv("METADATA_FILE", "./policy_corpus/metadata.json"))

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "8"))
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))   # 0 = pure dense
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.7"))

# ─── App ──────────────────────────────────────────────────────────────────────
PORT: int = int(os.getenv("PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ─── Apply LangSmith env vars so LangChain picks them up ──────────────────────
if LANGCHAIN_TRACING_V2.lower() == "true" and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
