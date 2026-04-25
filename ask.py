#!/usr/bin/env python3
"""
ask.py — CLI entry point.

Usage:
    python ask.py "What is the standard notice period at Meridian?"
    python ask.py "ما هي إجازة الحداد؟"
"""
from __future__ import annotations

import json
import sys
import textwrap
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # quiet for CLI

sys.path.insert(0, str(Path(__file__).parent))

import config
from ingestion.ingest import ingest_corpus, get_bm25_index
from agents.workflow import run_agent


def ensure_ingested():
    """Ingest corpus if BM25 index is empty (first run or memory reset)."""
    if get_bm25_index().size() == 0:
        if not config.CORPUS_DIR.exists():
            print(f"[ERROR] Corpus directory '{config.CORPUS_DIR}' not found.")
            print("Place policy documents in ./policy_corpus/ or set CORPUS_DIR env var.")
            sys.exit(1)
        print(f"[INFO] Ingesting corpus from '{config.CORPUS_DIR}' …")
        n = ingest_corpus(config.CORPUS_DIR, config.METADATA_FILE)
        print(f"[INFO] Indexed {n} chunks.\n")


def format_answer(result: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"QUESTION: {result['question']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("ANSWER:")
    lines.append(textwrap.fill(result["answer"], width=70))
    lines.append("")

    # Contradictions
    contra = result.get("contradictions", {})
    if contra.get("has_contradiction"):
        lines.append("⚠️  CONTRADICTIONS DETECTED:")
        for pair in contra.get("conflict_pairs", []):
            lines.append(f"  • [{pair['doc_a']}] says: {pair['claim_a']}")
            lines.append(f"    [{pair['doc_b']}] says: {pair['claim_b']}")
        lines.append("")

    # Citations
    citations = result.get("citations", [])
    if citations:
        lines.append("CITATIONS:")
        seen = set()
        for c in citations:
            key = c.get("doc_id", "")
            if key and key not in seen:
                seen.add(key)
                title = c.get("title", "")
                lines.append(f"  • [{key}] {title}")
        lines.append("")

    # Plan + confidence
    plan = result.get("plan", {})
    lines.append(f"CONFIDENCE: {result.get('confidence', 'medium').upper()}")
    lines.append(f"STRATEGY:   {plan.get('strategy', 'unknown')}")
    lines.append(f"REASONING:  {plan.get('reasoning', '')}")
    lines.append("")

    # Trace summary
    trace = result.get("trace", [])
    lines.append("AGENT TRACE:")
    for step in trace:
        lines.append(f"  [{step.get('agent', '?')}] {step.get('step', '?')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    ensure_ingested()

    print(f"\n[INFO] Running agentic RAG for: {question!r}\n")
    result = run_agent(question)
    print(format_answer(result))

    # Also dump raw JSON for programmatic use
    out_path = Path("last_answer.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Full JSON saved to {out_path}")


if __name__ == "__main__":
    main()
