#!/usr/bin/env python3
"""
evaluate.py — Evaluation harness for the Agentic RAG Policy Advisor.

Usage:
    python evaluate.py [--questions eval_questions.json] [--output results.json]

Reads each question from the eval set, runs it through the agent, and writes
a results.json containing:
  - agent answer
  - citations
  - confidence
  - contradictions detected
  - trace
  - auto-scored flags (contradiction_surfaced, has_citations, refused_correctly)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

import config
from ingestion.ingest import ingest_corpus, get_bm25_index
from agents.workflow import run_agent


def ensure_ingested():
    if get_bm25_index().size() == 0:
        if not config.CORPUS_DIR.exists():
            print(f"[ERROR] Corpus dir '{config.CORPUS_DIR}' not found.", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Ingesting corpus …", file=sys.stderr)
        n = ingest_corpus(config.CORPUS_DIR, config.METADATA_FILE)
        print(f"[INFO] Indexed {n} chunks.", file=sys.stderr)


def auto_score(question_meta: dict, result: dict) -> dict[str, bool | None]:
    """
    Apply lightweight auto-scoring heuristics.
    Returns flags that the reviewer uses as a starting point.
    """
    q_type = question_meta.get("type", "")
    answer = result.get("answer", "").lower()
    confidence = result.get("confidence", "medium")
    contradictions = result.get("contradictions", {})

    flags: dict[str, bool | None] = {
        "has_answer": bool(answer.strip()),
        "has_citations": len(result.get("citations", [])) > 0,
        "refused": confidence == "refused",
    }

    if q_type == "contradiction":
        flags["contradiction_surfaced"] = (
            contradictions.get("has_contradiction", False)
            or "conflict" in answer
            or "contradict" in answer
            or "disagree" in answer
            or "⚠" in result.get("answer", "")
            or "note:" in answer
        )

    if q_type == "out_of_scope":
        flags["correctly_refused"] = confidence == "refused" or (
            "cannot find" in answer
            or "not in the corpus" in answer
            or "outside the scope" in answer
        )

    if q_type == "supersession":
        flags["mentions_supersession"] = (
            "supersed" in answer
            or "updated" in answer
            or "replaced" in answer
            or "current version" in answer
            or "v2" in answer.lower()
        )

    return flags


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG Policy Advisor.")
    parser.add_argument(
        "--questions",
        default=str(config.CORPUS_DIR / "eval_questions.json"),
        help="Path to eval_questions.json",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between questions (respect rate limits)",
    )
    args = parser.parse_args()

    q_path = Path(args.questions)
    if not q_path.exists():
        print(f"[ERROR] Questions file not found: {q_path}", file=sys.stderr)
        sys.exit(1)

    with open(q_path) as f:
        eval_data = json.load(f)

    questions = eval_data.get("questions", [])
    print(f"[INFO] Running {len(questions)} questions …", file=sys.stderr)

    ensure_ingested()

    results = []
    for i, q_meta in enumerate(questions, 1):
        qid = q_meta.get("id", f"Q{i:02d}")
        question = q_meta.get("question", "")
        print(f"  [{i:02d}/{len(questions)}] {qid}: {question[:60]}…", file=sys.stderr)

        t0 = time.time()
        try:
            result = run_agent(question)
            elapsed = round(time.time() - t0, 2)
        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            result = {
                "question": question,
                "answer": f"[AGENT ERROR] {exc}",
                "citations": [],
                "confidence": "low",
                "contradictions": {},
                "plan": {},
                "trace": [{"agent": "error", "step": "exception", "output": str(exc)}],
            }

        flags = auto_score(q_meta, result)

        results.append(
            {
                "id": qid,
                "type": q_meta.get("type", ""),
                "expected_language": q_meta.get("expected_language", "en"),
                "question": question,
                "answer": result["answer"],
                "citations": result.get("citations", []),
                "confidence": result.get("confidence", "medium"),
                "contradictions": result.get("contradictions", {}),
                "plan": result.get("plan", {}),
                "trace": result.get("trace", []),
                "elapsed_s": elapsed,
                "auto_flags": flags,
            }
        )

        if args.delay > 0 and i < len(questions):
            time.sleep(args.delay)

    # ── Write results ─────────────────────────────────────────────────────────
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_version": eval_data.get("version", "1.0"),
                "total_questions": len(results),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Evaluation complete — {len(results)} questions processed.", file=sys.stderr)
    print(f"Results saved to: {out_path}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # Quick stats
    total = len(results)
    has_citations = sum(1 for r in results if r["auto_flags"].get("has_citations"))
    refused = sum(1 for r in results if r["auto_flags"].get("refused"))
    contra_surfaced = sum(
        1 for r in results if r["auto_flags"].get("contradiction_surfaced", None) is True
    )
    print(f"  With citations:          {has_citations}/{total}", file=sys.stderr)
    print(f"  Refused (out-of-scope):  {refused}", file=sys.stderr)
    print(f"  Contradictions surfaced: {contra_surfaced}", file=sys.stderr)

    # Print to stdout for piping
    print(out_path)


if __name__ == "__main__":
    main()
