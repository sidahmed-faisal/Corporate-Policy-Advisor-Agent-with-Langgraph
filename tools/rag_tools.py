"""
tools/rag_tools.py
────────────────────────────────────────────────────────────────────────────────
Three core tools the LangGraph agents can call:
  1. retrieve(query, top_k, filter_superseded)  — hybrid dense + BM25 search
  2. get_document_metadata(doc_id)              — metadata from metadata.json
  3. check_contradictions(doc_ids, question)    — LLM-based conflict detector

All tools return plain Python dicts (JSON-serialisable) so they can be
attached to the LangGraph state without trouble.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ingestion.ingest import embed, get_qdrant, get_bm25_index

logger = logging.getLogger(__name__)

# ─── metadata cache ───────────────────────────────────────────────────────────
_metadata: dict[str, dict] | None = None


def _load_metadata() -> dict[str, dict]:
    global _metadata
    if _metadata is None:
        with open(config.METADATA_FILE) as f:
            raw = json.load(f)
        _metadata = raw.get("documents", {})
    return _metadata


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: retrieve
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = 8,
    filter_superseded: bool = True,
    doc_id_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: dense (Qdrant) + BM25, RRF fusion, optional supersession filter.

    Returns list of chunk dicts with keys:
      chunk_id, doc_id, text, score, metadata
    """
    if get_bm25_index().size() == 0:
        logger.warning("BM25 index empty — retrieval may be degraded.")

    dense_w = config.DENSE_WEIGHT
    bm25_w = config.BM25_WEIGHT

    # ── Dense retrieval ───────────────────────────────────────────────────────
    q_vec = embed([query])[0]
    client = get_qdrant()

    qdrant_filter = None
    if filter_superseded:
        # Only return chunks where superseded_by is null
        # Qdrant has no native "is null" filter on string; we rely on payload
        pass  # handled post-retrieval below

    raw_dense = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=q_vec,
        limit=top_k * 3,
        with_payload=True,
    ).points

    dense_results: dict[str, tuple[dict, float]] = {}
    for hit in raw_dense:
        p = hit.payload or {}
        if filter_superseded and p.get("superseded_by"):
            continue
        if doc_id_filter and p.get("doc_id") != doc_id_filter:
            continue
        cid = p.get("chunk_id", str(hit.id))
        dense_results[cid] = (p, hit.score)

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    bm25_raw = get_bm25_index().query(query, top_k=top_k * 3)
    bm25_results: dict[str, tuple[dict, float]] = {}
    for chunk_dict, score in bm25_raw:
        p = chunk_dict["metadata"]
        if filter_superseded and p.get("superseded_by"):
            continue
        if doc_id_filter and chunk_dict.get("doc_id") != doc_id_filter:
            continue
        bm25_results[chunk_dict["chunk_id"]] = (chunk_dict, score)

    # ── Reciprocal Rank Fusion (RRF) ──────────────────────────────────────────
    k_rrf = 60
    rrf_scores: dict[str, float] = {}

    dense_ranked = sorted(dense_results.items(), key=lambda x: x[1][1], reverse=True)
    for rank, (cid, _) in enumerate(dense_ranked):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_w / (k_rrf + rank + 1)

    bm25_ranked = sorted(bm25_results.items(), key=lambda x: x[1][1], reverse=True)
    for rank, (cid, _) in enumerate(bm25_ranked):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + bm25_w / (k_rrf + rank + 1)

    # ── Merge payloads ────────────────────────────────────────────────────────
    all_payloads: dict[str, dict] = {}
    for cid, (payload, _) in dense_results.items():
        all_payloads[cid] = payload
    for cid, (chunk_dict, _) in bm25_results.items():
        if cid not in all_payloads:
            # BM25 chunk dict is structured differently
            all_payloads[cid] = {
                "chunk_id": chunk_dict["chunk_id"],
                "doc_id": chunk_dict["doc_id"],
                "text": chunk_dict["text"],
                **chunk_dict["metadata"],
            }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for cid, score in ranked:
        if cid not in all_payloads:
            continue
        p = all_payloads[cid]
        results.append(
            {
                "chunk_id": cid,
                "doc_id": p.get("doc_id", ""),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "score": round(score, 5),
                "effective_date": p.get("effective_date", ""),
                "superseded_by": p.get("superseded_by"),
                "category": p.get("category", ""),
                "department": p.get("department", ""),
            }
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: get_document_metadata
# ─────────────────────────────────────────────────────────────────────────────
def get_document_metadata(doc_id: str) -> dict[str, Any]:
    """Return metadata for a single document from metadata.json."""
    meta = _load_metadata()
    if doc_id not in meta:
        return {"error": f"Document '{doc_id}' not found in metadata."}
    doc = dict(meta[doc_id])
    doc["doc_id"] = doc_id
    # Resolve supersession chain
    if doc.get("superseded_by"):
        newer = meta.get(doc["superseded_by"], {})
        doc["superseded_by_title"] = newer.get("title", "")
    if doc.get("supersedes"):
        older = meta.get(doc["supersedes"], {})
        doc["supersedes_title"] = older.get("title", "")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: check_contradictions
# ─────────────────────────────────────────────────────────────────────────────
def check_contradictions(
    chunks: list[dict[str, Any]],
    question: str,
    llm_invoke,  # callable: str -> str
) -> dict[str, Any]:
    """
    Given retrieved chunks and the original question, ask the LLM whether
    any chunks contradict each other on the topic of the question.

    Returns:
      {
        "has_contradiction": bool,
        "conflict_pairs": [ {"doc_a": ..., "doc_b": ..., "claim_a": ..., "claim_b": ...} ],
        "reasoning": str
      }
    """
    if len(chunks) < 2:
        return {"has_contradiction": False, "conflict_pairs": [], "reasoning": "Only one chunk — no comparison possible."}

    # Build a condensed view of each doc's stance
    doc_stances: dict[str, list[str]] = {}
    for chunk in chunks:
        did = chunk["doc_id"]
        doc_stances.setdefault(did, []).append(chunk["text"][:600])

    doc_summaries = []
    for did, texts in doc_stances.items():
        combined = " … ".join(texts[:2])
        doc_summaries.append(f"[{did}]: {combined}")

    prompt = f"""You are a policy analyst. The user asked: "{question}"

Below are excerpts from {len(doc_stances)} policy documents retrieved as relevant.
Determine whether any two documents CONTRADICT each other on the topic of the question.
A contradiction means they give different, incompatible answers to the question.

Documents:
{chr(10).join(doc_summaries)}

Respond ONLY with valid JSON in this exact schema:
{{
  "has_contradiction": true | false,
  "conflict_pairs": [
    {{
      "doc_a": "DOC-ID",
      "doc_b": "DOC-ID",
      "claim_a": "what doc_a says",
      "claim_b": "what doc_b says"
    }}
  ],
  "reasoning": "one sentence explanation"
}}
If no contradiction, return an empty conflict_pairs array.
"""
    raw = llm_invoke(prompt)
    # strip markdown fences if present
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "has_contradiction": False,
            "conflict_pairs": [],
            "reasoning": raw[:400],
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reranker (cross-encoder style via LLM scoring)
# ─────────────────────────────────────────────────────────────────────────────
def rerank(
    chunks: list[dict[str, Any]],
    question: str,
    llm_invoke,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    LLM-based pointwise relevance reranking.  Scores each chunk 0-10 then sorts.
    Falls back to original order on parse failure.
    """
    if not chunks:
        return chunks
    if len(chunks) <= top_k:
        return chunks

    scored: list[tuple[int, dict]] = []
    for chunk in chunks:
        prompt = (
            f"Question: {question}\n\n"
            f"Passage: {chunk['text'][:800]}\n\n"
            "Rate how relevant this passage is to answering the question. "
            "Reply with a single integer from 0 (irrelevant) to 10 (highly relevant). "
            "No other text."
        )
        try:
            resp = llm_invoke(prompt).strip()
            score = int("".join(filter(str.isdigit, resp.split()[0])))
        except Exception:
            score = 5
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]