"""
api/app.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application exposing the LangGraph agentic RAG workflow.

Endpoints:
  POST /ask            — ask a policy question, returns full structured response
  POST /ingest         — (re-)ingest the corpus (idempotent)
  GET  /health         — liveness check
  GET  /docs           — Swagger UI (FastAPI default)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from agents.workflow import run_agent
from ingestion.ingest import ingest_corpus

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meridian Policy Advisor — Agentic RAG",
    description=(
        "Multi-agent RAG assistant (Planner → Retriever → Fact-Checker) "
        "for answering employee questions about corporate policy documents. "
        "Backed by Qdrant vector store + BM25 hybrid retrieval. "
        "Traced with LangSmith."
    ),
    version="1.0.0",
)

# ─── Startup: auto-ingest ─────────────────────────────────────────────────────
_ingested = False


@app.on_event("startup")
async def startup_event():
    global _ingested
    if config.CORPUS_DIR.exists() and config.METADATA_FILE.exists():
        logger.info("Auto-ingesting corpus on startup …")
        try:
            n = ingest_corpus(config.CORPUS_DIR, config.METADATA_FILE)
            logger.info("Ingested %d chunks.", n)
            _ingested = True
        except Exception as exc:
            logger.warning("Auto-ingest failed: %s. Call POST /ingest manually.", exc)
    else:
        logger.warning(
            "Corpus dir '%s' not found — skipping auto-ingest. "
            "Call POST /ingest after placing documents.",
            config.CORPUS_DIR,
        )


# ─── Schemas ─────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Employee policy question (English or Arabic)")


class CitationItem(BaseModel):
    doc_id: str
    title: str = ""
    chunk_id: str = ""


class ConflictPair(BaseModel):
    doc_a: str
    doc_b: str
    claim_a: str
    claim_b: str


class ContradictionResult(BaseModel):
    has_contradiction: bool
    conflict_pairs: list[ConflictPair] = []
    reasoning: str = ""


class TraceStep(BaseModel):
    agent: str
    step: str
    input: Any = None
    output: Any = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    confidence: str                 # high | medium | low | refused
    contradictions: ContradictionResult
    plan: dict[str, Any]
    trace: list[dict[str, Any]]     # raw trace steps for transparency


class IngestRequest(BaseModel):
    corpus_dir: str = Field(default="", description="Override corpus dir path (optional)")
    metadata_file: str = Field(default="", description="Override metadata.json path (optional)")


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    ingested: bool
    corpus_dir: str
    llm_provider: str


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        ingested=_ingested,
        corpus_dir=str(config.CORPUS_DIR),
        llm_provider=config.LLM_PROVIDER,
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask(req: QuestionRequest):
    if not _ingested:
        raise HTTPException(
            status_code=503,
            detail="Corpus not yet ingested. POST /ingest first or ensure corpus_dir is set.",
        )
    try:
        result = run_agent(req.question)
    except Exception as exc:
        logger.exception("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Normalise citations
    citations = []
    for c in result.get("citations", []):
        citations.append(
            CitationItem(
                doc_id=c.get("doc_id", ""),
                title=c.get("title", ""),
                chunk_id=c.get("chunk_id", ""),
            )
        )

    # Normalise contradictions
    contra_raw = result.get("contradictions", {})
    pairs = [
        ConflictPair(
            doc_a=p.get("doc_a", ""),
            doc_b=p.get("doc_b", ""),
            claim_a=p.get("claim_a", ""),
            claim_b=p.get("claim_b", ""),
        )
        for p in contra_raw.get("conflict_pairs", [])
    ]
    contradictions = ContradictionResult(
        has_contradiction=contra_raw.get("has_contradiction", False),
        conflict_pairs=pairs,
        reasoning=contra_raw.get("reasoning", ""),
    )

    return QuestionResponse(
        question=result["question"],
        answer=result["answer"],
        citations=citations,
        confidence=result.get("confidence", "medium"),
        contradictions=contradictions,
        plan=result.get("plan", {}),
        trace=result.get("trace", []),
    )


@app.post("/ingest", response_model=IngestResponse)
async def trigger_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    global _ingested
    corpus = Path(req.corpus_dir) if req.corpus_dir else config.CORPUS_DIR
    meta = Path(req.metadata_file) if req.metadata_file else config.METADATA_FILE

    if not corpus.exists():
        raise HTTPException(status_code=400, detail=f"Corpus directory not found: {corpus}")
    if not meta.exists():
        raise HTTPException(status_code=400, detail=f"Metadata file not found: {meta}")

    try:
        n = ingest_corpus(corpus, meta)
        _ingested = True
        return IngestResponse(status="ok", chunks_indexed=n)
    except Exception as exc:
        logger.exception("Ingest error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ─── CLI runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=config.PORT, reload=False)
