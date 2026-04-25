# Agentic RAG Policy Advisor — Meridian Consulting

**Assessment reference:** AIE-2026-01  
**Candidate:** Sidahmed Faisal

---

## What This System Does

An agentic RAG assistant that answers employee questions about corporate policy documents. It handles three failure modes that naive RAG cannot:

1. **Contradictions** — surfaces conflicts between documents instead of silently picking one  
2. **Supersession** — answers from the current policy version by default (respects `metadata.json`)  
3. **Composition** — decomposes multi-document questions and synthesises a single grounded answer

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         FastAPI  /ask  endpoint                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   LangGraph Graph    │
                        │                     │
                        │  ┌───────────────┐  │
                        │  │  Planner Agent│  │  ← orchestrates strategy
                        │  │  (Agent 1)    │  │    single_doc / multi_doc /
                        │  └──────┬────────┘  │    out_of_scope
                        │         │           │
                        │  ┌──────▼────────┐  │
                        │  │Retriever Agent│  │  ← hybrid search + rerank
                        │  │  (Agent 2)    │  │    + contradiction check
                        │  └──────┬────────┘  │
                        │         │           │
                        │  ┌──────▼────────┐  │
                        │  │FactCheck Agent│  │  ← validates every sentence
                        │  │  (Agent 3)    │  │    against retrieved chunks
                        │  └──────┬────────┘  │
                        └─────────┼───────────┘
                                  │
                         Structured JSON response
                    (answer · citations · confidence · trace)

Retrieval Stack:
  Qdrant (dense, cosine)  +  BM25Okapi  →  RRF fusion  →  LLM reranker
  Embeddings: BAAI/bge-small-en-v1.5 (local, no API key needed)

Document Parsing:
  PDF  → PyMuPDF (fitz) → markdown
  DOCX → python-docx    → markdown
  MD   → plain read

Tracing: LangSmith (LANGCHAIN_TRACING_V2=true)
```

---

## How to Run

### Prerequisites

- Python 3.11+
- Policy corpus unzipped to `./policy_corpus/`

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY (free tier works)
# Set LANGCHAIN_API_KEY if you want LangSmith tracing
```

### 3. Ingest + Ask (CLI)

```bash
# Ingest happens automatically on first question
python ask.py "What is the standard notice period at Meridian?"
python ask.py "ما هي إجازة الحداد المخصصة للموظفين؟"   # Arabic works
```

### 4. Start the API

```bash
uvicorn api.app:app --reload --port 8000
# or:
python -m api.app
```

API docs available at `http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of paternity leave am I entitled to?"}'
```

### 5. Run Evaluation Harness

```bash
python evaluate.py \
  --questions policy_corpus/eval_questions.json \
  --output results.json
```

### 6. Docker

```bash
# Standalone (in-memory Qdrant, no extra services)
docker build -t policy-advisor .
docker run -p 8000:8000 \
  -v $(pwd)/policy_corpus:/app/policy_corpus:ro \
  --env-file .env \
  policy-advisor

# With persistent Qdrant
docker compose --profile with-qdrant up
```

---

## Switching the LLM to Gemini Free Tier

The LLM is controlled by a **single environment variable**:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
```

No code changes required. The `.env.example` defaults to Gemini.

To switch providers:

| Provider  | `LLM_PROVIDER` | Key variable       |
|-----------|----------------|--------------------|
| Gemini    | `gemini`       | `GEMINI_API_KEY`   |
| OpenAI    | `openai`       | `OPENAI_API_KEY`   |
| Anthropic | `anthropic`    | `ANTHROPIC_API_KEY`|

---

## Running evaluate.py

```bash
python evaluate.py --questions policy_corpus/eval_questions.json --output results.json
```

`results.json` structure per question:
```json
{
  "id": "Q10",
  "type": "contradiction",
  "question": "How many days of paternity leave am I entitled to?",
  "answer": "⚠️ Note: POL-HR-002 states 10 days, while POL-HR-003 states 5 days ...",
  "citations": [{"doc_id": "POL-HR-002", "title": "Parental Leave Policy", "chunk_id": "..."}],
  "confidence": "medium",
  "contradictions": {"has_contradiction": true, "conflict_pairs": [...]},
  "plan": {"strategy": "multi_doc", ...},
  "trace": [...],
  "elapsed_s": 4.2,
  "auto_flags": {"contradiction_surfaced": true, "has_citations": true}
}
```

---

## Agent Walkthrough

### Example 1: Contradiction Question (Q10 — Paternity Leave)

```
User: "How many days of paternity leave am I entitled to?"

[Planner Agent]
→ strategy: "multi_doc"
→ requires_contradiction_check: true
→ sub_queries: ["paternity leave days", "paternity leave entitlement"]

[Retriever Agent]
→ Dense search (Qdrant): hits POL-HR-002, POL-HR-003
→ BM25 search: hits same docs, also POL-HR-001
→ RRF fusion: scores merged
→ LLM reranker: top 5 chunks selected
→ check_contradictions(): LLM detects POL-HR-002 vs POL-HR-003 disagree
→ Draft answer includes: "⚠️ POL-HR-002 says X, POL-HR-003 says Y"

[FactChecker Agent]
→ Every sentence checked against chunks
→ Conflict statement supported by both chunks → kept
→ confidence: "medium" (surfaced conflict = expected behavior)
```

### Example 2: Composition Question (Q06 — UAE Travel)

```
User: "If I travel from Dubai to Abu Dhabi for a meeting with a UAE government
       client, what approvals and expense rules apply?"

[Planner Agent]
→ strategy: "multi_doc"
→ expected_categories: ["Travel", "Finance", "Client"]
→ sub_queries: [
    "UAE business travel client site approvals",
    "per diem Dubai Abu Dhabi expense",
    "government client engagement rules"
  ]

[Retriever Agent]
→ 3 sub-queries → hits across POL-TRAVEL-001-v2, POL-TRAVEL-002,
  POL-TRAVEL-003, POL-CLIENT-001
→ Supersession filter removes POL-TRAVEL-001 (old version)
→ Draft composes: approval chain + per diem + engagement rules

[FactChecker Agent]
→ Each composed sentence traced to a specific chunk
→ confidence: "high"
```

---

## Retrieval Design Decisions

### Hybrid Retrieval (Dense + BM25 + RRF)

| Signal  | Weight | Why |
|---------|--------|-----|
| Dense (BGE embeddings) | 0.7 | Catches semantic paraphrases ("annual holiday" → "annual leave") |
| BM25 | 0.3 | Catches exact policy names and codes (e.g., "POL-HR-002") |
| RRF fusion (k=60) | — | Rank-order fusion — robust to score magnitude differences |

After fusion, an LLM-based pointwise reranker scores each chunk 0–10 for relevance to the question and selects the top 5.

### Chunking

- **Size:** ~600 words with 80-word overlap
- **Rationale:** Small enough for precise citation; large enough to capture a full policy section without truncation artefacts
- **Overlap:** Prevents answer split across chunk boundary

### Metadata Filtering

Supersession is applied **pre-retrieval** in Qdrant by checking the `superseded_by` payload field. Chunks where `superseded_by` is non-null are excluded from search results by default. If the user explicitly asks about an old version, the planner sets `filter_superseded=False`.

### Cross-lingual Retrieval

Arabic questions are detected via Unicode range `\u0600–\u06ff`. The planner prompt instructs the LLM to **also generate English sub-queries** alongside any Arabic ones. Retrieval runs on English corpus with English queries; the answer is composed in Arabic.

---

## LangSmith Tracing

Set these in `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key
LANGCHAIN_PROJECT=rag-policy-advisor
```

Every `run_agent()` call produces a full trace in LangSmith showing:
- Planner node input/output (plan JSON)
- Retriever node: sub-queries, Qdrant hits, BM25 hits, RRF scores, reranking, contradiction detection
- FactChecker node: unsupported sentences removed, final citations

---

## API Reference

### `POST /ask`

```json
{
  "question": "string (English or Arabic)"
}
```

Response:
```json
{
  "question": "...",
  "answer": "...",
  "citations": [{"doc_id": "...", "title": "...", "chunk_id": "..."}],
  "confidence": "high | medium | low | refused",
  "contradictions": {
    "has_contradiction": false,
    "conflict_pairs": [],
    "reasoning": "..."
  },
  "plan": {"strategy": "...", "reasoning": "...", "sub_queries": [...]},
  "trace": [{"agent": "...", "step": "...", ...}]
}
```

### `POST /ingest`

Re-ingest corpus (idempotent). Accepts optional `corpus_dir` and `metadata_file` overrides.

### `GET /health`

Returns `{"status": "ok", "ingested": true, "llm_provider": "gemini"}`.

---

## Known Weaknesses

### 1. In-memory BM25 index is lost on process restart

The BM25 index is rebuilt in-memory on every startup (triggered on first question). This adds ~5-15 seconds cold-start latency. **Fix:** Persist BM25 index to disk (pickle) or replace with Qdrant's built-in sparse vector support.

### 2. LLM reranker is expensive for large result sets

Reranking 20 chunks with individual LLM calls (one per chunk) costs ~20 API round-trips. On Gemini free tier this is slow (~30-60s for large queries). **Fix:** Use a cross-encoder model locally (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) which runs in a single batch.

### 3. Contradiction detection is recall-limited

The contradiction checker only operates on the top-K retrieved chunks. If two conflicting documents score below the top-K threshold for a query, the conflict is missed. **Fix:** Run a targeted second retrieval pass specifically for documents in the same category as the already-retrieved set.

### 4. Arabic support is heuristic-only

Language detection uses a Unicode character count heuristic. Mixed-language queries (e.g., Arabic question with English policy codes) may be misclassified. The system always retrieves from the English corpus but only guarantees Arabic *output* — it does not verify that the Arabic answer faithfully translates the English source content. **Fix:** Add a dedicated Arabic NLI translation step.

### 5. Chunking ignores document structure

The current fixed-size chunking can split a policy clause mid-sentence. Structural chunking (split at section headings, parsed from the markdown headers extracted by PyMuPDF) would improve citation precision. **Fix:** Use heading-aware chunking — detect `##` headers and treat each section as a primary unit.

---

## What I Would Build Next (with another week)

1. **Persistent Qdrant + sparse vectors** — Run Qdrant as a Docker service with sparse SPLADE vectors replacing BM25, giving true hybrid search without an in-process index.
2. **Structured citation spans** — Return character offsets into the source document (not just chunk IDs), enabling the UI to highlight the exact sentence in the original PDF.
3. **Cross-encoder reranking** — Replace LLM pointwise reranker with a local `ms-marco` cross-encoder for 10× faster reranking at higher precision.
4. **Version-aware multi-turn** — Track conversation history in LangGraph state so employees can ask follow-up questions ("what about the old policy?") without re-stating context.
5. **Automatic contradiction discovery** — Offline job that runs `check_contradictions` over all document pairs in the same category and stores the results, so contradiction detection at query time is O(1) lookup rather than an LLM call.
