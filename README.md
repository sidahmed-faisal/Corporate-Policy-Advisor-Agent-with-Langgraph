# Agentic RAG Policy Advisor 


---

## What This System Does

An agentic RAG assistant that answers employee questions about corporate policy documents. It handles three failure modes that naive RAG cannot:

1. **Contradictions** — surfaces conflicts between documents instead of silently picking one
2. **Supersession** — answers from the current policy version by default (respects `metadata.json`)
3. **Composition** — combines information across multiple documents into a single grounded answer

---

## Architecture

The system is a single ReAct-style agent loop. The LLM reasons through every step — planning, retrieval, contradiction checking, fact-checking — by deciding which tools to call and in what order.

```
                    ┌─────────────────────────────────────┐
                    │         FastAPI  /ask  endpoint      │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │           LangGraph (3 nodes)        │
                    │                                      │
                    │   ┌─────────┐     ┌─────────────┐   │
                    │   │  agent  │────▶│    tools    │   │
                    │   │  (LLM)  │◀────│  (ToolNode) │   │
                    │   └────┬────┘     └─────────────┘   │
                    │        │ no more tool_calls          │
                    │   ┌────▼────┐                        │
                    │   │finalize │                        │
                    │   └─────────┘                        │
                    └──────────────────────────────────────┘
                                       │
                          Structured JSON response
                   (answer · citations · confidence · trace)

Tools the LLM can call:
  retrieve_policy          — hybrid Qdrant dense + BM25 search
  get_doc_metadata         — version and supersession chain lookup
  check_for_contradictions — conflict detection across retrieved docs
  verify_and_finalize      — fact-check draft and produce final answer

Retrieval stack:
  Qdrant (dense cosine)  +  BM25Okapi  →  RRF fusion
  Embeddings: nomic-ai/nomic-embed-text-v1.5 (local, no API key needed)

Document parsing:
  PDF  → PyMuPDF (fitz) → markdown
  DOCX → python-docx    → markdown
  MD   → plain read

Tracing: LangSmith (set LANGCHAIN_TRACING_V2=true)
```

---

## How to Run

### Prerequisites

- Python 3.11+
- Policy corpus unzipped to `./policy_corpus/`
- (Optional) A running Qdrant instance — if you skip this, Qdrant runs in-process in memory and the index is rebuilt on every cold start

---

### Option A — Local development without Docker (recommended for iteration)

This is the fastest setup for development: a Python virtual environment plus a single Qdrant binary running on your machine. No Docker daemon required.

#### 1. Create and activate a virtual environment

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Confirm you are in the venv:

```bash
which python   # should point to .../.venv/bin/python
python --version  # 3.11.x
```

#### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first install takes 2–4 minutes — `sentence-transformers` will download the embedding model (`nomic-ai/nomic-embed-text-v1.5`, ~550 MB) into `~/.cache/huggingface/` on first use, not at install time.

#### 3. Run a local Qdrant instance (persistent)


**Option 1 — Native Qdrant binary (no Docker, fastest startup)**

Download the platform-specific release from the [Qdrant releases page](https://github.com/qdrant/qdrant/releases) and extract:

```bash
# macOS (Apple Silicon) — adjust version to latest
curl -L -o qdrant.tar.gz \
  https://github.com/qdrant/qdrant/releases/latest/download/qdrant-aarch64-apple-darwin.tar.gz
tar -xzf qdrant.tar.gz

# Run it (data persists in ./qdrant_storage)
./qdrant
```

Qdrant will listen on `http://localhost:6333` (REST) and `:6334` (gRPC). Leave this terminal running.

#### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
LLM_PROVIDER=gemini                                # or openai | anthropic
GEMINI_API_KEY=your-gemini-api-key-here

# If you ran a local Qdrant binary in step 3:
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=policy_docs

# If you skipped Qdrant (Option A.2): leave QDRANT_URL blank
# QDRANT_URL=

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key
LANGCHAIN_PROJECT=rag-policy-advisor

CORPUS_DIR=./policy_corpus
METADATA_FILE=./policy_corpus/metadata.json
```

#### 5. Ingest + ask (CLI)

The first question triggers ingestion automatically. To force ingestion up front:

```bash
python -c "from ingestion.ingest import ingest_corpus; import config; print(ingest_corpus(config.CORPUS_DIR, config.METADATA_FILE), 'chunks indexed')"
```

Then ask:

```bash
python ask.py "What is the standard notice period at Meridian?"
python ask.py "ما هي إجازة الحداد المخصصة للموظفين؟"   # Arabic works
```

#### 6. Start the API

```bash
uvicorn api.app:app --reload --port 8000
# or
python -m api.app
```

Interactive docs: `http://localhost:8000/docs`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of paternity leave am I entitled to?"}'
```

#### 7. Run the evaluation harness

```bash
python evaluate.py \
  --questions policy_corpus/eval_questions.json \
  --output results.json
```

#### 8. Tear down

```bash
# Stop the API (Ctrl-C in its terminal)
# Stop Qdrant   (Ctrl-C in its terminal)
deactivate                       # exit the venv
rm -rf qdrant_storage            # optional — wipe persisted vectors
```

---

### Option B — Docker / Docker Compose

```bash

# API + persistent Qdrant
docker compose up
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

## Agent Walkthrough — Comparative Behaviour on Real Traces

This section walks through the actual LangSmith-captured traces from running the same nine-question evaluation against two LLM backends:

- **`gemini-3-flash-preview`** (`LLM_PROVIDER=gemini`)
- **`gpt-4o`** (`LLM_PROVIDER=openai`)

The traces are in `Gemini_policy_test.jsonl` and `OpenAI_policy_test.jsonl`. The agent loop, prompts, retrieval stack, and tool definitions are identical between runs — only the model is swapped via `LLM_PROVIDER`. Differences in behaviour below therefore reflect how each model *uses* the same toolset.

The four tools the agent exposes:

| Tool | Purpose |
|------|---------|
| `retrieve_policy(query, filter_superseded=True)` | Hybrid retrieval (Qdrant dense + BM25 + RRF fusion). The `filter_superseded` flag drops chunks whose `superseded_by` payload is non-null. |
| `get_doc_metadata(doc_id)` | Returns the version, effective date, and supersession chain for a single doc. |
| `check_for_contradictions(question, chunks_json)` | LLM-judges whether the retrieved chunks disagree on the answer. |
| `verify_and_finalize(question, chunks_json, draft_answer)` | Sentence-level groundedness check; returns the final answer + citations + confidence. |

### Headline numbers

| Question type | Gemini tool calls | GPT-4o tool calls | Gemini graph steps | GPT-4o graph steps |
|---|---|---|---|---|
| Single-doc — notice period | 3 | **1** | 8 | **4** |
| Single-doc — Europe hotel cap | 3 | **1** | 8 | **4** |
| Arabic — bereavement leave | 3 | **1** | 8 | **4** |
| Supersession — travel request lead time | 4 | **1** | 10 | **4** |
| Supersession — remote work days | 5 | **1** | 12 | **4** |
| Contradiction — paternity leave | 3 | 3 | 8 | 8 |
| Contradiction — sick-day carry-forward | 5 | 3 | 12 | 8 |
| Composition — birth-of-child total leave (run 1) | 5 | 4 | 8 | 8 |
| Composition — birth-of-child total leave (run 2) | 6 | 4 | 10 | 8 |

GPT-4o averages **2× fewer tool calls** and **1.5–3× fewer graph steps** per question. Lower step counts translate directly into lower latency and lower token cost (each agent step on Gemini is a separate LLM round-trip with the full tool schema in context). On a free Gemini-3-flash-preview tier where rate limits and ~1–3 s/step latency dominate end-to-end time, this gap is significant.

The reason for the gap, however, is not "GPT-4o is smarter" — it is that **GPT-4o reads and uses the `filter_superseded` parameter on the retrieval tool, while Gemini-3-flash-preview ignores it on every single question**. The behavioural consequences are most visible on the supersession cases.

---

### Case 1 — Supersession: "How far in advance do I need to submit a travel request?"

**Ground truth.** The corpus contains both `POL-TRAVEL-001` (7 working days, superseded) and `POL-TRAVEL-001-v2` (10 working days, current as of 2025-09-01). The expected answer is **10 working days**, citing the v2 doc.

**GPT-4o trace** (1 tool call, 4 graph steps):

```
→ retrieve_policy(query="travel request submission deadline", filter_superseded=True)
   ↳ Qdrant pre-filters out POL-TRAVEL-001; only v2 returned
✓ Final answer (no verify_and_finalize call): "10 working days … updated from 7 to 10
  as of September 1, 2025 [POL-TRAVEL-001-v2]"
```

GPT-4o trusts the retrieval filter. One round-trip, correct answer. **However** — because it skipped `verify_and_finalize`, the response shipped with `confidence: low` and an empty `citations: []` array even though the doc ID is in the answer text. **This is a real defect in agent prompt design**: GPT-4o's "I'm done" heuristic kicks in too early and the structured citation block never gets populated. A user reading the JSON envelope would see a low-confidence answer with no machine-readable citations, which is worse for downstream consumers than a slower-but-fully-populated Gemini response.

**Gemini-3-flash-preview trace** (4 tool calls, 10 graph steps):

```
→ retrieve_policy(query="travel request submission lead time")          ← no filter
→ retrieve_policy(query="travel request form submission deadline")      ← retry, broader
→ check_for_contradictions(...)                                          ← v1 vs v2 chunks both present
→ verify_and_finalize(draft_answer="...10 working days [POL-TRAVEL-001-v2]...")
✓ Final answer: same 10 working days, but also adds nuance from POL-TRAVEL-003 about
  same-day intra-UAE travel. confidence: high, citations: [POL-TRAVEL-001-v2, POL-TRAVEL-003]
```

Gemini *infers* supersession from the chunk content rather than filtering it pre-retrieval. Both v1 and v2 land in the candidate set; the model reads the effective dates inside the chunks and the version suffix in the doc IDs (`-v2`), then composes an answer from v2 only. It also detours into a contradiction check that returns no conflict (the two travel-request lead times are not really a "contradiction" once supersession is accounted for — they are just one being out of date).

**Critical assessment.** Gemini reaches the right answer through the wrong mechanism. Inference-from-text works *here* because the doc ID literally contains `-v2`; on a corpus where superseded versions share a doc ID and only differ on `effective_date` payload, this strategy would fail. The intended design — supersession handled by the retrieval filter — is what GPT-4o exercises and Gemini bypasses.

### Case 2 — Supersession: "How many days per week can I work remotely?"

The pattern is identical and even more pronounced.

| Metric | GPT-4o | Gemini-3-flash-preview |
|---|---|---|
| Tool calls | **1** (retrieve_policy with filter_superseded=True) | **5** (3× retrieve_policy + get_doc_metadata + verify_and_finalize) |
| Graph steps | 4 | 12 |
| Final answer | "3 days per week … remote-first roles exempt" | Same answer, plus 4 extra bullets on core hours and office expectations |
| Citations | `[]` (empty due to skipped verify) | `[POL-HR-004-v2, POL-IT-002]` |
| Confidence | `low` | `high` |

Gemini explicitly calls `get_doc_metadata(doc_id="POL-HR-004-v2")` to confirm it is the live version — useful belt-and-braces, but redundant given that the metadata file was already used to populate the Qdrant payloads at ingestion time. **Both filtering at retrieval time (GPT-4o) and confirming via metadata lookup (Gemini) reach the correct policy**; the filter-based path is roughly 5× cheaper in tool calls.

### Case 3 — Contradiction: "How many days of paternity leave am I entitled to?"

This is where the two models converge.

```
Both models:
→ retrieve_policy(...)              ← pulls POL-HR-002 (5 days) and POL-HR-003 (7 days)
→ check_for_contradictions(...)     ← returns has_contradiction=true
→ verify_and_finalize(...)
✓ confidence: high, citations: [POL-HR-002, POL-HR-003]
```

Both surface the conflict. The answers diverge in style:

- **GPT-4o** is terse: `"⚠️ Conflict: POL-HR-002 says 5 working days, POL-HR-003 says 7 working days. Please verify with HR."`
- **Gemini** quotes more of the source and adds the precedence rule from the handbook: `"POL-HR-003 explicitly notes that where its summaries conflict with standalone policy documents, the standalone policy is authoritative."` That nuance is genuinely present in the corpus and Gemini surfaces it; GPT-4o omits it.

For a user, Gemini's answer is more actionable. For an evaluator scoring "did the agent flag the conflict?", both pass.

### Case 4 — Contradiction: "Can I carry forward unused sick days to next year?"

Now the models *diverge in correctness*.

The corpus contains:
- `POL-HR-011` (Wellness Leave): allows up to 10 days carry-forward
- `POL-HR-001` (Leave — General Overview): "sick leave does not accumulate across years"

These genuinely contradict.

**GPT-4o** retrieved only `POL-HR-011`. `check_for_contradictions` ran on a single-document set and returned no conflict. Final answer: `"Yes, up to 10 days … encourages employees to take sick leave when genuinely needed."` Confident, single-cited, and **silently wrong about the existence of a conflicting policy**. Confidence reported as `medium` — the only signal a downstream consumer has.

**Gemini** retrieved both POL-HR-011 and POL-HR-001 (likely because BM25 caught "sick leave" in the general overview), called `get_doc_metadata` on each, ran the contradiction checker, and produced: `"⚠️ Conflict: POL-HR-011 allows 10 days carry-forward, POL-HR-001 says sick leave does not accumulate. POL-HR-011 is the more specific document and likely represents current practice — verify with HR."`

**This is the recall-limited contradiction failure mode the original "Known Weaknesses" section calls out, made concrete.** GPT-4o's parsimonious one-shot retrieval missed a conflicting document; Gemini's broader retrieval (2 hits with different lexical anchors) caught it. The fix — running a second retrieval pass scoped to the same category as the first hit — would close this gap regardless of which model is driving.

### Case 5 — Composition: "Total leave in first year combining paternity and annual leave"

Both models correctly identify this requires composing across `POL-HR-002` (paternity) and `POL-HR-001` (annual leave), and both surface the paternity-leave conflict that bleeds in from the previous case.

| Metric | GPT-4o | Gemini |
|---|---|---|
| Tool calls | 4 | 5–6 (varies across the two recorded runs) |
| Final number | "27 to 29 days, depending on the correct paternity entitlement" | "27 working days" with the conflict noted as a side annotation |
| Citations | `[POL-HR-001, POL-HR-002, POL-HR-003]` | Same three docs |

Two judgement calls visible in these traces:

1. **GPT-4o hedges in the headline number.** It refuses to commit to a single figure because of the upstream paternity-leave conflict. Defensible but less directly useful.
2. **Gemini commits to 27 days** (using POL-HR-002's 5-day figure as authoritative because POL-HR-003 itself defers to standalone policies) and notes the conflict as a footnote. More useful, more interpretive, and arguably leaning further from the ground truth than GPT-4o is comfortable with.

Neither answer is wrong; they reflect different failure modes for ambiguous source content. A corporate deployment would likely prefer Gemini's behaviour for end-user UX and GPT-4o's for compliance/audit contexts.

### Case 6 — Single-doc & Arabic

For the three "easy" questions (notice period, hotel cap, Arabic bereavement leave):

- **GPT-4o**: 1 retrieval, no verify, correct answer, but `confidence: low` and `citations: []` in the JSON envelope on every one of these.
- **Gemini**: 3 tool calls every time (retrieve → check_for_contradictions → verify_and_finalize), correct answer, `confidence: high`, citations populated.

Gemini's "always run the full pipeline" behaviour is wasteful on simple questions but produces fully-populated structured output. GPT-4o's "I have enough, stop calling tools" behaviour is efficient but **trips a bug in the agent prompt where confidence and citations only get set when `verify_and_finalize` runs**. This is the single most actionable finding from the comparison — the agent prompt should require `verify_and_finalize` as the terminal step regardless of model.

---

### Links for Runs and datasets used for evaluation:

#### Runs examples: 

* https://smith.langchain.com/public/917f7650-5e3f-4dc4-b350-5c1ee7df3312/r
* https://smith.langchain.com/public/3d6ea398-1805-4a91-bcb4-e65eb2d16d8c/r
* https://smith.langchain.com/public/93f60cb9-b61b-480f-8359-015fa7de8411/r

#### Datasets for Gemini vs gpt-4o: 

* Gemini: https://smith.langchain.com/public/27f584c7-3231-47f8-ae43-c6a45a0bd995/d
* gpt-4o: https://smith.langchain.com/public/44d49de8-af1c-4bc6-b84a-92f5729416e4/d

---

### Summary of judgement calls

| Dimension | Winner | Caveat |
|---|---|---|
| Tool-call efficiency | GPT-4o | But skips `verify_and_finalize` ⇒ empty citations/low confidence in the response envelope |
| Tool semantics (uses `filter_superseded`) | GPT-4o | Gemini-3-flash-preview ignores the parameter and infers from chunk content instead |
| Supersession correctness | Tie | GPT-4o via filter, Gemini via inference; both arrive at the right policy |
| Contradiction recall | Gemini | GPT-4o's single-shot retrieval missed the POL-HR-001 vs POL-HR-011 conflict on sick days |
| Answer richness / nuance | Gemini | Surfaces precedence rules, edge cases, document hierarchy |
| Latency / cost | GPT-4o | Roughly half the tool calls and graph steps on this evaluation |
| Structured-output quality | Gemini | Confidence and citations are reliably populated |
| Cross-lingual (Arabic) | Tie | Both produce a fluent Arabic answer; Gemini's includes citations in the envelope |

The fact that both models reach correct final answers on 8/9 questions despite using the toolset very differently is itself evidence that the agent loop is working: even when one model bypasses an intended affordance (the retrieval filter), the loop's other safety nets (broader retrieval, contradiction check, verification) compensate. The 1/9 miss is the GPT-4o sick-day case, which is a retrieval-recall problem, not a model problem.

---

## Retrieval Design Decisions

### Hybrid Retrieval (Dense + BM25 + RRF)

| Signal  | Weight | Why |
|---------|--------|-----|
| Dense (Nomic embeddings) | 0.7 | Catches semantic paraphrases ("annual holiday" → "annual leave") |
| BM25 | 0.3 | Catches exact policy names and codes (e.g., "POL-HR-002") |
| RRF fusion (k=60) | — | Rank-order fusion — robust to score magnitude differences |

After fusion, the top-K (default 8, configurable via `TOP_K`) RRF-ranked chunks are returned directly to the agent — there is no separate reranking stage. See *Known Weaknesses §2* for why a cross-encoder reranker is the obvious next addition.

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
- Retriever node: sub-queries, Qdrant hits, BM25 hits, RRF scores, contradiction detection
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

### 2. No reranking stage after RRF fusion

Retrieval ends at RRF fusion of the dense and BM25 hit lists — the top-K RRF-ranked chunks are passed straight to the agent, with no second-pass scoring. This is fine for a small corpus where lexical and semantic signals already agree closely on the top results, but it leaves precision on the table on harder queries: RRF is a rank-only fusion, so it cannot tell apart a chunk that mentions all the query terms in passing from one that *answers* the question. **Fix:** add a cross-encoder reranker between fusion and return — a local `cross-encoder/ms-marco-MiniLM-L-6-v2` runs the top-20 fused candidates in a single batch on CPU, picks the top-5, and adds well under a second of latency. This would also reduce the load the agent currently puts on `verify_and_finalize` to compensate for noisier candidate sets.

### 3. Contradiction detection is recall-limited

The contradiction checker only operates on the top-K retrieved chunks. If two conflicting documents score below the top-K threshold for a query, the conflict is missed. **Fix:** Run a targeted second retrieval pass specifically for documents in the same category as the already-retrieved set.

### 4. Chunking ignores document structure

The current fixed-size chunking can split a policy clause mid-sentence. Structural chunking (split at section headings, parsed from the markdown headers extracted by PyMuPDF) would improve citation precision. **Fix:** Use heading-aware chunking — detect `##` headers and treat each section as a primary unit.

---

## What I Would Build Next (with another week)

1. **Persistent Qdrant + sparse vectors** — Run Qdrant as a Docker service with sparse SPLADE vectors replacing BM25, giving true hybrid search without an in-process index.
2. **Structured citation spans** — Return character offsets into the source document (not just chunk IDs), enabling the UI to highlight the exact sentence in the original PDF.
3. **Cross-encoder reranking** — Add a local `ms-marco` cross-encoder between RRF fusion and the agent (currently no reranker runs), using a single batched CPU inference call to rescore the top-20 fused candidates and return the top-5 — higher precision at well under a second of added latency.
4. **Version-aware multi-turn** — Track conversation history in LangGraph state so employees can ask follow-up questions ("what about the old policy?") without re-stating context.
5. **Automatic contradiction discovery** — Offline job that runs `check_contradictions` over all document pairs in the same category and stores the results, so contradiction detection at query time is O(1) lookup rather than an LLM call.


## Walkthrough the execution of the project:

* https://www.loom.com/share/a1b31c86278b45a48fd31afaaeb26d1f