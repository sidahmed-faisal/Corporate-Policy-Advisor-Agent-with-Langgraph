"""
agents/workflow.py
────────────────────────────────────────────────────────────────────────────────
Single-agent ReAct loop. The LLM has all five tools and reasons through every
step itself — plan, retrieve, check contradictions, fact-check, answer.

Graph: 3 nodes only
  agent  → the LLM (decides which tool to call, or emits the final answer)
  tools  → ToolNode (executes whichever tool the LLM chose)
  agent  (loop back until no more tool_calls)

LangSmith tracing is automatic when LANGCHAIN_TRACING_V2=true.
"""
from __future__ import annotations

import json
import logging
import operator
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from llm_client import get_llm
from tools.rag_tools import (
    check_contradictions as _check_contradictions_impl,
    get_document_metadata as _get_metadata_impl,
    retrieve as _retrieve_impl,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State — just the message history + extracted outputs for the API response
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    # Populated when the agent finishes (extracted from final AIMessage)
    answer: str
    citations: list[dict]
    confidence: str
    contradictions: dict[str, Any]
    trace: Annotated[list[dict], operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# Helper for tool implementations that need an LLM call
# ─────────────────────────────────────────────────────────────────────────────
def _llm(prompt: str) -> str:
    response = get_llm().invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        # Gemini returns a list of blocks when thought signatures are present
        parts = [b["text"] for b in content if isinstance(b, dict) and "text" in b]
        content = "\n".join(parts)
    return content.strip()



# ─────────────────────────────────────────────────────────────────────────────
# Tools — the LLM decides which to call and when
# ─────────────────────────────────────────────────────────────────────────────

@tool
def retrieve_policy(query: str, top_k: int = 8, filter_superseded: bool = True) -> str:
    """
    Search Meridian's policy corpus for chunks relevant to the query.

    Call this for every distinct search angle — different phrasings surface
    different documents. Always call this at least once before answering.
    The corpus is English-only; translate Arabic questions before searching.

    Args:
        query: Short English search string (3-8 words).
        top_k: Number of chunks to return (default 8, max 12).
        filter_superseded: If True (default), excludes superseded policy versions.
            Set False only when the user explicitly asks about an old policy version.

    Returns:
        JSON list of chunks. Each chunk has: doc_id, title, effective_date,
        superseded_by, score, and the FULL policy text of that chunk.
        Multiple chunks from the same document may be returned — read all of them.
    """
    results = _retrieve_impl(query=query, top_k=min(top_k, 12), filter_superseded=filter_superseded)
    # Return full text — never truncate here; truncation causes missed facts downstream
    return json.dumps(results, ensure_ascii=False)


@tool
def get_doc_metadata(doc_id: str) -> str:
    """
    Get metadata for a document: effective date, supersession chain, category.

    Call this when a retrieved chunk has a non-null superseded_by field, or when
    you need to confirm which version of a policy is currently in force.

    Args:
        doc_id: Document ID such as "POL-HR-004" or "POL-TRAVEL-001-v2".

    Returns:
        JSON with title, category, department, effective_date, superseded_by,
        supersedes, and resolved titles for the version chain.
    """
    return json.dumps(_get_metadata_impl(doc_id), ensure_ascii=False)


@tool
def check_for_contradictions(chunks_json: str, question: str) -> str:
    """
    Detect whether any retrieved chunks contradict each other on this question.

    Call this whenever you have chunks from more than one document on the same
    topic. MANDATORY before answering any question about leave entitlements,
    sick days, training budget, or expense limits — these areas have known
    conflicts between documents.

    # IMPORTANT: read the FULL text of every chunk before calling this tool. The chunks should address the same policy area to use this tool effectively.
    Args:
        chunks_json: JSON string — the list of chunk dicts from retrieve_policy.
        question: The original employee question.

    Returns:
        JSON: { "has_contradiction": bool, "conflict_pairs": [...], "reasoning": str }
        Each conflict_pair contains: doc_a, doc_b, claim_a, claim_b.
    """
    try:
        chunks = json.loads(chunks_json)
    except json.JSONDecodeError:
        return json.dumps({"has_contradiction": False, "conflict_pairs": [], "reasoning": "Parse error."})
    return json.dumps(_check_contradictions_impl(chunks, question, _llm), ensure_ascii=False)


@tool
def verify_and_finalize(draft_answer: str, chunks_json: str, question: str) -> str:
    """
    Fact-check the draft answer against retrieved chunks, then produce the
    final structured response.

    Call this ONCE after you have finished all retrieval and contradiction checks.
    It removes any unsupported claims and returns the final answer as JSON.

    Args:
        draft_answer: The answer you have composed from the retrieved policy text.
        chunks_json: JSON string — all chunks retrieved during this session.
        question: The original employee question.

    Returns:
        JSON:
        {
          "answer": "final grounded answer text",
          "citations": [{"doc_id": "...", "title": "...", "chunk_id": "..."}],
          "confidence": "high" | "medium" | "low",
          "unsupported_removed": ["list of sentences that were cut"]
        }
    """
    try:
        chunks = json.loads(chunks_json)
    except json.JSONDecodeError:
        chunks = []

    # Include the FULL text of every chunk — truncation was causing false "unsupported" verdicts
    chunk_summary = "\n\n".join(
        f"[{c.get('chunk_id','?')}|{c.get('doc_id','?')}|{c.get('title','')}]\n{c.get('text','')}"
        for c in chunks[:10]
    )

    prompt = f"""You are a fact-checker for a corporate policy assistant.

Original question: {question}

DRAFT ANSWER:
{draft_answer}

RETRIEVED POLICY CHUNKS (authorised sources — read every word carefully):
{chunk_summary}

Task:
1. Read every chunk in full above.
2. For each sentence in the draft answer and each factual claim and information, check whether the information it contains
   appears ANYWHERE in the chunks — exact wording, paraphrase, or clear implication all count.
3. Only remove a sentence if the information is genuinely absent from ALL chunks.
   Do NOT remove a sentence just because the wording differs from the chunk.
   Keep ⚠️ conflict warnings always.
4. Return ONLY valid JSON (no markdown fences):
{{
  "answer": "revised answer with all supported claims kept",
  "citations": [{{"doc_id": "...", "title": "...", "chunk_id": "..."}}],
  "confidence": "high" | "medium" | "low",
  "unsupported_removed": ["only sentences with zero support in any chunk"]
}}
confidence: high=everything supported, medium=1 sentence removed, low=multiple removed."""

    raw = _llm(prompt).strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        json.loads(raw)   # validate
        return raw
    except json.JSONDecodeError:
        # Safe fallback — return draft as-is
        fallback = {
            "answer": draft_answer,
            "citations": [
                {"doc_id": c.get("doc_id", ""), "title": c.get("title", ""), "chunk_id": c.get("chunk_id", "")}
                for c in chunks[:5]
            ],
            "confidence": "medium",
            "unsupported_removed": [],
        }
        return json.dumps(fallback, ensure_ascii=False)


TOOLS = [retrieve_policy, get_doc_metadata, check_for_contradictions, verify_and_finalize]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — everything the agent needs to reason through a question
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a corporate policy advisor at Meridian Consulting. You answer employee
questions about internal policy documents by calling tools to search, verify,
and fact-check before responding.

Available tools:
  • retrieve_policy           — search the policy corpus (call once per search angle)
  • get_doc_metadata          — check a document's version and supersession chain
  • check_for_contradictions  — detect conflicts between retrieved documents
  • verify_and_finalize       — fact-check your draft and produce the final answer JSON

Reasoning protocol (follow in order):

1. PLAN — identify which policy areas are relevant. For multi-part questions,
   plan one search query per distinct topic in the question.
   If the question is in Arabic, plan English sub-queries (corpus is English-only).

2. RETRIEVE — call retrieve_policy for each search angle (1-3 calls). Use short,
   specific English queries. Always use filter_superseded=True unless the user
   explicitly asks about an old policy version.
   IMPORTANT: Read the FULL text of every chunk returned — all the facts you need
   are in those chunks. Do not assume a chunk is irrelevant before reading it.

3. CHECK VERSION — if any retrieved chunk has a non-null superseded_by field,
   call get_doc_metadata to confirm which version is current.

4. CHECK CONTRADICTIONS — if chunks from more than one document address the same
   topic and policy area, call check_for_contradictions. Mandatory for: leave, sick days,
   training budget, expenses.

5. DRAFT — compose your answer using ONLY information from the retrieved chunks.
   Cite every claim as [DOC-ID]. Surface any conflicts as:
   "⚠️ Conflict: [DOC-A] states X, but [DOC-B] states Y — verify with HR/Operations."
   For Arabic questions, write the answer in Arabic (keep DOC-IDs in English).

6. FINALIZE — call verify_and_finalize with:
   - draft_answer: your complete drafted answer
   - chunks_json: ALL chunks you retrieved across ALL retrieve_policy calls
     (merge them into one JSON array — do not pass only the last call's results)
   - question: the original question

Out-of-scope: if the question cannot be answered from Meridian policy documents,
call verify_and_finalize with draft_answer="I cannot find an authoritative answer
in the policy corpus." and chunks_json="[]".

Never fabricate policy content. Never answer without calling verify_and_finalize.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes — just 3
# ─────────────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """LLM step: reason and decide which tool to call next (or finalize)."""
    llm = get_llm().bind_tools(TOOLS)
    response = llm.invoke(state["messages"])

    tool_calls = getattr(response, "tool_calls", []) or []
    trace_entry = {
        "step": "agent",
        "tool_calls": [{"name": tc["name"], "args": {k: str(v)[:120] for k, v in tc["args"].items()}}
                       for tc in tool_calls],
        "has_text": bool(response.content and not tool_calls),
    }
    return {"messages": [response], "trace": [trace_entry]}


def tools_node(state: AgentState) -> dict:
    """Execute whichever tool the LLM chose — ToolNode dispatches by name."""
    result = ToolNode(TOOLS).invoke({"messages": state["messages"]})
    new_msgs: list[BaseMessage] = result["messages"]

    trace_entry = {
        "step": "tool_results",
        "results": [
            {"tool": m.name, "preview": (m.content[:200] if isinstance(m.content, str) else "")[:200]}
            for m in new_msgs if isinstance(m, ToolMessage)
        ],
    }
    return {"messages": new_msgs, "trace": [trace_entry]}


def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    """
    The LLM decides whether to keep calling tools or stop.
    If its last message has tool_calls → keep going.
    If it emitted plain text (shouldn't happen with our prompt) → also finalize.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


def finalize_node(state: AgentState) -> dict:
    """
    Extract the final answer from the last verify_and_finalize ToolMessage.
    Falls back to the last AIMessage text if parsing fails.
    """
    messages = state["messages"]

    # Find the most recent verify_and_finalize result
    final_json: dict | None = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "verify_and_finalize":
            try:
                final_json = json.loads(msg.content)
                break
            except (json.JSONDecodeError, TypeError):
                pass

    if final_json:
        return {
            "answer": final_json.get("answer", ""),
            "citations": final_json.get("citations", []),
            "confidence": final_json.get("confidence", "medium"),
            "contradictions": {},   # surfaced inline in the answer
            "trace": [{"step": "finalize", "source": "verify_and_finalize",
                       "confidence": final_json.get("confidence"),
                       "unsupported_removed": final_json.get("unsupported_removed", [])}],
        }

    # Fallback: use last AIMessage text
    fallback_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            fallback_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    return {
        "answer": fallback_text,
        "citations": [],
        "confidence": "low",
        "contradictions": {},
        "trace": [{"step": "finalize", "source": "fallback_text"}],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build the graph — 3 nodes, 1 conditional edge
# ─────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    b = StateGraph(AgentState)

    b.add_node("agent",    agent_node)
    b.add_node("tools",    tools_node)
    b.add_node("finalize", finalize_node)

    b.set_entry_point("agent")
    b.add_conditional_edges("agent", should_continue,
                            {"tools": "tools", "finalize": "finalize"})
    b.add_edge("tools", "agent")   # always loop back to the LLM
    b.add_edge("finalize", END)

    return b.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_agent(question: str) -> dict[str, Any]:
    graph = get_graph()

    initial: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "answer": "",
        "citations": [],
        "confidence": "medium",
        "contradictions": {},
        "trace": [],
    }

    final = graph.invoke(initial)

    # Pull contradiction info out of ToolMessages for the API response
    contradictions: dict = {"has_contradiction": False, "conflict_pairs": [], "reasoning": ""}
    for msg in final["messages"]:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "check_for_contradictions":
            try:
                contradictions = json.loads(msg.content)
                break
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        "question": question,
        "answer": final["answer"],
        "citations": final.get("citations", []),
        "confidence": final.get("confidence", "medium"),
        "contradictions": contradictions,
        "trace": final.get("trace", []),
    }