"""
ask.py  —  Interactive RAG query script for the Legal Document pipeline.

Drop this file into the root of `section 2/` (next to ingest.py / query.py).
Run:
    python ask.py                          # interactive loop
    python ask.py --once                   # single question from CLI prompt
    python ask.py -q "What is notice period?"   # pass question directly

Requirements:  the ChromaDB index must already exist (run ingest.py first).
All dependencies are already in requirements.txt.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import math
import re
import time
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── optional colour output (works on all platforms) ───────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False


# =============================================================================
# Config  —  mirror the defaults used in ingest.py / query.py
# =============================================================================

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR    = os.path.join(BASE_DIR, "chroma_db")
COLLECTION     = "legal_rag"
LLM_MODEL      = "gpt-4o-mini"
TOP_K          = 3          # chunks passed to the LLM
CANDIDATE_K    = 30         # candidates retrieved per modality before fusion
RRF_K          = 60         # RRF damping constant
CONFIDENCE_THRESHOLD = 0.30 # below this → answer refused

SYSTEM_PROMPT = """You are a precise legal document analyst.

RULES — follow them strictly:
1. Answer ONLY using the provided context excerpts. Do not use external knowledge.
2. If the answer is not present in the context, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
3. Quote relevant clause text verbatim where possible, using double quotes.
4. Always end your answer with the source reference:
   [Source: <document name>, Page <number>]
   If the answer spans multiple sources, list each one.
5. Be concise. Do not pad the answer.
6. Never fabricate names, dates, amounts, or clause numbers."""

REFUSAL = (
    "I cannot find sufficient information in the provided documents "
    "to answer this question confidently. Please consult the source documents directly."
)


# =============================================================================
# Lazy imports  —  only loaded after we know the venv is active
# =============================================================================

def _import_deps():
    """Import heavy dependencies and return them as a namespace."""
    try:
        import chromadb
        from chromadb.config import Settings
        from openai import OpenAI
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        print(f"\n[ERROR] Missing dependency: {exc}")
        print("  Make sure your virtual environment is active and run:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)
    return chromadb, Settings, OpenAI, BM25Okapi


# =============================================================================
# ChromaDB  —  connect to both collections
# =============================================================================

def connect_store(persist_dir: str, collection: str):
    chromadb, Settings, _, _ = _import_deps()
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    children_name = f"{collection}_children"
    parents_name  = f"{collection}_parents"

    try:
        children = client.get_collection(children_name)
        parents  = client.get_collection(parents_name)
    except Exception:
        print(f"\n[ERROR] Collections '{children_name}' / '{parents_name}' not found in:")
        print(f"  {persist_dir}")
        print("\n  Run ingest.py first:\n  python ingest.py --docs_dir ./my_pdfs\n")
        sys.exit(1)

    n_children = children.count()
    n_parents  = parents.count()
    _print(f"[DB] {n_children} child chunks  |  {n_parents} parent chunks", style="dim")
    return children, parents


# =============================================================================
# Embedding  —  OpenAI text-embedding-3-small
# =============================================================================

def embed_query(query: str, client) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query.replace("\n", " ")],
    )
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# =============================================================================
# BM25 index  —  built once from all child texts
# =============================================================================

def build_bm25(children_col):
    _, _, _, BM25Okapi = _import_deps()
    result = children_col.get(include=["documents", "metadatas"])
    texts     = result["documents"]
    metadatas = result["metadatas"]
    ids       = result["ids"]

    tokenised = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenised)

    # chunk cache: id → (text, metadata)
    cache = {cid: (txt, meta) for cid, txt, meta in zip(ids, texts, metadatas)}
    _print(f"[BM25] Index built over {len(texts)} child chunks", style="dim")
    return bm25, ids, cache


# =============================================================================
# Hybrid retrieval  —  Dense + BM25 → RRF → page-dedup → parent expansion
# =============================================================================

def retrieve(
    query: str,
    children_col,
    parents_col,
    openai_client,
    bm25,
    bm25_ids: list[str],
    chunk_cache: dict,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:

    # ── 1. Dense retrieval ────────────────────────────────────────────
    query_vec = embed_query(query, openai_client)
    dense_res = children_col.query(
        query_embeddings=[query_vec],
        n_results=min(candidate_k, children_col.count()),
        include=["documents", "metadatas", "distances"],
    )
    dense_ids    = dense_res["ids"][0]
    dense_scores = {
        cid: float(1.0 - dist)
        for cid, dist in zip(dense_res["ids"][0], dense_res["distances"][0])
    }

    # ── 2. BM25 retrieval ─────────────────────────────────────────────
    query_tokens = query.lower().split()
    bm25_scores_raw = bm25.get_scores(query_tokens)
    top_bm25_idx = np.argsort(bm25_scores_raw)[::-1][:candidate_k]
    bm25_ranked  = [bm25_ids[i] for i in top_bm25_idx if bm25_scores_raw[i] > 0]
    max_bm25     = max((bm25_scores_raw[i] for i in top_bm25_idx), default=1.0) or 1.0
    bm25_scores  = {
        bm25_ids[i]: float(bm25_scores_raw[i]) / max_bm25
        for i in top_bm25_idx if bm25_scores_raw[i] > 0
    }

    # ── 3. RRF fusion ─────────────────────────────────────────────────
    rrf: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(bm25_ranked, start=1):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    sorted_cands = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    # ── 4. Page-diversity deduplication ──────────────────────────────
    seen_pages: set[tuple] = set()
    diverse: list[tuple]   = []
    overflow: list[tuple]  = []
    for cid, rrf_score in sorted_cands:
        meta = chunk_cache.get(cid, ("", {}))[1]
        page_key = (meta.get("document", ""), int(meta.get("page_number", 0)))
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            diverse.append((cid, rrf_score))
        else:
            overflow.append((cid, rrf_score))

    top_k_list = (diverse + overflow)[:top_k]

    # ── 5. Batch parent expansion (ONE .get() call) ───────────────────
    parent_ids = [
        chunk_cache.get(cid, ("", {}))[1].get("parent_chunk_id", "")
        for cid, _ in top_k_list
    ]
    unique_pids = list(dict.fromkeys(p for p in parent_ids if p))
    parent_map: dict[str, dict] = {}
    if unique_pids:
        try:
            pr = parents_col.get(ids=unique_pids, include=["documents", "metadatas"])
            for pid, ptxt, pmeta in zip(pr["ids"], pr["documents"], pr["metadatas"]):
                parent_map[pid] = {
                    "text":          ptxt,
                    "document":      pmeta.get("document", ""),
                    "page_number":   int(pmeta.get("page_number", 0)),
                    "section_title": pmeta.get("section_title", ""),
                }
        except Exception as exc:
            _print(f"[WARN] Parent fetch failed: {exc}", style="yellow")

    # ── 6. Assemble results ───────────────────────────────────────────
    results = []
    for rank, ((cid, rrf_score), parent_id) in enumerate(
        zip(top_k_list, parent_ids), start=1
    ):
        child_text, meta = chunk_cache.get(cid, ("", {}))
        parent           = parent_map.get(parent_id)
        gen_text         = parent["text"] if parent else child_text

        results.append({
            "rank":          rank,
            "chunk_id":      cid,
            "document":      meta.get("document", ""),
            "page_number":   int(meta.get("page_number", 0)),
            "section_title": meta.get("section_title", ""),
            "child_text":    child_text,
            "text":          gen_text,       # parent text → LLM context
            "dense_score":   dense_scores.get(cid, 0.0),
            "bm25_score":    bm25_scores.get(cid, 0.0),
            "rrf_score":     rrf_score,
        })

    return results


# =============================================================================
# Hallucination check  —  source grounding + confidence
# =============================================================================

def _extract_phrases(text: str) -> list[str]:
    phrases = []
    phrases += re.findall(
        r'(?:₹|Rs\.?|INR|USD|\$)\s*[\d,]+(?:\.\d+)?\s*(?:crore|lakh|thousand|million)?',
        text, flags=re.IGNORECASE,
    )
    phrases += re.findall(r'"([^"]{3,60})"', text)
    phrases += re.findall(r'\b(?:[A-Z][a-z]+\s+){1,4}[A-Z][a-z]+\b', text)
    phrases += re.findall(r'\b\d+\s+(?:days?|months?|years?|weeks?)\b', text, re.IGNORECASE)
    phrases += re.findall(r'\b\d{1,3}(?:,\d{3})*\b', text)
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        p = p.strip()
        if len(p) >= 3 and p not in seen:
            seen.add(p); out.append(p)
    return out

def grounding_ratio(answer: str, chunks: list[dict]) -> float:
    phrases = _extract_phrases(answer)
    if not phrases:
        return 0.5
    combined = " ".join(c["text"].lower() for c in chunks)
    matched  = sum(1 for p in phrases if p.lower() in combined)
    return matched / len(phrases)

def confidence_score(chunks: list[dict], grounding: float) -> float:
    if not chunks:
        return 0.0
    top_rrf   = chunks[0]["rrf_score"]
    sig_rrf   = 1.0 / (1.0 + math.exp(-top_rrf * 30))   # sigmoid
    scores    = [1.0 / (1.0 + math.exp(-c["rrf_score"] * 30)) for c in chunks]
    spread    = max(scores) - min(scores) if len(scores) > 1 else 0.0
    agreement = max(0.0, 1.0 - spread)
    return round(min(1.0, max(0.0,
        0.40 * grounding + 0.40 * sig_rrf + 0.20 * agreement
    )), 4)


# =============================================================================
# Generation  —  GPT-4o-mini
# =============================================================================

def generate(question: str, chunks: list[dict], openai_client) -> str:
    context_parts = []
    for i, c in enumerate(chunks, start=1):
        context_parts.append(
            f"--- Context {i} ---\n"
            f"Document: {c['document']}\n"
            f"Page: {c['page_number']}\n"
            f"Section: {c['section_title'] or 'N/A'}\n\n"
            f"{c['text']}"
        )
    user_msg = (
        "CONTEXT EXCERPTS:\n"
        + "\n\n".join(context_parts)
        + f"\n\nQUESTION: {question}\n\nANSWER (cite sources inline):"
    )
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# Display helpers
# =============================================================================

def _print(msg: str, style: str = ""):
    if _RICH:
        console.print(msg, style=style)
    else:
        print(msg)

def print_retrieval(chunks: list[dict]):
    if _RICH:
        table = Table(
            title="Retrieved chunks",
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold",
        )
        table.add_column("#",        width=3,  justify="right",  style="dim")
        table.add_column("Document", max_width=30)
        table.add_column("Page",     width=5,  justify="center")
        table.add_column("RRF",      width=7,  justify="right")
        table.add_column("Dense",    width=7,  justify="right")
        table.add_column("BM25",     width=7,  justify="right")
        table.add_column("Child preview (256t)", max_width=45, style="dim")
        for c in chunks:
            table.add_row(
                str(c["rank"]),
                c["document"],
                str(c["page_number"]),
                f"{c['rrf_score']:.4f}",
                f"{c['dense_score']:.3f}",
                f"{c['bm25_score']:.3f}",
                c["child_text"][:80].replace("\n", " ") + "…",
            )
        console.print(table)
    else:
        print("\n--- Retrieved chunks ---")
        for c in chunks:
            print(
                f"  #{c['rank']}  {c['document']}  p.{c['page_number']}"
                f"  RRF={c['rrf_score']:.4f}"
                f"  dense={c['dense_score']:.3f}"
                f"  bm25={c['bm25_score']:.3f}"
            )
            print(f"     \"{c['child_text'][:80].strip()}…\"")


def print_answer(answer: str, conf: float, chunks: list[dict]):
    label = "HIGH" if conf >= 0.6 else ("MEDIUM" if conf >= 0.3 else "LOW")
    colour = "green" if conf >= 0.6 else ("yellow" if conf >= 0.3 else "red")
    if _RICH:
        console.print(Panel(
            answer,
            title=f"[bold]Answer[/bold]  —  confidence {conf:.3f} [{label}]",
            border_style=colour,
        ))
        console.print("\n[bold]Sources:[/bold]")
        for c in chunks:
            console.print(f"  → [cyan]{c['document']}[/cyan]  p.{c['page_number']}")
    else:
        print(f"\n=== Answer  (confidence={conf:.3f} [{label}]) ===")
        print(answer)
        print("\nSources:")
        for c in chunks:
            print(f"  → {c['document']}  p.{c['page_number']}")


# =============================================================================
# Core ask() function  —  the single entry point
# =============================================================================

def ask(
    question: str,
    children_col,
    parents_col,
    openai_client,
    bm25,
    bm25_ids: list[str],
    chunk_cache: dict,
    verbose: bool = True,
) -> dict:
    """
    Run the full RAG pipeline for one question.

    Returns:
        {
          "answer":     str,
          "confidence": float,
          "refused":    bool,
          "sources":    [{"document": str, "page": int, "chunk": str}, …],
          "chunks":     list[dict],   # full retrieval detail
        }
    """
    if verbose:
        _print(f"\n[bold]Question:[/bold] {question}")

    t0 = time.perf_counter()

    # ── retrieve ──────────────────────────────────────────────────────
    chunks = retrieve(
        question, children_col, parents_col,
        openai_client, bm25, bm25_ids, chunk_cache,
    )

    if not chunks:
        if verbose:
            _print("[red]No chunks retrieved — is the index populated?[/red]")
        return {"answer": REFUSAL, "confidence": 0.0, "refused": True, "sources": [], "chunks": []}

    if verbose:
        print_retrieval(chunks)

    # ── generate ──────────────────────────────────────────────────────
    answer_text = generate(question, chunks[:TOP_K], openai_client)

    # ── hallucination check ───────────────────────────────────────────
    gr   = grounding_ratio(answer_text, chunks[:TOP_K])
    conf = confidence_score(chunks[:TOP_K], gr)
    refused = conf < CONFIDENCE_THRESHOLD

    if refused:
        answer_text = REFUSAL

    elapsed = time.perf_counter() - t0

    if verbose:
        print_answer(answer_text, conf, chunks[:TOP_K])
        _print(f"\n[dim]Latency: {elapsed:.2f}s  |  grounding: {gr:.2f}  |  confidence: {conf:.3f}[/dim]")

    return {
        "answer":     answer_text,
        "confidence": conf,
        "refused":    refused,
        "sources":    [{"document": c["document"], "page": c["page_number"], "chunk": c["text"]} for c in chunks[:TOP_K]],
        "chunks":     chunks,
    }


# =============================================================================
# Main  —  interactive loop or single-shot
# =============================================================================

def main():
    global LLM_MODEL, TOP_K
    parser = argparse.ArgumentParser(
        description="Ask questions against the Legal RAG vector store."
    )
    parser.add_argument(
        "--question", "-q",
        default="",
        help="Ask a single question and exit (non-interactive mode)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Prompt for one question then exit",
    )
    parser.add_argument(
        "--persist_dir",
        default=PERSIST_DIR,
        help=f"ChromaDB directory (default: {PERSIST_DIR})",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION,
        help=f"Collection base name (default: {COLLECTION})",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
        help=f"Chunks passed to LLM (default: {TOP_K})",
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"OpenAI model (default: {LLM_MODEL})",
    )
    args = parser.parse_args()

    LLM_MODEL = args.model
    TOP_K     = args.top_k

    # ── validate API key ──────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    # ── lazy import + connect ─────────────────────────────────────────
    _, _, OpenAI, _ = _import_deps()
    openai_client   = OpenAI(api_key=api_key)

    _print("\n[bold]Legal RAG — Interactive Query[/bold]")
    _print(f"  Persist dir : {args.persist_dir}")
    _print(f"  Collection  : {args.collection}")
    _print(f"  LLM model   : {LLM_MODEL}")
    _print(f"  Top-K       : {TOP_K}")

    children_col, parents_col = connect_store(args.persist_dir, args.collection)
    bm25, bm25_ids, chunk_cache = build_bm25(children_col)

    # ── single question via CLI flag ──────────────────────────────────
    if args.question:
        ask(args.question, children_col, parents_col, openai_client, bm25, bm25_ids, chunk_cache)
        return

    # ── single prompt then exit ───────────────────────────────────────
    if args.once:
        q = input("\nYour question: ").strip()
        if q:
            ask(q, children_col, parents_col, openai_client, bm25, bm25_ids, chunk_cache)
        return

    # ── interactive loop ──────────────────────────────────────────────
    _print("\nType your question and press Enter. Type [bold]exit[/bold] or [bold]quit[/bold] to stop.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\n[dim]Bye![/dim]")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            _print("[dim]Bye![/dim]")
            break

        ask(q, children_col, parents_col, openai_client, bm25, bm25_ids, chunk_cache)
        print()


if __name__ == "__main__":
    main()