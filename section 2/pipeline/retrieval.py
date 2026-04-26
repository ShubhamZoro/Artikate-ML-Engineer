"""
retrieval.py — Hybrid BM25 + Dense retrieval with RRF fusion.

Pipeline:
  1. Dense search via ChromaDB children collection (top-30 candidates)
  2. BM25 keyword search (top-30 candidates)
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. Page-diversity deduplication
  5. Batch-fetch parent texts from ChromaDB parents collection (single .get() call)

Parent expansion is now a single batched ChromaDB .get() — no JSON sidecar,
no RAM pre-load, no disk I/O outside ChromaDB's own storage engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import OpenAIEmbedder
from .vectorstore import LegalVectorStore, VectorSearchResult


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Final retrieved chunk returned to the pipeline after parent expansion."""
    chunk_id: str
    document: str
    page_number: int
    section_title: str
    text: str                 # parent text (512t) for LLM generation
    dense_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float       # alias for rrf_score, kept for API compatibility
    final_rank: int

    def to_source_dict(self) -> dict:
        return {
            "document": self.document,
            "page": self.page_number,
            "chunk": self.text,
        }


# ---------------------------------------------------------------------------
# BM25 index (in-memory, rebuilt from ChromaDB on startup)
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Lightweight in-memory BM25 index over all child corpus chunks.
    Rebuilt from ChromaDB on each pipeline initialisation.
    At 50k-doc scale replace with Elasticsearch.
    """

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._chunk_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def build(
        self,
        texts: list[str],
        metadatas: list[dict],
        chunk_ids: list[str],
    ) -> None:
        self._texts = texts
        self._metadatas = metadatas
        self._chunk_ids = chunk_ids
        self._corpus_tokens = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        print(f"BM25 index built: {len(texts)} child chunks")

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call .build() first.")
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def get_metadata(self, idx: int) -> dict:
        return self._metadatas[idx]

    def get_text(self, idx: int) -> str:
        return self._texts[idx]

    def get_chunk_id(self, idx: int) -> str:
        return self._chunk_ids[idx]

    @property
    def size(self) -> int:
        return len(self._texts)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Merge ranked lists of chunk_ids: score(d) = Σ 1/(k + rank(d))."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Dense + BM25 hybrid retrieval with RRF fusion and parent chunk expansion.

    Parent expansion uses a single batched ChromaDB .get() call after
    the final top-K is selected — one round-trip regardless of K.

    Args:
        vector_store:  Populated LegalVectorStore (children + parents).
        embedder:      OpenAIEmbedder for query embedding.
        candidate_k:   Candidates retrieved per modality before RRF.
        final_k:       Final results after RRF + diversity dedup.
    """

    def __init__(
        self,
        vector_store: LegalVectorStore,
        embedder: OpenAIEmbedder,
        candidate_k: int = 30,
        final_k: int = 6,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._final_k = final_k

        # Build BM25 and chunk cache from the children collection
        self._bm25 = BM25Index()
        self._chunk_cache: dict[str, tuple[str, dict]] = {}
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        texts, metadatas, ids = self._vector_store.get_all_chunks_text()
        if texts:
            self._bm25.build(texts, metadatas, ids)
            for cid, text, meta in zip(ids, texts, metadatas):
                self._chunk_cache[cid] = (text, meta)

    def _expand_bm25_query(self, query: str) -> str:
        """
        Append document-name tokens when the query references a known document.
        Boosts BM25 recall for queries like "shuttle service agreement".
        """
        query_lower = query.lower()
        extra: list[str] = []
        for doc in self._vector_store.list_documents():
            stem_words = [
                w for w in doc.replace(".pdf", "").replace("_", " ").replace("-", " ").lower().split()
                if len(w) > 3
            ]
            if sum(1 for w in stem_words if w in query_lower) >= 1:
                extra.extend(stem_words)
        return (query + " " + " ".join(extra)) if extra else query

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval and return parent-expanded results.

        Steps:
          1. Dense search (ChromaDB children, top candidate_k)
          2. BM25 search (in-memory, top candidate_k)
          3. RRF fusion
          4. Page-diversity deduplication → top final_k child IDs
          5. Batch-fetch parent texts (ONE ChromaDB .get() call)
          6. Assemble RetrievedChunk objects
        """

        # ── 1. Dense retrieval ───────────────────────────────────────
        query_emb = self._embedder.embed_query(query)
        dense_results: list[VectorSearchResult] = self._vector_store.search(
            query_emb,
            top_k=self._candidate_k,
            where=metadata_filter,
        )
        dense_scores = {r.chunk_id: r.score for r in dense_results}
        dense_ranked = [r.chunk_id for r in dense_results]

        # ── 2. BM25 retrieval ────────────────────────────────────────
        bm25_query = self._expand_bm25_query(query)
        bm25_hits = self._bm25.search(bm25_query, top_k=self._candidate_k)

        bm25_scores_raw: dict[str, float] = {}
        bm25_ranked: list[str] = []
        for idx, score in bm25_hits:
            cid = self._bm25.get_chunk_id(idx)
            bm25_scores_raw[cid] = score
            bm25_ranked.append(cid)

        max_bm25 = max(bm25_scores_raw.values(), default=1.0) or 1.0
        bm25_scores = {cid: s / max_bm25 for cid, s in bm25_scores_raw.items()}

        # ── 3. RRF fusion ────────────────────────────────────────────
        rrf_scores = _reciprocal_rank_fusion([dense_ranked, bm25_ranked])
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ── 4. Page-diversity deduplication ─────────────────────────
        seen_pages: set[tuple[str, int]] = set()
        diverse: list[tuple[str, float]] = []
        overflow: list[tuple[str, float]] = []

        for cid, rrf_score in sorted_candidates:
            meta = self._chunk_cache.get(cid, ("", {}))[1]
            page_key = (meta.get("document", ""), int(meta.get("page_number", 0)))
            if page_key not in seen_pages:
                seen_pages.add(page_key)
                diverse.append((cid, rrf_score))
            else:
                overflow.append((cid, rrf_score))

        top_k: list[tuple[str, float]] = (diverse + overflow)[: self._final_k]

        # ── 5. Batch-fetch parent texts (single ChromaDB round-trip) ─
        parent_ids = [
            self._chunk_cache.get(cid, ("", {}))[1].get("parent_chunk_id", "")
            for cid, _ in top_k
        ]
        # One .get() call for ALL needed parents — not one per chunk
        parent_map = self._vector_store.get_parents(
            [pid for pid in parent_ids if pid]
        )

        # ── 6. Assemble results ──────────────────────────────────────
        final_chunks: list[RetrievedChunk] = []
        for rank, ((cid, rrf_score), parent_id) in enumerate(
            zip(top_k, parent_ids), start=1
        ):
            child_text, meta = self._chunk_cache.get(cid, ("", {}))

            # Prefer parent text for richer LLM context; fall back to child
            parent = parent_map.get(parent_id)
            generation_text = parent["text"] if parent else child_text

            final_chunks.append(RetrievedChunk(
                chunk_id=cid,
                document=meta.get("document", ""),
                page_number=int(meta.get("page_number", 0)),
                section_title=meta.get("section_title", ""),
                text=generation_text,
                dense_score=float(dense_scores.get(cid, 0.0)),
                bm25_score=float(bm25_scores.get(cid, 0.0)),
                rrf_score=float(rrf_score),
                rerank_score=float(rrf_score),
                final_rank=rank,
            ))

        return final_chunks