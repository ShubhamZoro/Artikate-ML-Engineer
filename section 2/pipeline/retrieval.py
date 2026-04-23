"""
retrieval.py — Hybrid BM25 + Dense retrieval with RRF fusion.

Pipeline:
  1. Dense search via ChromaDB (top-20 candidates) using OpenAI embeddings
  2. BM25 keyword search (top-20 candidates)
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
  4. Final top-K selected by RRF score

Note: Cross-encoder re-ranking (sentence-transformers) has been replaced
by RRF-based ranking to keep the stack fully OpenAI-based.
RRF is robust, parameter-free, and performs comparably to cross-encoders
on short candidate sets (<25 docs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Final retrieved chunk after RRF fusion, returned to the pipeline."""
    chunk_id: str
    document: str
    page_number: int
    section_title: str
    text: str
    dense_score: float    # cosine similarity from vector search
    bm25_score: float     # normalised BM25 score
    rrf_score: float      # Reciprocal Rank Fusion score (final ranking signal)
    rerank_score: float   # same as rrf_score (kept for API compatibility)
    final_rank: int

    def to_source_dict(self) -> dict:
        return {
            "document": self.document,
            "page": self.page_number,
            "chunk": self.text,
        }


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Lightweight in-memory BM25 index over all corpus chunks.
    Rebuilt from ChromaDB on each pipeline initialisation.
    At 50k-doc scale this would be replaced by Elasticsearch.
    """

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
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
        print(f"BM25 index built: {len(texts)} chunks")

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call .build() first.")
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def get_chunk_metadata(self, corpus_index: int) -> dict:
        return self._metadatas[corpus_index]

    def get_chunk_text(self, corpus_index: int) -> str:
        return self._texts[corpus_index]

    def get_chunk_id(self, corpus_index: int) -> str:
        return self._chunk_ids[corpus_index]

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
    """
    Merge multiple ranked lists of chunk_ids using RRF.
    score(d) = Σ 1 / (k + rank(d))  for each ranked list.
    """
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
    Combines dense vector search (OpenAI embeddings), BM25, and RRF fusion.

    Args:
        vector_store:  Populated LegalVectorStore.
        embedder:      OpenAIEmbedder for query embedding.
        candidate_k:   How many candidates to retrieve per modality before RRF.
        final_k:       Final number of results after RRF ranking.
    """

    def __init__(
        self,
        vector_store: LegalVectorStore,
        embedder: OpenAIEmbedder,
        candidate_k: int = 30,
        final_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._final_k = final_k

        # Build BM25 index from ChromaDB corpus
        self._bm25 = BM25Index()
        self._rebuild_bm25()

        # Cache: chunk_id → (text, metadata) to avoid repeated DB round-trips
        self._chunk_cache: dict[str, tuple[str, dict]] = {}
        self._build_cache()

    def _rebuild_bm25(self) -> None:
        texts, metadatas, ids = self._vector_store.get_all_chunks_text()
        if texts:
            self._bm25.build(texts, metadatas, ids)

    def _build_cache(self) -> None:
        texts, metadatas, ids = self._vector_store.get_all_chunks_text()
        for chunk_id, text, meta in zip(ids, texts, metadatas):
            self._chunk_cache[chunk_id] = (text, meta)

    def _expand_query_for_bm25(self, query: str) -> str:
        """
        Inject document-name tokens when the query explicitly names a document.
        E.g. "shuttle service agreement" → appends "SampleContract Shuttle"
        so BM25 strongly boosts chunks from that document.
        """
        query_lower = query.lower()
        known_docs = self._vector_store.list_documents()
        extra_tokens: list[str] = []
        for doc in known_docs:
            # Strip extension and split into name parts
            stem = doc.replace(".pdf", "").replace("_", " ").replace("-", " ")
            stem_words = [w for w in stem.lower().split() if len(w) > 3]
            matches = sum(1 for w in stem_words if w in query_lower)
            if matches >= 1:
                extra_tokens.extend(stem.split())
        if extra_tokens:
            return query + " " + " ".join(extra_tokens)
        return query

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        metadata_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval for a query.

        Args:
            query:           User question string.
            metadata_filter: Optional ChromaDB metadata filter dict.

        Returns:
            List of RetrievedChunk objects (length ≤ self._final_k),
            sorted by RRF score descending.
        """
        # ── Stage 1: Dense retrieval ────────────────────────────────
        query_embedding = self._embedder.embed_query(query)
        dense_results: list[VectorSearchResult] = self._vector_store.search(
            query_embedding,
            top_k=self._candidate_k,
            where=metadata_filter,
        )
        dense_scores: dict[str, float] = {r.chunk_id: r.score for r in dense_results}
        dense_ranked: list[str] = [r.chunk_id for r in dense_results]

        # ── Stage 2: BM25 retrieval ─────────────────────────────────
        # Expand query with document-name tokens when query references a
        # specific document (e.g. "shuttle agreement" → boosts Shuttle PDF)
        bm25_query = self._expand_query_for_bm25(query)
        bm25_hits = self._bm25.search(bm25_query, top_k=self._candidate_k)
        bm25_scores_raw: dict[str, float] = {}
        bm25_ranked: list[str] = []
        for idx, score in bm25_hits:
            cid = self._bm25.get_chunk_id(idx)
            bm25_scores_raw[cid] = score
            bm25_ranked.append(cid)

        # Normalise BM25 scores to [0, 1]
        if bm25_scores_raw:
            max_bm25 = max(bm25_scores_raw.values()) or 1.0
            bm25_scores: dict[str, float] = {
                cid: s / max_bm25 for cid, s in bm25_scores_raw.items()
            }
        else:
            bm25_scores = {}

        # ── Stage 3: RRF fusion ──────────────────────────────────────
        rrf_scores = _reciprocal_rank_fusion([dense_ranked, bm25_ranked])

        # Sort all candidates by RRF score descending → final ranking
        sorted_candidates = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        # ── Stage 4: Page-diversity deduplication ───────────────────
        # Prefer results from different (document, page) pairs so that
        # top-K slots aren't all wasted on the same (wrong) page.
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

        # Fill up to final_k: diverse first, then overflow
        top = (diverse + overflow)[: self._final_k]

        # ── Stage 5: Assemble final results ─────────────────────────
        final_chunks: list[RetrievedChunk] = []
        for rank, (cid, rrf_score) in enumerate(top, start=1):
            if cid in self._chunk_cache:
                text, meta = self._chunk_cache[cid]
            else:
                # Fallback: find in dense results
                text, meta = "", {}
                for r in dense_results:
                    if r.chunk_id == cid:
                        text = r.text
                        break

            final_chunks.append(RetrievedChunk(
                chunk_id=cid,
                document=meta.get("document", ""),
                page_number=int(meta.get("page_number", 0)),
                section_title=meta.get("section_title", ""),
                text=text,
                dense_score=float(dense_scores.get(cid, 0.0)),
                bm25_score=float(bm25_scores.get(cid, 0.0)),
                rrf_score=float(rrf_score),
                rerank_score=float(rrf_score),   # kept for API compatibility
                final_rank=rank,
            ))

        return final_chunks
