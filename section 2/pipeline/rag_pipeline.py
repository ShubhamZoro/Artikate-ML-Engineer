"""
rag_pipeline.py — Main RAGPipeline class (public interface).

Orchestrates: ingestion → hierarchical chunking → embedding →
              vector store (children + parents) → hybrid retrieval →
              generation → hallucination check.

Parent chunks are now stored in a second ChromaDB collection and fetched
on demand via a single batched .get() call. The JSON sidecar is gone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .ingestion import load_directory, IngestedCorpus
from .chunking import chunk_corpus, DocumentChunk, ParentChunk
from .embeddings import OpenAIEmbedder
from .vectorstore import LegalVectorStore
from .retrieval import HybridRetriever, RetrievedChunk
from .generation import LLMGenerator
from .hallucination import (
    check_source_grounding,
    compute_confidence,
    should_refuse,
    REFUSAL_MESSAGE,
)


class QueryResult:
    def __init__(
        self,
        answer: str,
        sources: list[dict],
        confidence: float,
        refused: bool = False,
        grounding_ratio: float = 0.0,
        retrieved_chunks: Optional[list[RetrievedChunk]] = None,
    ) -> None:
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.refused = refused
        self.grounding_ratio = grounding_ratio
        self.retrieved_chunks = retrieved_chunks or []

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        return (
            f"QueryResult(confidence={self.confidence:.3f}, "
            f"sources={len(self.sources)}, refused={self.refused})"
        )


class RAGPipeline:
    """
    Production-grade RAG pipeline for legal document Q&A.

    Components:
      - OpenAI text-embedding-3-small embeddings
      - ChromaDB children collection (child chunks, 256t, vector-indexed)
      - ChromaDB parents collection  (parent chunks, 512t, fetched by ID)
      - Hybrid BM25 + dense retrieval with RRF fusion
      - Batch parent expansion (single .get() per query, not one per chunk)
      - OpenAI GPT-4o-mini generation with strict grounding prompt
      - Source grounding check + confidence scoring + answer refusal
    """

    def __init__(
        self,
        vector_store: LegalVectorStore,
        embedder: OpenAIEmbedder,
        retriever: HybridRetriever,
        generator: LLMGenerator,
        top_k: int = 3,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    # ------------------------------------------------------------------
    # Factory: build pipeline from a directory of PDFs
    # ------------------------------------------------------------------

    @classmethod
    def from_documents(
        cls,
        docs_dir: str | Path,
        persist_dir: str | Path = "./chroma_db",
        collection_name: str = "legal_rag",
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        top_k: int = 3,
        reset_store: bool = False,
    ) -> "RAGPipeline":
        """
        Build a fully initialised pipeline by ingesting a directory of PDFs.

        On first run (or reset_store=True):
          - Loads PDFs, chunks into parent+child pairs
          - Embeds child chunks, stores in ChromaDB children collection
          - Stores parent texts in ChromaDB parents collection (no embeddings)

        On subsequent runs (index exists):
          - Skips ingestion entirely, loads both collections from disk
        """
        print("\n" + "=" * 60)
        print("  RAG Pipeline Initialisation")
        print("=" * 60)

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # ── Embedder ─────────────────────────────────────────────────
        embedder = OpenAIEmbedder(api_key=api_key, show_progress=True)

        # ── Vector store (children + parents) ────────────────────────
        vector_store = LegalVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            reset=reset_store,
        )

        # ── Ingest only when the children collection is empty / reset ─
        if reset_store or vector_store.count() == 0:
            print(f"\n[Ingestion] Loading PDFs from: {docs_dir}")
            corpus: IngestedCorpus = load_directory(docs_dir)
            print(f"  Total pages loaded: {len(corpus)}")

            print("\n[Chunking] Applying hierarchical parent-child chunking …")
            parents, children = chunk_corpus(corpus)
            print(f"  Parent chunks : {len(parents)} (512t, generation context)")
            print(f"  Child chunks  : {len(children)} (256t, retrieval index)")

            print("\n[Embedding] Encoding child chunks …")
            embeddings = embedder.embed_chunks(children)
            print(f"  Embedding shape: {embeddings.shape}")

            print("\n[Indexing] Storing child chunks in ChromaDB children collection …")
            vector_store.add_chunks(children, embeddings)

            print("\n[Indexing] Storing parent chunks in ChromaDB parents collection …")
            vector_store.add_parents(parents)

        else:
            print(
                f"\n[Skipping ingestion] Existing index: "
                f"{vector_store.count()} child chunks, "
                f"{vector_store.parent_count()} parent chunks"
            )

        # ── Retriever ─────────────────────────────────────────────────
        print("\n[Retriever] Building BM25 index …")
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            final_k=top_k * 2,
        )

        # ── Generator ─────────────────────────────────────────────────
        generator = LLMGenerator(model=llm_model, api_key=api_key)

        print("\n[Pipeline] All components ready\n")
        return cls(
            vector_store=vector_store,
            embedder=embedder,
            retriever=retriever,
            generator=generator,
            top_k=top_k,
        )

    @classmethod
    def load(
        cls,
        persist_dir: str | Path = "./chroma_db",
        collection_name: str = "legal_rag",
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        top_k: int = 3,
    ) -> "RAGPipeline":
        """Load a pre-existing pipeline from persisted ChromaDB collections."""
        print("[Pipeline] Loading from existing store …")
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        embedder = OpenAIEmbedder(api_key=api_key, show_progress=False)
        vector_store = LegalVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            reset=False,
        )

        if vector_store.count() == 0:
            raise RuntimeError(
                f"No child chunks found in ChromaDB at '{persist_dir}'. "
                "Run ingest.py first or use RAGPipeline.from_documents()."
            )

        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            final_k=top_k * 2,
        )
        generator = LLMGenerator(model=llm_model, api_key=api_key)

        print(
            f"[Pipeline] Loaded — "
            f"{vector_store.count()} child chunks, "
            f"{vector_store.parent_count()} parent chunks\n"
        )
        return cls(
            vector_store=vector_store,
            embedder=embedder,
            retriever=retriever,
            generator=generator,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Core query interface
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        metadata_filter: Optional[dict] = None,
    ) -> dict:
        """
        Answer a question using the RAG pipeline.

        Returns dict with keys: answer (str), sources (list), confidence (float).
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        retrieved: list[RetrievedChunk] = self._retriever.retrieve(
            query=question,
            metadata_filter=metadata_filter,
        )

        if not retrieved:
            return QueryResult(
                answer=REFUSAL_MESSAGE,
                sources=[],
                confidence=0.0,
                refused=True,
            ).to_dict()

        generation_chunks = retrieved[: self._top_k]

        answer_text = self._generator.generate(question, generation_chunks)

        grounding = check_source_grounding(answer_text, generation_chunks)
        confidence = compute_confidence(generation_chunks, grounding)

        if should_refuse(confidence):
            answer_text = REFUSAL_MESSAGE

        sources = [chunk.to_source_dict() for chunk in generation_chunks]

        return QueryResult(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            refused=should_refuse(confidence),
            grounding_ratio=grounding.grounding_ratio,
            retrieved_chunks=retrieved,
        ).to_dict()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_documents(self) -> list[str]:
        return self._vector_store.list_documents()

    def index_size(self) -> int:
        return self._vector_store.count()