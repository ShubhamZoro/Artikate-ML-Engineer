"""
rag_pipeline.py — Main RAGPipeline class (public interface).

Orchestrates: ingestion → chunking → embedding → vector store →
              hybrid retrieval → generation → hallucination check.

Usage:
    from pipeline import RAGPipeline

    pipeline = RAGPipeline.from_documents("./sample_docs")

    result = pipeline.query("What is the notice period in the NDA with Vendor X?")
    # result: {
    #   'answer': str,
    #   'sources': [{'document': str, 'page': int, 'chunk': str}, ...],
    #   'confidence': float,
    # }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .ingestion import load_directory, IngestedCorpus
from .chunking import chunk_corpus, DocumentChunk
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


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class QueryResult:
    """Structured output of a pipeline.query() call."""

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
            f"sources={len(self.sources)}, "
            f"refused={self.refused})"
        )


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Production-grade RAG pipeline for legal document Q&A.

    Components:
      - BGE-large-en-v1.5 embeddings
      - ChromaDB persistent vector store
      - Hybrid BM25 + dense retrieval with cross-encoder re-ranking
      - OpenAI GPT-4o-mini generation with strict grounding prompt
      - Source grounding check + confidence scoring + answer refusal

    Args:
        vector_store:   Populated LegalVectorStore.
        embedder:       BGEEmbedder instance.
        retriever:      HybridRetriever instance.
        generator:      LLMGenerator instance.
        top_k:          Final number of chunks to use for generation.
    """

    def __init__(
        self,
        vector_store: LegalVectorStore,
        embedder: BGEEmbedder,
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

        Args:
            docs_dir:        Directory containing .pdf files.
            persist_dir:     Where ChromaDB persists data.
            collection_name: ChromaDB collection name.
            openai_api_key:  OpenAI API key (falls back to OPENAI_API_KEY env var).
            llm_model:       OpenAI model name.
            top_k:           Chunks passed to LLM (default 3).
            reset_store:     Wipe and re-index the vector store.

        Returns:
            Fully initialised RAGPipeline.
        """
        print("\n" + "="*60)
        print("  RAG Pipeline Initialisation")
        print("="*60)

        # ── 1. Embedder ──────────────────────────────────────────────
        embedder = OpenAIEmbedder(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            show_progress=True,
        )

        # ── 2. Vector store ──────────────────────────────────────────
        vector_store = LegalVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            reset=reset_store,
        )

        # ── 3. Ingest + chunk + embed (only if store is empty / reset) ──
        if reset_store or vector_store.count() == 0:
            print(f"\n[Ingestion] Loading PDFs from: {docs_dir}")
            corpus: IngestedCorpus = load_directory(docs_dir)
            print(f"  Total pages loaded: {len(corpus)}")

            print("\n[Chunking] Applying legal-aware hierarchical chunking …")
            chunks: list[DocumentChunk] = chunk_corpus(corpus)
            print(f"  Total chunks created: {len(chunks)}")

            print("\n[Embedding] Encoding chunks with OpenAI text-embedding-3-small …")
            embeddings = embedder.embed_chunks(chunks)
            print(f"  Embedding shape: {embeddings.shape}")

            print("\n[Indexing] Storing in ChromaDB …")
            vector_store.add_chunks(chunks, embeddings)
            print(f"  Total vectors stored: {vector_store.count()}")
        else:
            print(f"\n[Skipping ingestion] Using existing index "
                  f"({vector_store.count()} vectors)")

        # ── 4. Retriever ─────────────────────────────────────────────
        print("\n[Retriever] Building BM25 index + initialising hybrid retriever …")
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            final_k=top_k * 2,   # retrieve 2× top_k, then cut after grounding
        )

        # ── 5. Generator ─────────────────────────────────────────────
        generator = LLMGenerator(
            model=llm_model,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        )

        print("\n[Pipeline] ✓ All components ready\n")
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
        """
        Load a pre-existing pipeline from a persisted ChromaDB store.
        No ingestion is performed — the index must already exist.
        """
        print("[Pipeline] Loading from existing ChromaDB store …")
        embedder = OpenAIEmbedder(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            show_progress=False,
        )
        vector_store = LegalVectorStore(
            persist_directory=persist_dir,
            collection_name=collection_name,
            reset=False,
        )
        if vector_store.count() == 0:
            raise RuntimeError(
                f"No vectors found in ChromaDB at '{persist_dir}'. "
                "Run ingest.py first or use RAGPipeline.from_documents()."
            )
        retriever = HybridRetriever(
            vector_store=vector_store,
            embedder=embedder,
            final_k=top_k * 2,
        )
        generator = LLMGenerator(
            model=llm_model,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        )
        print(f"[Pipeline] ✓ Loaded ({vector_store.count()} vectors)\n")
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

        Args:
            question:        The user's natural-language question.
            metadata_filter: Optional ChromaDB filter to restrict search scope,
                             e.g. {"document": {"$eq": "nda_vendor_x.pdf"}}.

        Returns:
            dict with keys:
              - answer     (str)   : the generated answer
              - sources    (list)  : [{'document', 'page', 'chunk'}, ...]
              - confidence (float) : composite confidence score 0–1
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        # ── Stage 1: Retrieve ────────────────────────────────────────
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

        # Use only top_k for generation
        generation_chunks = retrieved[: self._top_k]

        # ── Stage 2: Generate ────────────────────────────────────────
        answer_text = self._generator.generate(question, generation_chunks)

        # ── Stage 3: Hallucination check ─────────────────────────────
        grounding = check_source_grounding(answer_text, generation_chunks)
        confidence = compute_confidence(generation_chunks, grounding)

        if should_refuse(confidence):
            answer_text = REFUSAL_MESSAGE

        # ── Stage 4: Assemble result ─────────────────────────────────
        sources = [chunk.to_source_dict() for chunk in generation_chunks]

        result = QueryResult(
            answer=answer_text,
            sources=sources,
            confidence=confidence,
            refused=should_refuse(confidence),
            grounding_ratio=grounding.grounding_ratio,
            retrieved_chunks=retrieved,
        )

        return result.to_dict()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_documents(self) -> list[str]:
        """Return names of all indexed documents."""
        return self._vector_store.list_documents()

    def index_size(self) -> int:
        """Return total number of indexed chunks."""
        return self._vector_store.count()
