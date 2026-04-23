"""
vectorstore.py — ChromaDB persistent vector store wrapper + ParentChunkStore.

Stores child chunk embeddings + metadata (document, page, section, chunk_id)
in ChromaDB for retrieval.

Stores parent chunk texts in a JSON sidecar file so the retriever can
expand each retrieved child chunk to its richer parent context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings

from .chunking import DocumentChunk, ParentChunk


# ---------------------------------------------------------------------------
# Parent chunk store (JSON sidecar)
# ---------------------------------------------------------------------------

class ParentChunkStore:
    """
    Stores parent chunk texts on disk (JSON) so retrieval can expand
    child hits to their full parent context.

    Format: { parent_chunk_id: { "text": str, "document": str,
                                  "page_number": int, "section_title": str } }
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._store: dict[str, dict] = {}
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                self._store = json.load(f)
            print(f"Parent store loaded: {len(self._store)} parents from {self._path.name}")

    def add_parents(self, parents: list[ParentChunk]) -> None:
        """Add parent chunks to the store and persist to disk."""
        for p in parents:
            self._store[p.chunk_id] = {
                "text": p.text,
                "document": p.document,
                "page_number": p.page_number,
                "section_title": p.section_title,
            }
        self._save()
        print(f"Parent store: {len(self._store)} parents saved to {self._path.name}")

    def get(self, parent_chunk_id: str) -> Optional[dict]:
        """Return parent chunk dict or None if not found."""
        return self._store.get(parent_chunk_id)

    def clear(self) -> None:
        self._store = {}
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._store, f)

    def __len__(self) -> int:
        return len(self._store)



# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class VectorSearchResult:
    """Single result from vector similarity search."""
    __slots__ = ("chunk_id", "document", "page_number", "section_title",
                 "text", "score", "chunk_index")

    def __init__(
        self,
        chunk_id: str,
        document: str,
        page_number: int,
        section_title: str,
        text: str,
        score: float,
        chunk_index: int,
    ) -> None:
        self.chunk_id = chunk_id
        self.document = document
        self.page_number = page_number
        self.section_title = section_title
        self.text = text
        self.score = score           # cosine similarity (0–1, higher = better)
        self.chunk_index = chunk_index

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "text": self.text,
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# ChromaDB wrapper
# ---------------------------------------------------------------------------

class LegalVectorStore:
    """
    Persistent ChromaDB-backed vector store for legal document chunks.

    The collection uses cosine distance (1 - cosine_similarity).
    Embeddings must be L2-normalised before insertion (BGEEmbedder
    already handles this via normalize_embeddings=True).

    Args:
        persist_directory: Path where ChromaDB stores its data.
        collection_name:   Name of the Chroma collection.
        reset:             If True, wipe the collection before use.
    """

    COLLECTION_NAME_DEFAULT = "legal_rag"

    def __init__(
        self,
        persist_directory: str | Path = "./chroma_db",
        collection_name: str = COLLECTION_NAME_DEFAULT,
        reset: bool = False,
    ) -> None:
        self._persist_dir = Path(persist_directory)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        if reset:
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection '{collection_name}' ready "
              f"({self._collection.count()} vectors stored)")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        batch_size: int = 500,
    ) -> None:
        """
        Add chunks + precomputed embeddings to the collection.

        Args:
            chunks:     List of DocumentChunk objects.
            embeddings: Float32 array (N, dim) aligned with chunks.
            batch_size: Chroma insert batch size.
        """
        assert len(chunks) == len(embeddings), (
            f"Chunk count {len(chunks)} != embedding count {len(embeddings)}"
        )

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "document": c.document,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "parent_chunk_id": getattr(c, "parent_chunk_id", ""),
                "source": c.metadata.get("source", ""),
            }
            for c in chunks
        ]

        # Insert in batches to avoid Chroma's gRPC message-size limit
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  Indexed chunks {start}–{end-1} / {total}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """
        Search for the top-K most similar chunks.

        Args:
            query_embedding: L2-normalised query vector.
            top_k:           Number of results to return.
            where:           Optional Chroma metadata filter
                             e.g. {"document": {"$eq": "nda_vendor_x.pdf"}}.

        Returns:
            List of VectorSearchResult sorted by descending similarity.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output: list[VectorSearchResult] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            similarity = 1.0 - dist    # cosine distance → similarity
            output.append(VectorSearchResult(
                chunk_id=chunk_id,
                document=meta.get("document", ""),
                page_number=int(meta.get("page_number", 0)),
                section_title=meta.get("section_title", ""),
                text=text,
                score=float(similarity),
                chunk_index=int(meta.get("chunk_index", 0)),
            ))

        output.sort(key=lambda r: r.score, reverse=True)
        return output

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[str]:
        """Return unique document names in the store."""
        results = self._collection.get(include=["metadatas"])
        docs = {m.get("document", "") for m in results["metadatas"]}
        return sorted(docs)

    def get_all_chunks_text(self) -> list[str]:
        """Return raw text of all chunks (for BM25 index building)."""
        results = self._collection.get(include=["documents", "metadatas"])
        # Note: "ids" is always returned by ChromaDB get() automatically —
        # passing it in include= raises ValueError in chromadb>=0.5
        return results["documents"], results["metadatas"], results["ids"]
