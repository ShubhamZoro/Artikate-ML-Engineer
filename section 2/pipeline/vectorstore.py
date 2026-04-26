"""
vectorstore.py — ChromaDB persistent vector store wrapper.

Two collections:
  legal_rag_children  — child chunk embeddings (256t) used for retrieval
  legal_rag_parents   — parent chunk texts (512t) used for generation, fetched by ID

Eliminates the JSON sidecar entirely. Parent chunks are stored in ChromaDB
and fetched on demand via .get(ids=[...]) — O(1) key-value lookup, no RAM
pre-load, no disk serialisation outside ChromaDB's own storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings

from .chunking import DocumentChunk, ParentChunk


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
        self.score = score
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
# ChromaDB wrapper — two collections
# ---------------------------------------------------------------------------

class LegalVectorStore:
    """
    Persistent ChromaDB-backed store with two collections:
      - <name>_children : child chunk embeddings for retrieval
      - <name>_parents  : parent chunk texts for generation (fetched by ID)

    Args:
        persist_directory: Path where ChromaDB stores its data.
        collection_name:   Base name; suffixes _children / _parents are appended.
        reset:             If True, wipe both collections before use.
    """

    def __init__(
        self,
        persist_directory: str | Path = "./chroma_db",
        collection_name: str = "legal_rag",
        reset: bool = False,
    ) -> None:
        self._persist_dir = Path(persist_directory)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._base_name = collection_name

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        child_name = f"{collection_name}_children"
        parent_name = f"{collection_name}_parents"

        if reset:
            for name in (child_name, parent_name):
                try:
                    self._client.delete_collection(name)
                except Exception:
                    pass

        self._children = self._client.get_or_create_collection(
            name=child_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Parents collection: no embeddings, text + metadata only.
        # We store a dummy embedding of dim=1 just to satisfy ChromaDB's
        # schema — we never query by vector on this collection.
        self._parents = self._client.get_or_create_collection(
            name=parent_name,
        )

        print(
            f"ChromaDB ready — "
            f"{self._children.count()} child vectors, "
            f"{self._parents.count()} parent chunks"
        )

    # ------------------------------------------------------------------
    # Indexing — children
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        batch_size: int = 500,
    ) -> None:
        """
        Add child chunks + precomputed embeddings to the children collection.
        """
        assert len(chunks) == len(embeddings)

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "document": c.document,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "parent_chunk_id": c.parent_chunk_id,
                "source": c.metadata.get("source", ""),
            }
            for c in chunks
        ]

        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._children.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  Indexed child chunks {start}–{end-1} / {total}")

    # ------------------------------------------------------------------
    # Indexing — parents
    # ------------------------------------------------------------------

    def add_parents(
        self,
        parents: list[ParentChunk],
        batch_size: int = 500,
    ) -> None:
        """
        Store parent chunks in the parents collection (text + metadata, no vectors).

        Uses upsert so re-ingestion of the same document is idempotent.
        """
        ids = [p.chunk_id for p in parents]
        texts = [p.text for p in parents]
        metadatas = [
            {
                "document": p.document,
                "page_number": p.page_number,
                "section_title": p.section_title,
                "chunk_index": p.chunk_index,
            }
            for p in parents
        ]

        total = len(parents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            # ChromaDB requires embeddings even for non-vector collections.
            # We pass a trivial [0.0] placeholder — this collection is never
            # searched by vector, only fetched by ID.
            dummy_embeddings = [[0.0]] * (end - start)
            self._parents.upsert(
                ids=ids[start:end],
                embeddings=dummy_embeddings,
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )
            print(f"  Stored parent chunks {start}–{end-1} / {total}")

        print(f"Parent store: {self._parents.count()} total parents in ChromaDB")

    # ------------------------------------------------------------------
    # Query — children (vector search)
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[VectorSearchResult]:
        """
        Search for top-K most similar child chunks by cosine similarity.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, self._children.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._children.query(**kwargs)

        output: list[VectorSearchResult] = []
        for chunk_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(VectorSearchResult(
                chunk_id=chunk_id,
                document=meta.get("document", ""),
                page_number=int(meta.get("page_number", 0)),
                section_title=meta.get("section_title", ""),
                text=text,
                score=float(1.0 - dist),   # cosine distance → similarity
                chunk_index=int(meta.get("chunk_index", 0)),
            ))

        output.sort(key=lambda r: r.score, reverse=True)
        return output

    # ------------------------------------------------------------------
    # Fetch — parents (by ID, batch)
    # ------------------------------------------------------------------

    def get_parents(self, parent_ids: list[str]) -> dict[str, dict]:
        """
        Fetch parent chunks by their IDs from the parents collection.

        Returns:
            Dict mapping parent_chunk_id → {"text": str, "document": str,
                                             "page_number": int, "section_title": str}
            Missing IDs are silently omitted (caller should fall back to child text).
        """
        if not parent_ids:
            return {}

        # Deduplicate — multiple children may share the same parent
        unique_ids = list(dict.fromkeys(parent_ids))

        try:
            results = self._parents.get(
                ids=unique_ids,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            print(f"  [ParentFetch] Warning: {exc}")
            return {}

        out: dict[str, dict] = {}
        for pid, text, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            out[pid] = {
                "text": text,
                "document": meta.get("document", ""),
                "page_number": int(meta.get("page_number", 0)),
                "section_title": meta.get("section_title", ""),
            }
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of indexed child chunks."""
        return self._children.count()

    def parent_count(self) -> int:
        """Number of stored parent chunks."""
        return self._parents.count()

    def list_documents(self) -> list[str]:
        """Unique document names in the child collection."""
        results = self._children.get(include=["metadatas"])
        docs = {m.get("document", "") for m in results["metadatas"]}
        return sorted(docs)

    def get_all_chunks_text(self) -> tuple[list[str], list[dict], list[str]]:
        """Return (texts, metadatas, ids) for all child chunks (used by BM25)."""
        results = self._children.get(include=["documents", "metadatas"])
        return results["documents"], results["metadatas"], results["ids"]