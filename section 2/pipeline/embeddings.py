"""
embeddings.py — OpenAI text-embedding-3-small wrapper.

Replaces the sentence-transformers/BGE approach with the OpenAI
Embeddings API. Benefits for this project:
  - No local model download (~1.3 GB saved)
  - Same API key already used for generation
  - text-embedding-3-small: 1536-dim, fast, cost-efficient ($0.02/1M tokens)
  - text-embedding-3-large available for higher accuracy if needed

Rate-limit handling: batches chunks in groups of 500 (API max is 2048
inputs per request, but 500 is safer for token-length variance).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from openai import OpenAI

from .chunking import DocumentChunk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "text-embedding-3-small"   # 1536-dim, cost-efficient
BATCH_SIZE    = 500                         # chunks per API call
MAX_RETRIES   = 3
RETRY_DELAY   = 2.0                         # seconds between retries


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------

class OpenAIEmbedder:
    """
    OpenAI Embeddings API wrapper for both corpus indexing and query embedding.

    Args:
        model:     OpenAI embedding model (default: text-embedding-3-small).
        api_key:   OpenAI API key — reads OPENAI_API_KEY env var if None.
        show_progress: Print batch progress during corpus embedding.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        show_progress: bool = True,
    ) -> None:
        self.model = model
        self.show_progress = show_progress
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        # Probe the model to confirm connectivity + get dimension
        probe = self._embed_batch(["test"])
        self._embedding_dim = len(probe[0])
        print(f"OpenAI embedder ready: model={model}, dim={self._embedding_dim}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI API for a single batch, with retry logic."""
        # Strip newlines — OpenAI recommends this for best quality
        cleaned = [t.replace("\n", " ") for t in texts]

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=cleaned,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    print(f"  [Embedding] Retry {attempt+1}/{MAX_RETRIES} after error: {exc}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise

    def _normalise(self, vecs: list[list[float]]) -> np.ndarray:
        """L2-normalise vectors for cosine similarity via dot product."""
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    # ------------------------------------------------------------------
    # Corpus embedding (used during ingestion)
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """
        Embed a list of DocumentChunks for indexing.

        Processes in batches of BATCH_SIZE to respect API limits.

        Returns:
            Float32 array of shape (len(chunks), embedding_dim), L2-normalised.
        """
        texts = [c.text for c in chunks]
        total = len(texts)
        all_embeddings: list[list[float]] = []

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch = texts[start:end]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
            if self.show_progress:
                print(f"  Embedded chunks {end}/{total} ({end/total*100:.0f}%)")

        return self._normalise(all_embeddings)

    # ------------------------------------------------------------------
    # Query embedding (used during retrieval)
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            Float32 array of shape (embedding_dim,), L2-normalised.
        """
        embeddings = self._embed_batch([query])
        return self._normalise(embeddings)[0]

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Batch-embed multiple query strings."""
        all_embeddings: list[list[float]] = []
        for start in range(0, len(queries), BATCH_SIZE):
            batch = queries[start:start + BATCH_SIZE]
            all_embeddings.extend(self._embed_batch(batch))
        return self._normalise(all_embeddings)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
