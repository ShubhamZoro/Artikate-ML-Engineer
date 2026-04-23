# DESIGN.md — Production-Grade RAG Pipeline for Legal Documents

## Problem Statement

We are building a question-answering system over a legal document corpus of 500+ PDF contracts and policy documents (averaging 40 pages each). Users ask precise, high-stakes questions like:

- *"What is the notice period in the NDA signed with Vendor X?"*
- *"Which contracts contain a limitation of liability clause above ₹1 crore?"*

Hallucinated answers are unacceptable — every answer must cite an exact source document and page.

---

## 1. Chunking Strategy

### Choice: Legal-Aware Hierarchical Chunking (512 tokens, 128-token overlap, section-boundary aware)

**Why not naive fixed-size chunking?**
Fixed-size chunking (e.g., every 500 characters) routinely splits legal clauses mid-sentence, destroying the semantic unit that answers the question. If a limitation-of-liability clause reads "...shall not exceed ₹1,00,00,000 (Rupees One Crore)..." and the amount lands in chunk N+1 while the clause header is in chunk N, no retriever will correctly surface both halves together.

**The approach:**
1. **Section detection first**: We parse the PDF and detect section headers using regex patterns common in legal drafting — `WHEREAS`, `NOW THEREFORE`, `Article [IVX\d]`, `Section \d+`, `Clause \d+`, numbered sub-clauses like `1.1`, `(a)`, `(i)`. Each section boundary resets the chunk window.
2. **Token-bounded sliding window within sections**: After respecting section boundaries, we apply a 512-token sliding window with 128-token overlap. This means multi-paragraph clauses are still split, but with generous overlap so the context from each side carries across chunk boundaries.
3. **Metadata tagging**: Every chunk carries `{filename, page_number, section_title, chunk_index}`. The page number is the page on which the chunk *starts*, matching what a lawyer would look up.

**Why 512 tokens?**
The BGE-large model's max sequence length is 512 tokens. Going larger either truncates content silently or requires a different (slower) model. 512 tokens covers ~380 words, which comfortably fits a 3–5 paragraph legal clause.

**Why 128-token overlap?**
Legal sentences often have antecedents that span paragraphs. "The aforementioned party" at the start of a clause may refer to a definition two paragraphs earlier. 128-token overlap (≈95 words) gives enough context for entity resolution without doubling storage.

---

## 2. Embedding Model

### Choice: `BAAI/bge-large-en-v1.5`

**Why not OpenAI `text-embedding-3-large`?**
| Criterion | BGE-large | text-embedding-3-large |
|---|---|---|
| Privacy | ✅ Runs locally, no data leaves your infra | ❌ Contract text sent to OpenAI API |
| Cost at scale | ✅ One-time GPU cost | ❌ $0.00013 / 1K tokens × millions of chunks |
| MTEB score | 64.6 (retrieval) | 62.3 (retrieval) |
| Control | ✅ Fine-tunable on domain data | ❌ Black box |

For a legal corpus containing sensitive contracts, **data sovereignty is non-negotiable**. BGE-large also outperforms OpenAI's model on retrieval benchmarks as of mid-2024.

**Why not a smaller model like MiniLM-L6?**
MiniLM-L6 is 5× faster but loses ~4 NDCG points on legal retrieval benchmarks. For a precision-critical domain where a wrong answer has legal consequences, the latency cost is worth it.

**Instruction prefix for queries:**
BGE-large supports an instruction prefix: `"Represent this sentence for searching relevant passages: "`. This is applied only to *queries*, not to indexed chunks, following the model's designed usage pattern. This materially improves retrieval recall for question-style inputs.

---

## 3. Vector Store

### Choice: ChromaDB (primary) + FAISS (evaluation harness)

**ChromaDB for production:**
- **Persistent**: Data survives restarts without re-embedding the entire corpus.
- **Metadata filtering**: Supports filtering by `document`, `page_range`, `date` at query time. Critical for queries like "only look in contracts from 2023".
- **Cosine similarity**: Natively supported, which is appropriate for normalized BGE embeddings.
- **Scalability**: Comfortably handles up to ~500K vectors locally. Our 500-doc corpus (≈40 pages × ~10 chunks/page = 200K chunks) fits well.

**Why not Pinecone?**
Pinecone is excellent but introduces (a) network latency per query, (b) cost at $0.096/1M reads, (c) data leaving your infrastructure. At 500-doc scale, ChromaDB is strictly better.

**Why not pure FAISS?**
FAISS has no native persistence — you must serialize/deserialize the index manually and there is no metadata query support. It is excellent for in-memory benchmarking (used here for the evaluation harness), but not for production.

**Why not Weaviate?**
Weaviate is excellent at scale (>1M vectors) but adds operational complexity (Docker, schema management). Overkill for 500 documents.

---

## 4. Retrieval Strategy

### Choice: Hybrid BM25 + Dense Retrieval → RRF Fusion → Cross-Encoder Re-ranking

**Stage 1 — Dense retrieval (ChromaDB, top-20 candidates):**
BGE-large embeddings capture semantic similarity. "What is the notice period?" retrieves chunks about termination timelines even if they say "written notification" rather than "notice period".

**Stage 2 — BM25 retrieval (rank_bm25, top-20 candidates):**
BM25 is essential for legal text because lawyers are precise with terminology. "limitation of liability above ₹1 crore" contains exact numeric and currency tokens that dense models may embed imprecisely. BM25 guarantees exact-term matching for vendor names, clause identifiers, and monetary amounts.

**Stage 3 — Reciprocal Rank Fusion (RRF):**
RRF formula: `score(d) = Σ 1/(k + rank(d))` where k=60. This merges both ranked lists without requiring score normalization. It is robust to the different score scales of BM25 vs. cosine similarity.

**Stage 4 — Cross-Encoder Re-ranking (top-20 → top-5):**
`cross-encoder/ms-marco-MiniLM-L-6-v2` reads the query and each candidate chunk *jointly*, producing a relevance score far more accurate than bi-encoder cosine similarity. Applied to only 20 candidates, this adds ~150ms latency but dramatically improves precision for the final top-3 results shown to the user.

**Why not naive top-K?**
Top-K dense-only retrieval has two failure modes for legal text:
1. Exact-term misses (BM25 handles this).
2. Bi-encoder imprecision — two chunks may have similar embeddings but very different legal relevance to a specific question (cross-encoder handles this).

---

## 5. Hallucination Mitigation

### Choice: Source Grounding Check + Confidence Scoring + Answer Refusal

**Strategy:**

1. **Source Grounding Check**: After the LLM generates an answer, we extract key noun phrases and named entities from the answer (using simple regex + noun-chunk detection). We then check whether each key phrase appears verbatim (or near-verbatim via token overlap) in at least one retrieved chunk. A `grounding_ratio` is computed: `matched_phrases / total_phrases`. If `grounding_ratio < 0.5`, the answer is flagged as potentially hallucinated.

2. **Confidence Score**: A composite score (0–1) computed as:
   - 40%: Maximum cross-encoder relevance score (how relevant is the best chunk?)
   - 40%: Source grounding ratio (how much of the answer is traceable to sources?)
   - 20%: Similarity score spread (low spread = retrieved chunks agree → higher confidence)

3. **Answer Refusal**: If `confidence < 0.3`, the pipeline returns a structured refusal:
   ```
   "I cannot find sufficient information in the provided documents to answer this question confidently."
   ```
   This is preferable to a hallucinated answer in a legal context.

**Why not perplexity-based scoring?**
Perplexity requires access to token log-probabilities, which is only available on some models (not GPT-4o via API). Our approach works with any black-box LLM.

**Why not a separate fact-checker LLM?**
A second LLM call doubles latency and cost, and introduces a new failure mode (the checker can also hallucinate). Source grounding via direct text matching is deterministic and auditable — critical for legal applications.

---

## 6. Prompt Design

The system prompt instructs the LLM to:
1. Answer *only* from the provided context.
2. If the answer is not in the context, say so explicitly.
3. Include exact document name and page number in the answer.
4. Use precise legal phrasing from the source text rather than paraphrasing.

This "context-anchored" prompting is the first line of defense against hallucination before the grounding check runs.

---

## 7. Scaling to 50,000 Documents

If the corpus grows from 500 to 50,000 documents (~50M chunks), every component becomes a bottleneck in a different way:

| Component | Bottleneck at 50k docs | Concrete Remedy |
|---|---|---|
| **PDF Ingestion** | Sequential PDF parsing takes hours | Parallelize with `multiprocessing.Pool` across CPUs; use async I/O for file reads |
| **Embedding** | BGE-large on CPU takes ~0.5s/chunk × 50M chunks = months | (a) GPU batch inference on A100 (1000 chunks/sec); (b) Switch to OpenAI `text-embedding-3-large` with rate-limit-aware async batching for burst capacity |
| **Vector Store** | ChromaDB degrades past ~2M vectors; no distributed sharding | Migrate to **Pinecone** (managed, distributed HNSW) or **Weaviate** (self-hosted, multi-tenancy, horizontal scaling). Use document-type namespacing. |
| **BM25** | `rank_bm25` holds the full inverted index in RAM (~10GB at 50k docs) | Replace with **Elasticsearch** or **OpenSearch** with BM25 native. Enables distributed sharding, partial updates, and filtered BM25. |
| **Re-ranking** | Cross-encoder running serially on single process saturates CPU | Deploy cross-encoder as a separate microservice behind a load balancer; use ONNX-quantized model for 3× speed; batch re-ranking requests |
| **LLM Generation** | Single OpenAI API call latency is stable, but cost scales with usage | Move to a self-hosted **Llama 3.1 70B** on vLLM for high-volume queries; keep GPT-4o for complex / low-frequency questions |
| **Metadata Filtering** | Full-corpus search without pre-filtering is O(N) | Build a document-level metadata index (SQL: document_type, vendor_name, date, jurisdiction) and pre-filter candidates before vector search |
| **Freshness** | Re-embedding the full corpus for each new document is wasteful | Implement incremental indexing: each new document is chunked and embedded in isolation and upserted into the vector store |

---

## 8. Evaluation

**Metric: Precision@3**

`Precision@3 = (# questions where correct chunk is in top-3 retrieved results) / total questions`

"Correct chunk" is defined as a chunk whose `(document, page)` metadata matches the `relevant_document` and `relevant_page` fields in the QA pair, OR whose text contains at least 2 of the `relevant_chunk_keywords`.

This metric directly measures the retrieval stage — the most critical component of the pipeline. If retrieval fails, no amount of LLM sophistication can recover the correct answer.

---

## Summary

| Decision | Choice | Key Reason |
|---|---|---|
| Chunking | Legal-aware hierarchical (512t, 128 overlap) | Preserves clause integrity |
| Embedding | BGE-large-en-v1.5 | Best retrieval MTEB, runs locally |
| Vector Store | ChromaDB | Persistent, metadata-filterable, right-sized |
| Retrieval | Hybrid BM25 + Dense + Cross-encoder | Handles exact-term + semantic + precision |
| Hallucination | Source grounding + confidence + refusal | Deterministic, auditable, model-agnostic |
