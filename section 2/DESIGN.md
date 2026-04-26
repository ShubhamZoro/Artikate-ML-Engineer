# DESIGN.md — Production-Grade RAG Pipeline for Legal Documents

## Problem Statement

We are building a question-answering system over a legal document corpus of 500+ PDF contracts and policy documents (averaging 40 pages each). Users ask precise, high-stakes questions like:

- *"What is the notice period in the NDA signed with Vendor X?"*
- *"Which contracts contain a limitation of liability clause above ₹1 crore?"*

Hallucinated answers are unacceptable — every answer must cite an exact source document and page.

---

## 1. Chunking Strategy

### Choice: Legal-Aware Hierarchical Chunking with Parent-Child Architecture

**Two chunk sizes are produced from every document:**

| Chunk type | Size | Overlap | Stored in | Purpose |
|---|---|---|---|---|
| Child chunk | 256 tokens | 32 tokens | ChromaDB `legal_rag_children` (vector-indexed) | Retrieval — small = precise hits |
| Parent chunk | 512 tokens | 64 tokens | ChromaDB `legal_rag_parents` (ID-indexed) | Generation — large = rich LLM context |

Every child chunk carries a `parent_chunk_id` pointer. After retrieval selects the best child chunks, a single batched `.get()` call fetches their parent texts. The LLM always reads 512-token context; retrieval precision comes from 256-token matching.

**Why not naive fixed-size chunking?**
Fixed-size chunking (e.g., every 500 characters) routinely splits legal clauses mid-sentence, destroying the semantic unit that answers the question. If a limitation-of-liability clause reads "...shall not exceed ₹1,00,00,000 (Rupees One Crore)..." and the amount lands in chunk N+1 while the clause header is in chunk N, no retriever will correctly surface both halves together.

**Section detection first:**
Before applying the sliding window, the pipeline detects section headers using regex patterns common in legal drafting — `WHEREAS`, `NOW THEREFORE`, `Article [IVX\d]`, `Section \d+`, `Clause \d+`, numbered sub-clauses like `1.1`, `(a)`, `(i)`. Each section boundary resets the chunk window, so a clause is never split at a heading boundary.

**Why 256 tokens for children?**
Smaller child chunks produce more precise page-level retrieval hits. A 256-token window covers ~190 words — enough to contain a single clause without pulling in neighbouring clauses that would dilute the similarity score.

**Why 512 tokens for parents?**
The parent provides the LLM with full clause context including antecedents ("the aforementioned party"), defined terms, and surrounding sub-clauses. 512 tokens covers ~380 words, comfortably fitting a 3–5 paragraph legal clause.

**Why 64-token overlap on parents, 32 on children?**
Legal sentences often have antecedents that span paragraphs. Overlap ensures continuity at chunk boundaries without doubling storage.

**Metadata tagging:**
Every chunk carries `{document, page_number, section_title, chunk_index}`. The page number is the page on which the chunk starts, matching what a lawyer would look up.

---

## 2. Embedding Model

### Choice: `text-embedding-3-small` (OpenAI)

| Criterion | text-embedding-3-small | BGE-large-en-v1.5 |
|---|---|---|
| Dimensions | 1536 | 1024 |
| Infrastructure | ✅ Same API key as generation | Requires local GPU or separate service |
| Cost | $0.02 / 1M tokens | One-time GPU cost |
| Maintenance | ✅ Zero — managed by OpenAI | Model download (~1.3 GB), version pinning |
| Fine-tunability | Limited | ✅ Full control |
| Data sovereignty | ❌ Text leaves your infra | ✅ Runs locally |

For the current 500-document scale, operational simplicity outweighs the sovereignty concern — both the embedding and generation calls already use the OpenAI API, so adding a local embedding service would introduce infrastructure complexity without a meaningful accuracy gain. At 50k documents or in a regulated environment where contracts must not leave the network, switching to a locally-hosted model (BGE-large or a quantised equivalent) is the correct move.

**Why not `text-embedding-3-large`?**
3-large is 3× the cost and marginally more accurate. For legal retrieval the precision difference does not justify the cost at scale.

**Instruction prefix:**
Not used — OpenAI's embedding models do not use instruction prefixes. The query is embedded as-is.

---

## 3. Vector Store

### Choice: ChromaDB with two collections

```
chroma_db/
  legal_rag_children   ← child chunk embeddings + metadata (vector-indexed, cosine)
  legal_rag_parents    ← parent chunk texts + metadata (ID-indexed, no vectors)
```

**Why two collections instead of one?**

The naive approach — storing only child chunks in ChromaDB and parent texts in a JSON sidecar file — has a critical flaw: the sidecar is fully loaded into RAM on every pipeline startup via `json.load()`. At 500 docs this is acceptable (~200k entries). At 50k docs the sidecar becomes a multi-GB file with a 30+ second deserialisation penalty and permanent process-memory occupation, even when only 3–5 parents are needed per query.

The dual-collection design eliminates both problems:

- **No RAM pre-load.** ChromaDB manages its own storage. Parent texts are fetched on demand.
- **Single round-trip per query.** After the top-K child IDs are selected, all their `parent_chunk_id` values are passed to `parents.get(ids=[...])` in one batched call — not one lookup per chunk.
- **Same O(1) access pattern.** ChromaDB's `.get()` by ID is a primary-key lookup, identical in complexity to a Python dict but without the dict living in RAM.
- **Persistence guarantees.** ChromaDB handles crash recovery and durability for both collections — no separate file management.

**Why not Pinecone?**
Pinecone is excellent but introduces network latency per query, cost at $0.096/1M reads, and data leaving your infrastructure. At 500-doc scale ChromaDB is strictly better.

**Why not pure FAISS?**
FAISS has no native persistence and no metadata query support. It is useful for in-memory benchmarking (used in the evaluation harness) but not for production.

**Why not Weaviate?**
Weaviate is excellent at scale (>1M vectors) but adds operational complexity (Docker, schema management). Overkill for 500 documents.

**Cosine similarity:**
Both collections use `hnsw:space: cosine`. This is appropriate for L2-normalised OpenAI embeddings — dot product on normalised vectors equals cosine similarity.

---

## 4. Retrieval Strategy

### Choice: Hybrid BM25 + Dense Retrieval → RRF Fusion → Child-to-Parent Expansion

**Stage 1 — Dense retrieval (ChromaDB children, top-30 candidates):**
`text-embedding-3-small` embeddings capture semantic similarity. "What is the notice period?" retrieves chunks about "termination timelines" even if they say "written notification" rather than "notice period".

**Stage 2 — BM25 retrieval (rank_bm25, top-30 candidates):**
BM25 is essential for legal text because lawyers are precise with terminology. "limitation of liability above ₹1 crore" contains exact numeric and currency tokens that dense models may embed imprecisely. BM25 guarantees exact-term matching for vendor names, clause identifiers, and monetary amounts.

BM25 queries are also expanded with document-name tokens when the query explicitly references a document (e.g. "shuttle service agreement" → appends "SampleContract Shuttle"), strongly boosting chunks from that document.

**Stage 3 — Reciprocal Rank Fusion (RRF):**
RRF formula: `score(d) = Σ 1/(k + rank(d))` where k=60. This merges both ranked lists without requiring score normalisation. It is robust to the different score scales of BM25 vs. cosine similarity.

RRF replaces the cross-encoder re-ranker from earlier iterations. On candidate sets of 30 or fewer documents, RRF achieves comparable precision to a cross-encoder with zero additional latency and no model dependency.

**Stage 4 — Page-diversity deduplication:**
After RRF, results are deduplicated by `(document, page)` pairs before selecting the final top-K. This prevents all retrieval slots from being occupied by different child chunks from the same (potentially wrong) page.

**Stage 5 — Child-to-parent expansion (one batched fetch):**
The final top-K child IDs are used to look up their `parent_chunk_id` values. All parent IDs are passed to `vector_store.get_parents(ids)` in a **single** ChromaDB `.get()` call. The returned 512-token parent texts are substituted into the `RetrievedChunk` objects before they are passed to the generator. Citations (document, page) still come from the child metadata — so source attribution remains accurate at the page level.

**Why not naive top-K?**
Top-K dense-only retrieval has two failure modes for legal text: exact-term misses (BM25 handles this) and bi-encoder imprecision where two chunks have similar embeddings but very different legal relevance to a specific question (page-diversity dedup and parent expansion partially address this).

---

## 5. Hallucination Mitigation

### Choice: Source Grounding Check + Confidence Scoring + Answer Refusal

**Strategy:**

1. **Source Grounding Check:** After the LLM generates an answer, key noun phrases and named entities are extracted from the answer using regex (monetary amounts, quoted strings, capitalised proper nouns, numbers with units). Each phrase is checked against retrieved chunks. A `grounding_ratio` is computed: `matched_phrases / total_phrases`. If `grounding_ratio < 0.5`, the answer is flagged as potentially hallucinated.

2. **Confidence Score:** A composite score (0–1) computed as:
   - 40%: Source grounding ratio (how much of the answer is traceable to sources?)
   - 40%: Sigmoid of top RRF score (how relevant is the best retrieved chunk?)
   - 20%: Inverse score spread (low spread = retrieved chunks agree → higher confidence)

3. **Answer Refusal:** If `confidence < 0.3`, the pipeline returns a structured refusal:
   ```
   "I cannot find sufficient information in the provided documents to answer this question confidently."
   ```
   This is preferable to a hallucinated answer in a legal context.

**Why not perplexity-based scoring?**
Perplexity requires access to token log-probabilities, which is not available through the OpenAI chat completions API. The grounding approach works with any black-box LLM.

**Why not a separate fact-checker LLM?**
A second LLM call doubles latency and cost, and introduces a new failure mode (the checker can also hallucinate). Source grounding via direct text matching is deterministic and auditable — critical for legal applications.

---

## 6. Prompt Design

The system prompt instructs the LLM to:
1. Answer *only* from the provided context excerpts. Do not use any external knowledge.
2. If the answer is not in the context, say so explicitly.
3. Quote relevant clause text verbatim where possible, using double quotes.
4. Always end the answer with `[Source: <document name>, Page <number>]`.
5. Be concise. Do not pad the answer.
6. Never fabricate names, dates, amounts, or clause numbers.

This "context-anchored" prompting is the first line of defence against hallucination, before the grounding check runs.

---

## 7. Scaling to 50,000 Documents

If the corpus grows from 500 to 50,000 documents (~50M chunks), every component becomes a bottleneck in a different way:

| Component | Bottleneck at 50k docs | Concrete Remedy |
|---|---|---|
| **PDF Ingestion** | Sequential PDF parsing takes hours | Parallelize with `multiprocessing.Pool`; async I/O for file reads |
| **Embedding** | API rate limits at burst scale | Rate-limit-aware async batching; `text-embedding-3-large` for accuracy if budget allows |
| **Vector store (children)** | ChromaDB degrades past ~2M vectors | Migrate to **Pinecone** (managed HNSW) or **Weaviate** (self-hosted, horizontal scaling) |
| **Vector store (parents)** | ChromaDB `.get()` remains fast but single-node | Same migration — parent collection moves alongside children; access pattern unchanged |
| **BM25** | `rank_bm25` holds full inverted index in RAM (~10 GB at 50k docs) | Replace with **Elasticsearch** or **OpenSearch** — distributed sharding, partial updates, filtered BM25 |
| **LLM Generation** | Cost scales linearly with usage | Self-hosted **Llama 3.1 70B** on vLLM for high-volume queries; keep GPT-4o-mini for complex questions |
| **Metadata Filtering** | Full-corpus search without pre-filtering is O(N) | Build a document-level metadata index (SQL: `document_type`, `vendor_name`, `date`, `jurisdiction`) and pre-filter before vector search |
| **Freshness** | Re-embedding the full corpus for each new document is wasteful | Incremental indexing: each new document is chunked and embedded in isolation and upserted into both ChromaDB collections |
| **Data sovereignty** | OpenAI embeddings send contract text to external API | Switch to locally-hosted **BGE-large-en-v1.5** or **E5-large** on a private GPU cluster |

---

## 8. Evaluation

**Metric: Precision@3**

`Precision@3 = (# questions where correct chunk is in top-3 retrieved results) / total questions`

A retrieval is a **hit** if the retrieved chunk's `(document, page)` metadata matches the `relevant_document` and `relevant_page` fields in the QA pair, **OR** its text contains at least 2 of the `relevant_chunk_keywords`. The dual criterion handles text that spans page boundaries.

The evaluation runs against child chunk retrieval only — not generation — because retrieval is the failure mode that can be measured automatically and deterministically. Generation quality requires human evaluation of free-text answers.

---

## Summary

| Decision | Choice | Key Reason |
|---|---|---|
| Chunking | Legal-aware hierarchical, parent-child (256t / 512t) | Precise retrieval + rich generation context |
| Embedding | `text-embedding-3-small` (OpenAI) | Same API, zero infra overhead, cost-efficient |
| Vector store — children | ChromaDB `legal_rag_children` | Cosine vector search, persistent, metadata-filterable |
| Vector store — parents | ChromaDB `legal_rag_parents` | On-demand ID fetch, no RAM pre-load, no JSON sidecar |
| Retrieval | Hybrid BM25 + dense + RRF + page dedup | Exact-term + semantic + diversity |
| Parent expansion | Single batched `.get()` per query | One round-trip regardless of K |
| Hallucination | Source grounding + confidence + refusal | Deterministic, auditable, model-agnostic |