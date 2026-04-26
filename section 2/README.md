# Legal Document RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system for legal document Q&A, with hybrid retrieval, parent-chunk expansion, hallucination mitigation, and a measurable evaluation harness.

---

## Project Structure

```
section 2/
├── DESIGN.md                         # Architecture decisions & trade-offs
├── requirements.txt
├── .env.example                      # Copy to .env and add your API key
│
├── pipeline/
│   ├── ingestion.py                  # PDF loading (PyMuPDF)
│   ├── chunking.py                   # Legal-aware hierarchical chunking
│   ├── embeddings.py                 # OpenAI text-embedding-3-small wrapper
│   ├── vectorstore.py                # ChromaDB — two collections (children + parents)
│   ├── retrieval.py                  # Hybrid BM25 + dense + RRF + parent expansion
│   ├── generation.py                 # OpenAI GPT generation
│   ├── hallucination.py              # Grounding check + confidence score
│   └── rag_pipeline.py              # Main RAGPipeline class
│
├── evaluation/
│   ├── harness.py                    # Evaluation runner
│   ├── metrics.py                    # Precision@K
│   └── qa_pairs_template.json        # Fill this with your QA pairs
│
├── ingest.py                         # CLI: ingest PDFs
├── query.py                          # CLI: run a query
└── evaluate.py                       # CLI: run evaluation harness
```

---

## How the vector store works

The pipeline maintains **two ChromaDB collections** inside the same `chroma_db` directory:

| Collection | Contents | Used for |
|---|---|---|
| `legal_rag_children` | Child chunks (256 tokens) + embeddings | Vector similarity search at query time |
| `legal_rag_parents` | Parent chunks (512 tokens), no vectors | Fetched by ID after retrieval to give the LLM richer context |

At query time, the retriever finds the best child chunks (small = precise hits), then issues a **single batched `.get()` call** to fetch their parent texts (large = rich LLM context). This is O(1) per query regardless of corpus size — no JSON file, no RAM pre-load.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

### 3. Put your PDFs in a folder

```
./my_pdfs/
├── nda_vendor_x.pdf
├── service_agreement.pdf
└── policy_document.pdf
```

### 4. Ingest documents

```bash
python ingest.py --docs_dir ./my_pdfs
```

This builds both ChromaDB collections. On subsequent runs the index is reused automatically — re-ingestion is skipped unless you pass `--reset`.

### 5. Query the pipeline

```bash
python query.py --question "What is the notice period in the NDA with Vendor X?"
```

**Output format:**
```python
result = pipeline.query("What is the notice period in the NDA with Vendor X?")
# {
#   'answer': str,           # the generated answer
#   'sources': [             # list of source references
#     {
#       'document': str,     # filename
#       'page': int,         # page number
#       'chunk': str,        # retrieved text chunk (parent, 512t)
#     }
#   ],
#   'confidence': float,     # confidence score (0–1)
# }
```

### 6. Run evaluation

Fill in `evaluation/qa_pairs_template.json` with your QA pairs (see format below), then:

```bash
python evaluate.py --qa_file ./evaluation/qa_pairs_template.json --k 3
```

---

## QA Pair Format

Fill `evaluation/qa_pairs_template.json` with your own questions. Each entry:

```json
{
  "id": "q001",
  "question": "What is the notice period in the NDA with Vendor X?",
  "expected_answer": "30 days written notice",
  "relevant_document": "nda_vendor_x.pdf",
  "relevant_page": 3,
  "relevant_chunk_keywords": ["notice period", "thirty days", "written notice"]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique question ID |
| `question` | string | The question to ask |
| `expected_answer` | string | Human-readable expected answer (for reference) |
| `relevant_document` | string | PDF filename containing the answer |
| `relevant_page` | int | Page number (1-indexed) where the answer appears |
| `relevant_chunk_keywords` | list[str] | Key terms that should appear in the correct chunk |

The evaluation harness marks a retrieval as a **hit** if:
- The retrieved chunk's `document` matches `relevant_document`, **AND**
- The chunk's `page_number` matches `relevant_page` **OR** at least 2 of `relevant_chunk_keywords` appear in the chunk text.

---

## Python API

```python
from pipeline import RAGPipeline

# Build from documents (first run — ingests PDFs into both ChromaDB collections)
pipeline = RAGPipeline.from_documents(
    docs_dir="./my_pdfs",
    persist_dir="./chroma_db",
    reset_store=False,          # True to re-index from scratch
)

# Load existing index (subsequent runs — faster, no re-embedding)
pipeline = RAGPipeline.load(persist_dir="./chroma_db")

# Query
result = pipeline.query("What is the notice period in the NDA with Vendor X?")
print(result["answer"])
print(result["confidence"])
for src in result["sources"]:
    print(f"  → {src['document']} p.{src['page']}")

# Optional: filter to a specific document
result = pipeline.query(
    "What is the liability cap?",
    metadata_filter={"document": {"$eq": "service_agreement.pdf"}}
)

# Utility
pipeline.list_documents()   # ['nda_vendor_x.pdf', 'service_agreement.pdf', ...]
pipeline.index_size()       # 1842 (total indexed child chunks)
```

---

## Re-ingestion (adding new documents)

To add new PDFs without wiping the existing index:

```bash
python ingest.py --docs_dir ./new_pdfs
```

ChromaDB skips already-indexed chunks (by chunk ID). To force a full re-index of both collections:

```bash
python ingest.py --docs_dir ./my_pdfs --reset
```

---

## Architecture Summary

| Component | Choice | Reason |
|---|---|---|
| PDF parsing | PyMuPDF | Fastest, preserves text layout |
| Chunking | Legal-aware hierarchical | Respects clause boundaries |
| Embedding | `text-embedding-3-small` | Same API key, 1536-dim, cost-efficient |
| Vector store (children) | ChromaDB `legal_rag_children` | Cosine vector search, persistent |
| Parent store (parents) | ChromaDB `legal_rag_parents` | ID-based fetch, no RAM pre-load |
| Retrieval | Hybrid BM25 + dense + RRF | Exact-term + semantic precision |
| Parent expansion | Single batched `.get()` | One round-trip per query, O(1) |
| Generation | OpenAI GPT-4o-mini | Cost-efficient, instruction-following |
| Hallucination | Source grounding + confidence | Deterministic, model-agnostic |


---
