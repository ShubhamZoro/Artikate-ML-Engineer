"""
chunking.py — Legal-aware hierarchical chunking with parent-child architecture.

Strategy:
  Parent chunks (512 tokens, 64 overlap)  → stored for LLM generation context
  Child chunks  (256 tokens, 32 overlap)  → indexed in ChromaDB for retrieval

Each child chunk references its parent via parent_chunk_id.
Retrieval uses small child chunks (precise page hits).
Generation uses large parent chunks (rich context for the LLM).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .ingestion import IngestedCorpus, PageContent


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ParentChunk:
    """Large chunk stored for LLM generation. NOT indexed in the vector store."""
    chunk_id: str             # "<doc>_parent_p<page>_c<idx>"
    document: str
    page_number: int
    section_title: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Small child chunk indexed in ChromaDB for retrieval."""
    chunk_id: str             # "<doc>_child_p<page>_c<idx>"
    parent_chunk_id: str      # → ParentChunk.chunk_id
    document: str
    page_number: int
    section_title: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Legal section detection
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r'^(?:ARTICLE|Article)\s+(?:[IVXivx\d]+)[\.\s]', re.MULTILINE),
    re.compile(r'^(?:SECTION|Section)\s+\d+(?:\.\d+)*[\.\s]', re.MULTILINE),
    re.compile(r'^\d+(?:\.\d+)*\s+[A-Z][A-Za-z\s]{3,}$', re.MULTILINE),
    re.compile(r'^(?:WHEREAS|NOW[,\s]+THEREFORE|IN WITNESS WHEREOF)', re.MULTILINE),
    re.compile(r'^(?:SCHEDULE|EXHIBIT|ANNEXURE|APPENDIX)\s+[A-Z\d]', re.MULTILINE),
    re.compile(r'^[A-Z][A-Z\s]{10,}$', re.MULTILINE),
]


def _detect_sections(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.group().strip()))
    hits.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    last_end = -1
    for offset, title in hits:
        if offset >= last_end:
            deduped.append((offset, title))
            last_end = offset + len(title)
    return deduped


# ---------------------------------------------------------------------------
# Token-aware sliding window
# ---------------------------------------------------------------------------

def _naive_tokenize(text: str) -> list[str]:
    return text.split()


def _sliding_window(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = _naive_tokenize(text)
    if not tokens:
        return []
    chunks: list[str] = []
    stride = max(1, max_tokens - overlap_tokens)
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += stride
    return chunks


# ---------------------------------------------------------------------------
# Main hierarchical chunking function
# ---------------------------------------------------------------------------

def chunk_corpus(
    corpus: IngestedCorpus,
    parent_max_tokens: int = 512,
    parent_overlap_tokens: int = 64,
    child_max_tokens: int = 256,
    child_overlap_tokens: int = 32,
    min_tokens: int = 20,
) -> tuple[list[ParentChunk], list[DocumentChunk]]:
    """
    Hierarchical chunking: produces parent chunks (for generation) and
    child chunks (for retrieval), with each child referencing its parent.

    Args:
        corpus:                 Ingested pages.
        parent_max_tokens:      Tokens per parent chunk (default 512).
        parent_overlap_tokens:  Overlap between parent chunks (default 64).
        child_max_tokens:       Tokens per child chunk (default 256).
        child_overlap_tokens:   Overlap between child chunks (default 32).
        min_tokens:             Discard chunks shorter than this.

    Returns:
        (parent_chunks, child_chunks) — both lists, aligned by parent_chunk_id.
    """
    all_parents: list[ParentChunk] = []
    all_children: list[DocumentChunk] = []

    for doc_name, pages in corpus.iter_documents():
        parent_index = 0
        child_index = 0

        for page in pages:
            page_text = page.text
            if not page_text.strip():
                continue

            section_hits = _detect_sections(page_text)
            if not section_hits:
                section_hits = [(0, "")]

            # Build section segments
            segments: list[tuple[str, str]] = []
            for i, (offset, title) in enumerate(section_hits):
                seg_start = offset
                seg_end = (
                    section_hits[i + 1][0]
                    if i + 1 < len(section_hits)
                    else len(page_text)
                )
                seg_text = page_text[seg_start:seg_end].strip()
                if seg_text:
                    segments.append((title, seg_text))

            for section_title, seg_text in segments:
                # ── Parent chunks ─────────────────────────────────────
                parent_windows = _sliding_window(
                    seg_text, parent_max_tokens, parent_overlap_tokens
                )
                for parent_text in parent_windows:
                    if len(_naive_tokenize(parent_text)) < min_tokens:
                        continue

                    parent_id = f"{doc_name}_parent_p{page.page_number}_c{parent_index}"
                    parent = ParentChunk(
                        chunk_id=parent_id,
                        document=doc_name,
                        page_number=page.page_number,
                        section_title=section_title,
                        text=parent_text,
                        chunk_index=parent_index,
                        metadata={"source": page.metadata.get("source", "")},
                    )
                    all_parents.append(parent)
                    parent_index += 1

                    # ── Child chunks (subdivide the parent) ───────────
                    child_windows = _sliding_window(
                        parent_text, child_max_tokens, child_overlap_tokens
                    )
                    for child_text in child_windows:
                        if len(_naive_tokenize(child_text)) < min_tokens:
                            continue

                        child_id = (
                            f"{doc_name}_child_p{page.page_number}_c{child_index}"
                        )
                        all_children.append(DocumentChunk(
                            chunk_id=child_id,
                            parent_chunk_id=parent_id,
                            document=doc_name,
                            page_number=page.page_number,
                            section_title=section_title,
                            text=child_text,
                            chunk_index=child_index,
                            metadata={"source": page.metadata.get("source", "")},
                        ))
                        child_index += 1

    return all_parents, all_children
