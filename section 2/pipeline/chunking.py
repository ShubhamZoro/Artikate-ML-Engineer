"""
chunking.py — Legal-aware hierarchical chunking.

Strategy:
1. Detect section boundaries using legal-document regex patterns.
2. Within each section, apply a sliding-window token-based split
   (512 tokens, 128-token overlap).
3. Tag every chunk with: document, page_number, section_title, chunk_index.
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
class DocumentChunk:
    """A single retrievable unit of text from a legal document."""
    chunk_id: str             # unique identifier  "<doc>_p<page>_c<idx>"
    document: str             # source filename
    page_number: int          # page where chunk starts (1-indexed)
    section_title: str        # nearest detected section heading (or "")
    text: str                 # chunk text
    chunk_index: int          # sequential index within the document
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Legal section detection
# ---------------------------------------------------------------------------

# Patterns common in legal contracts and policy documents (English)
_SECTION_PATTERNS: list[re.Pattern] = [
    # "ARTICLE I", "ARTICLE IV", "Article 1."
    re.compile(r'^(?:ARTICLE|Article)\s+(?:[IVXivx\d]+)[\.\s]', re.MULTILINE),
    # "SECTION 1.", "Section 2.1"
    re.compile(r'^(?:SECTION|Section)\s+\d+(?:\.\d+)*[\.\s]', re.MULTILINE),
    # "1. DEFINITIONS", "2.1 Term"
    re.compile(r'^\d+(?:\.\d+)*\s+[A-Z][A-Za-z\s]{3,}$', re.MULTILINE),
    # "WHEREAS", "NOW, THEREFORE", "IN WITNESS WHEREOF"
    re.compile(r'^(?:WHEREAS|NOW[,\s]+THEREFORE|IN WITNESS WHEREOF)', re.MULTILINE),
    # "SCHEDULE A", "EXHIBIT 1", "ANNEXURE B"
    re.compile(r'^(?:SCHEDULE|EXHIBIT|ANNEXURE|APPENDIX)\s+[A-Z\d]', re.MULTILINE),
    # ALL-CAPS headings of 3+ words at start of line
    re.compile(r'^[A-Z][A-Z\s]{10,}$', re.MULTILINE),
]


def _detect_sections(text: str) -> list[tuple[int, str]]:
    """
    Return list of (char_offset, section_title) sorted by position.
    """
    hits: list[tuple[int, str]] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.group().strip()))

    # Sort and de-duplicate overlapping matches
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
    """
    Split text into whitespace tokens.
    (No ML tokenizer dependency at chunk time — fast and deterministic.)
    """
    return text.split()


def _sliding_window(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 128,
) -> list[str]:
    """
    Split text into overlapping windows of `max_tokens` words.
    """
    tokens = _naive_tokenize(text)
    if not tokens:
        return []

    chunks: list[str] = []
    stride = max_tokens - overlap_tokens
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end == len(tokens):
            break
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk_corpus(
    corpus: IngestedCorpus,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    min_chunk_tokens: int = 20,
) -> list[DocumentChunk]:
    """
    Convert an IngestedCorpus into a list of DocumentChunks using
    legal-aware hierarchical chunking.

    Args:
        corpus:           Ingested pages.
        max_tokens:       Maximum tokens per chunk (default 512).
        overlap_tokens:   Token overlap between consecutive chunks (default 128).
        min_chunk_tokens: Discard chunks shorter than this (likely artefacts).

    Returns:
        Flat list of DocumentChunk objects ready for embedding.
    """
    all_chunks: list[DocumentChunk] = []

    for doc_name, pages in corpus.iter_documents():
        chunk_index = 0

        for page in pages:
            page_text = page.text
            if not page_text.strip():
                continue

            # Detect section boundaries within this page's text
            section_hits = _detect_sections(page_text)

            if not section_hits:
                # No sections detected — treat whole page as one unit
                section_hits = [(0, "")]

            # Build section segments: text between consecutive section hits
            segments: list[tuple[str, str]] = []   # (section_title, segment_text)
            for i, (offset, title) in enumerate(section_hits):
                seg_start = offset
                seg_end = section_hits[i + 1][0] if i + 1 < len(section_hits) else len(page_text)
                seg_text = page_text[seg_start:seg_end].strip()
                if seg_text:
                    segments.append((title, seg_text))

            # Apply sliding window within each segment
            for section_title, seg_text in segments:
                windows = _sliding_window(seg_text, max_tokens, overlap_tokens)
                for window_text in windows:
                    if len(_naive_tokenize(window_text)) < min_chunk_tokens:
                        continue   # too short, likely a heading artefact

                    chunk_id = f"{doc_name}_p{page.page_number}_c{chunk_index}"
                    all_chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        document=doc_name,
                        page_number=page.page_number,
                        section_title=section_title,
                        text=window_text,
                        chunk_index=chunk_index,
                        metadata={
                            "source": page.metadata.get("source", ""),
                        },
                    ))
                    chunk_index += 1

    return all_chunks
