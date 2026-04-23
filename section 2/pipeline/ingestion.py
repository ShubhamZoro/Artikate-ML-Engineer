"""
ingestion.py — PDF loading with per-page and per-chunk metadata.

Uses PyMuPDF (fitz) for fast, layout-preserving extraction.
Returns a list of Page objects with raw text + metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    """Raw content of a single PDF page."""
    document: str          # basename of the source file
    page_number: int       # 1-indexed
    text: str              # extracted text
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestedCorpus:
    """Collection of pages from all ingested documents."""
    pages: list[PageContent]

    def __len__(self) -> int:
        return len(self.pages)

    def iter_documents(self) -> Iterator[tuple[str, list[PageContent]]]:
        """Yield (document_name, pages) grouped by document."""
        docs: dict[str, list[PageContent]] = {}
        for page in self.pages:
            docs.setdefault(page.document, []).append(page)
        for doc_name, pages in docs.items():
            yield doc_name, pages


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace and remove non-printable characters."""
    text = re.sub(r'\x00', '', text)           # null bytes
    text = re.sub(r'\r\n|\r', '\n', text)      # normalise line endings
    text = re.sub(r'[ \t]{2,}', ' ', text)    # collapse repeated spaces/tabs
    text = re.sub(r'\n{3,}', '\n\n', text)    # max two consecutive blank lines
    return text.strip()


def load_pdf(path: str | Path) -> list[PageContent]:
    """
    Load a single PDF and return one PageContent per page.

    Args:
        path: Path to the PDF file.

    Returns:
        List of PageContent objects, one per page (1-indexed page numbers).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_name = path.name
    pages: list[PageContent] = []

    with fitz.open(str(path)) as pdf:
        total_pages = len(pdf)
        for page_idx in range(total_pages):
            page = pdf[page_idx]
            raw_text = page.get_text("text")   # plain text extraction
            cleaned = _clean_text(raw_text)

            if not cleaned:          # skip blank pages
                continue

            pages.append(PageContent(
                document=doc_name,
                page_number=page_idx + 1,        # 1-indexed
                text=cleaned,
                metadata={
                    "source": str(path),
                    "total_pages": total_pages,
                },
            ))

    return pages


def load_directory(directory: str | Path, glob: str = "*.pdf") -> IngestedCorpus:
    """
    Recursively load all PDFs in a directory.

    Args:
        directory: Path to search for PDFs.
        glob: Glob pattern for files (default: *.pdf).

    Returns:
        IngestedCorpus with all pages from all found PDFs.
    """
    directory = Path(directory)
    pdf_files = sorted(directory.rglob(glob))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {directory!r} matching '{glob}'")

    all_pages: list[PageContent] = []
    for pdf_path in pdf_files:
        try:
            pages = load_pdf(pdf_path)
            all_pages.extend(pages)
            print(f"  ✓ {pdf_path.name}: {len(pages)} pages loaded")
        except Exception as exc:
            print(f"  ✗ {pdf_path.name}: failed — {exc}")

    return IngestedCorpus(pages=all_pages)
