"""
ingest.py — CLI to ingest a directory of PDFs into the vector store.

Usage:
    python ingest.py --docs_dir ./my_pdfs
    python ingest.py --docs_dir ./my_pdfs --reset
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the Legal RAG vector store."
    )
    parser.add_argument(
        "--docs_dir", "-d",
        default="./my_pdfs",
        help="Directory containing PDF files (default: ./my_pdfs)",
    )
    parser.add_argument(
        "--persist_dir",
        default="./chroma_db",
        help="ChromaDB persistence directory (default: ./chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default="legal_rag",
        help="ChromaDB collection name (default: legal_rag)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing index and re-ingest from scratch",
    )
    args = parser.parse_args()

    # Validate docs directory
    docs_path = Path(args.docs_dir)
    if not docs_path.exists():
        print(f"[ERROR] Directory not found: {docs_path.resolve()}")
        print(f"  Create it and place your PDF files inside, then run again.")
        sys.exit(1)

    pdf_files = list(docs_path.rglob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDF files found in: {docs_path.resolve()}")
        sys.exit(1)

    print("=" * 60)
    print("  Legal RAG — Document Ingestion")
    print("=" * 60)
    print(f"  Source dir   : {docs_path.resolve()}")
    print(f"  PDFs found   : {len(pdf_files)}")
    print(f"  Persist dir  : {args.persist_dir}")
    print(f"  Collection   : {args.collection}")
    print(f"  Reset index  : {args.reset}")
    print()

    from pipeline import RAGPipeline

    pipeline = RAGPipeline.from_documents(
        docs_dir=args.docs_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        reset_store=args.reset,
        openai_api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder"),
    )

    print()
    print("=" * 60)
    print("  Ingestion Complete")
    print("=" * 60)
    print(f"  Total chunks indexed : {pipeline.index_size()}")
    print(f"  Documents            : {', '.join(pipeline.list_documents())}")
    print()
    print("Next step: python query.py --question \"Your question here\"")


if __name__ == "__main__":
    main()
