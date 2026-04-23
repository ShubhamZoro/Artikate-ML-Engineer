"""
query.py — CLI to run a single query against the RAG pipeline.

Usage:
    python query.py --question "What is the notice period in the NDA with Vendor X?"
    python query.py -q "Which contracts mention limitation of liability?" --json
"""

import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Query the Legal RAG pipeline."
    )
    parser.add_argument(
        "--question", "-q",
        required=True,
        help="The question to ask the pipeline",
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
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of chunks to use for generation (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print raw JSON output",
    )
    args = parser.parse_args()

    from pipeline import RAGPipeline

    print("=" * 60)
    print("  Legal RAG — Query")
    print("=" * 60)

    pipeline = RAGPipeline.load(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        llm_model=args.model,
        top_k=args.top_k,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    print(f"\nQuestion: {args.question}\n")
    result = pipeline.query(args.question)

    if args.json_output:
        print(json.dumps(result, indent=2))
        return

    # ── Answer ───────────────────────────────────────────────────────
    conf = result["confidence"]
    conf_label = "HIGH" if conf >= 0.6 else ("MEDIUM" if conf >= 0.3 else "LOW")
    print(f"Answer (confidence={conf:.3f} [{conf_label}]):")
    print("-" * 60)
    print(result["answer"])
    print()

    # ── Sources ──────────────────────────────────────────────────────
    sources = result.get("sources", [])
    if sources:
        print("Sources:")
        for i, src in enumerate(sources, 1):
            print(f"  [{i}] {src['document']}  |  Page {src['page']}")
            snippet = src.get("chunk", "")[:150]
            if snippet:
                print(f"      \"{snippet}{'...' if len(src.get('chunk','')) > 150 else ''}\"")
        print()


if __name__ == "__main__":
    main()
