"""
evaluate.py — CLI to run the evaluation harness and report Precision@K.

Usage:
    python evaluate.py
    python evaluate.py --qa_file ./evaluation/qa_pairs_template.json --k 3
    python evaluate.py --save_report ./evaluation/report.json
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Run the Legal RAG evaluation harness (Precision@K)."
    )
    parser.add_argument(
        "--qa_file", "-q",
        default="./evaluation/qa_pairs_template.json",
        help="Path to QA pairs JSON file (default: ./evaluation/qa_pairs_template.json)",
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
        "--k",
        type=int,
        default=3,
        help="K for Precision@K metric (default: 3)",
    )
    parser.add_argument(
        "--save_report",
        default="",
        help="Optional path to save JSON evaluation report",
    )
    args = parser.parse_args()

    from pipeline import RAGPipeline
    from evaluation import EvaluationHarness

    print("=" * 60)
    print("  Legal RAG — Evaluation Harness")
    print("=" * 60)
    print(f"  QA file    : {args.qa_file}")
    print(f"  Metric     : Precision@{args.k}")
    print()

    pipeline = RAGPipeline.load(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        llm_model=args.model,
        top_k=args.k,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    harness = EvaluationHarness(pipeline=pipeline, qa_path=args.qa_file)
    metrics = harness.run(k=args.k)

    if args.save_report:
        harness.save_report(metrics, args.save_report)

    print()
    print("=" * 60)
    print(f"  FINAL SCORE: Precision@{args.k} = {metrics['precision_at_k']:.4f}")
    print(f"  ({metrics['hits']}/{metrics['total']} questions retrieved correctly)")
    print("=" * 60)


if __name__ == "__main__":
    main()
