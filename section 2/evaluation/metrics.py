"""
metrics.py — Evaluation metrics for the RAG pipeline.

Implements Precision@K: fraction of questions where the correct chunk
appears in the top-K retrieved results.
"""

from __future__ import annotations

from typing import Callable


def _is_correct_chunk(
    retrieved_chunk: dict,
    qa_pair: dict,
) -> bool:
    """
    Check if a single retrieved chunk matches the expected answer location.

    A chunk is considered correct if ALL of the following hold:
      1. Its document filename matches qa_pair['relevant_document'] (exact or
         substring match — handles path vs basename differences).
      2. Its page number matches qa_pair['relevant_page'].
         OR at least 2 of qa_pair['relevant_chunk_keywords'] appear in the chunk text.

    This dual criterion handles edge cases where text spans page boundaries.
    """
    doc_match = (
        qa_pair["relevant_document"].lower() in retrieved_chunk["document"].lower()
        or retrieved_chunk["document"].lower() in qa_pair["relevant_document"].lower()
    )
    page_match = retrieved_chunk.get("page_number", retrieved_chunk.get("page", 0)) == qa_pair["relevant_page"]

    keywords = qa_pair.get("relevant_chunk_keywords", [])
    chunk_text_lower = retrieved_chunk.get("text", retrieved_chunk.get("chunk", "")).lower()
    keyword_match = (
        sum(1 for kw in keywords if kw.lower() in chunk_text_lower) >= min(2, len(keywords))
        if keywords else False
    )

    return doc_match and (page_match or keyword_match)


def precision_at_k(
    retrieved_results: list[list[dict]],
    qa_pairs: list[dict],
    k: int = 3,
) -> dict:
    """
    Compute Precision@K for a set of questions.

    Args:
        retrieved_results: For each question, a list of retrieved chunk dicts
                           (each with 'document', 'page_number'/'page', 'text'/'chunk').
                           The list should be ordered best-first.
        qa_pairs:          List of QA pair dicts (see qa_pairs_template.json).
        k:                 The K in Precision@K.

    Returns:
        Dict with:
          - precision_at_k (float): overall score
          - k (int): the K used
          - per_question (list): per-question breakdown
          - hits (int): number of questions with correct chunk in top-K
          - total (int): total questions
    """
    assert len(retrieved_results) == len(qa_pairs), (
        f"Mismatch: {len(retrieved_results)} result sets vs {len(qa_pairs)} QA pairs"
    )

    per_question = []
    hits = 0

    for i, (results, qa) in enumerate(zip(retrieved_results, qa_pairs)):
        top_k_results = results[:k]
        correct = any(_is_correct_chunk(r, qa) for r in top_k_results)
        if correct:
            hits += 1

        per_question.append({
            "id": qa.get("id", f"q{i+1:03d}"),
            "question": qa["question"],
            "hit": correct,
            "top_k_docs": [(r.get("document", ""), r.get("page_number", r.get("page", 0)))
                           for r in top_k_results],
            "expected_doc": qa["relevant_document"],
            "expected_page": qa["relevant_page"],
        })

    total = len(qa_pairs)
    score = hits / total if total > 0 else 0.0

    return {
        "precision_at_k": round(score, 4),
        "k": k,
        "hits": hits,
        "total": total,
        "per_question": per_question,
    }
