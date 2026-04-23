"""
hallucination.py — Source grounding check and confidence scoring.

Three-layer hallucination mitigation:
  1. Source Grounding Check: verify key noun phrases from the answer
     exist in at least one retrieved chunk.
  2. Confidence Score: weighted combination of retrieval quality signals.
  3. Answer Refusal: return a structured refusal when confidence < threshold.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Optional

from .retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFUSAL_MESSAGE = (
    "I cannot find sufficient information in the provided documents "
    "to answer this question confidently. Please consult the source documents directly."
)

CONFIDENCE_REFUSAL_THRESHOLD = 0.3    # below this → refuse to answer
GROUNDING_WEIGHT       = 0.40         # 40% of confidence from grounding ratio
RERANK_WEIGHT          = 0.40         # 40% from top cross-encoder score
SPREAD_WEIGHT          = 0.20         # 20% from score spread (agreement)


# ---------------------------------------------------------------------------
# Noun-phrase extraction (no spaCy dependency — regex-based)
# ---------------------------------------------------------------------------

def _extract_key_phrases(text: str) -> list[str]:
    """
    Extract key noun phrases and named entities using regex heuristics.
    Targets: capitalised terms, numbers, monetary amounts, quoted strings,
    legal clause terms.
    """
    phrases: list[str] = []

    # Monetary amounts (₹1 crore, $500,000, Rs. 10 lakhs …)
    phrases += re.findall(
        r'(?:₹|Rs\.?|INR|USD|\$)\s*[\d,]+(?:\.\d+)?\s*(?:crore|lakh|thousand|million|billion)?',
        text, flags=re.IGNORECASE
    )

    # Quoted strings
    phrases += re.findall(r'"([^"]{3,60})"', text)
    phrases += re.findall(r"'([^']{3,60})'", text)

    # Capitalised multi-word phrases (likely proper nouns / clause titles)
    phrases += re.findall(r'\b(?:[A-Z][a-z]+\s+){1,4}[A-Z][a-z]+\b', text)

    # Numbers with units (30 days, 12 months, 5 years)
    phrases += re.findall(
        r'\b\d+\s+(?:days?|months?|years?|weeks?|hours?)\b',
        text, flags=re.IGNORECASE
    )

    # Standalone numbers that look like amounts or periods
    phrases += re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)

    # De-duplicate and filter very short phrases
    seen: set[str] = set()
    unique: list[str] = []
    for p in phrases:
        p = p.strip()
        if len(p) >= 3 and p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def _phrase_in_context(phrase: str, chunks: list[RetrievedChunk]) -> bool:
    """Return True if the phrase appears (case-insensitive) in any chunk."""
    phrase_lower = phrase.lower()
    for chunk in chunks:
        if phrase_lower in chunk.text.lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

@dataclass
class GroundingResult:
    grounding_ratio: float          # matched_phrases / total_phrases
    total_phrases: int
    matched_phrases: int
    unmatched: list[str]            # phrases not found in any chunk


def check_source_grounding(
    answer: str,
    retrieved_chunks: list[RetrievedChunk],
) -> GroundingResult:
    """
    Verify that key phrases in the answer are traceable to retrieved chunks.

    Returns a GroundingResult with grounding_ratio in [0, 1].
    A ratio of 1.0 means every key phrase is grounded in a source chunk.
    """
    phrases = _extract_key_phrases(answer)
    if not phrases:
        # No extractable phrases → can't assess grounding, assume neutral
        return GroundingResult(
            grounding_ratio=0.5,
            total_phrases=0,
            matched_phrases=0,
            unmatched=[],
        )

    matched = [p for p in phrases if _phrase_in_context(p, retrieved_chunks)]
    unmatched = [p for p in phrases if not _phrase_in_context(p, retrieved_chunks)]

    ratio = len(matched) / len(phrases)
    return GroundingResult(
        grounding_ratio=ratio,
        total_phrases=len(phrases),
        matched_phrases=len(matched),
        unmatched=unmatched,
    )


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Map cross-encoder score (−∞ to +∞) to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def compute_confidence(
    retrieved_chunks: list[RetrievedChunk],
    grounding: GroundingResult,
) -> float:
    """
    Composite confidence score (0–1):
      40% — source grounding ratio
      40% — sigmoid(top cross-encoder score)
      20% — 1 – normalised score spread (agreement among top chunks)

    Args:
        retrieved_chunks: Ranked results from the retriever.
        grounding:        Output of check_source_grounding.

    Returns:
        Confidence float in [0.0, 1.0].
    """
    if not retrieved_chunks:
        return 0.0

    # Component 1: Grounding ratio
    grounding_component = grounding.grounding_ratio

    # Component 2: Sigmoid of top re-rank score
    top_rerank = retrieved_chunks[0].rerank_score
    rerank_component = _sigmoid(top_rerank)

    # Component 3: Score agreement (low spread → high confidence)
    if len(retrieved_chunks) > 1:
        scores = [_sigmoid(c.rerank_score) for c in retrieved_chunks]
        spread = max(scores) - min(scores)
        # Normalise: spread in [0, 1]; invert so low spread → high score
        agreement = max(0.0, 1.0 - spread)
    else:
        agreement = 1.0

    confidence = (
        GROUNDING_WEIGHT * grounding_component
        + RERANK_WEIGHT   * rerank_component
        + SPREAD_WEIGHT   * agreement
    )

    # Clamp to [0, 1]
    return round(min(1.0, max(0.0, confidence)), 4)


# ---------------------------------------------------------------------------
# Answer refusal
# ---------------------------------------------------------------------------

def should_refuse(confidence: float) -> bool:
    """Return True if confidence is below the refusal threshold."""
    return confidence < CONFIDENCE_REFUSAL_THRESHOLD
