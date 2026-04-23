"""
generation.py — LLM generation with grounded source citation.

Uses OpenAI GPT-4o (or any compatible model) with a tightly constrained
system prompt that forces context-anchored answers and explicit citation.
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from .retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise legal document analyst.

RULES — follow them strictly:
1. Answer ONLY using the provided context excerpts. Do not use any external knowledge.
2. If the answer is not present in the context, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
3. Quote relevant clause text verbatim where possible, using double quotes.
4. Always end your answer with the source reference in this format:
   [Source: <document name>, Page <number>]
   If the answer spans multiple sources, list each one.
5. Be concise. Do not pad the answer.
6. Never fabricate names, dates, amounts, or clause numbers."""


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format the context + question for the LLM."""
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"--- Context {i} ---\n"
            f"Document: {chunk.document}\n"
            f"Page: {chunk.page_number}\n"
            f"Section: {chunk.section_title or 'N/A'}\n\n"
            f"{chunk.text}"
        )

    context_block = "\n\n".join(context_parts)
    return (
        f"CONTEXT EXCERPTS:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (cite sources inline):"
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LLMGenerator:
    """
    Wraps the OpenAI chat completion API for grounded legal Q&A.

    Args:
        model:        OpenAI model name (default: gpt-4o-mini for cost efficiency).
        temperature:  Sampling temperature (0 for deterministic answers).
        max_tokens:   Maximum tokens in the generated answer.
        api_key:      OpenAI API key (reads OPENAI_API_KEY env var if None).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        """
        Generate a grounded answer from the question and retrieved chunks.

        Returns:
            The LLM's text response.
        """
        user_message = _build_user_message(question, retrieved_chunks)

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )

        return response.choices[0].message.content.strip()
