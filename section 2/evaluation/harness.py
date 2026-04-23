"""
harness.py — Self-contained evaluation harness for the RAG pipeline.

Loads QA pairs from a JSON file, runs each question through the pipeline's
retriever, and computes Precision@3.

The harness evaluates the RETRIEVAL stage only (not generation), because:
- Retrieval is the critical failure mode in RAG.
- Generation quality requires human evaluation of free-text answers.
- Retrieval can be measured automatically via chunk matching.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .metrics import precision_at_k

console = Console()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class EvaluationHarness:
    """
    Runs the retrieval evaluation over a set of QA pairs.

    Args:
        pipeline: A fully initialised RAGPipeline instance.
        qa_path:  Path to the JSON file with QA pairs.
    """

    def __init__(self, pipeline, qa_path: str | Path) -> None:
        self._pipeline = pipeline
        self._qa_path = Path(qa_path)

        if not self._qa_path.exists():
            raise FileNotFoundError(f"QA pairs file not found: {self._qa_path}")

        with open(self._qa_path, encoding="utf-8") as f:
            self._qa_pairs: list[dict] = json.load(f)

        console.print(f"[green]✓[/green] Loaded {len(self._qa_pairs)} QA pairs from "
                      f"[cyan]{self._qa_path.name}[/cyan]")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, k: int = 3, verbose: bool = True) -> dict:
        """
        Execute the evaluation harness.

        Args:
            k:       K for Precision@K.
            verbose: Print per-question results.

        Returns:
            Results dict from precision_at_k().
        """
        console.print(f"\n[bold]Running Retrieval Evaluation (Precision@{k})[/bold]")
        console.print(f"  Questions: {len(self._qa_pairs)}")
        console.print(f"  Documents: {', '.join(self._pipeline.list_documents())}\n")

        all_retrieved: list[list[dict]] = []
        timings: list[float] = []

        for i, qa in enumerate(self._qa_pairs, start=1):
            question = qa["question"]
            console.print(f"[dim]Q{i:02d}:[/dim] {question[:80]}{'…' if len(question)>80 else ''}")

            t0 = time.perf_counter()
            # Use pipeline's internal retriever directly for efficiency
            # (avoids LLM call + grounding check during evaluation)
            result = self._pipeline.query(question)
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            # Normalise sources to retrieval format expected by metrics
            sources = result.get("sources", [])
            all_retrieved.append(sources)

            console.print(
                f"  → Retrieved {len(sources)} sources | "
                f"confidence={result.get('confidence', 0.0):.3f} | "
                f"latency={elapsed:.2f}s"
            )

        # ── Compute metrics ──────────────────────────────────────────
        metrics = precision_at_k(all_retrieved, self._qa_pairs, k=k)
        metrics["avg_latency_s"] = round(sum(timings) / len(timings), 3) if timings else 0.0

        # ── Print results ────────────────────────────────────────────
        self._print_results(metrics, k)

        return metrics

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def _print_results(self, metrics: dict, k: int) -> None:
        console.print()

        # Summary panel
        score = metrics["precision_at_k"]
        colour = "green" if score >= 0.7 else ("yellow" if score >= 0.4 else "red")
        console.print(Panel(
            f"[bold {colour}]Precision@{k} = {score:.4f}  "
            f"({metrics['hits']}/{metrics['total']} questions)[/bold {colour}]\n"
            f"Avg retrieval latency: {metrics.get('avg_latency_s', 0):.2f}s",
            title="[bold]Evaluation Results[/bold]",
            border_style=colour,
        ))

        # Per-question table
        table = Table(
            title=f"Per-Question Results (top-{k})",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("ID", style="dim", width=6)
        table.add_column("Question", max_width=45)
        table.add_column("Hit?", justify="center", width=6)
        table.add_column("Expected Doc / Page", max_width=30)
        table.add_column("Retrieved Docs / Pages", max_width=35)

        for pq in metrics["per_question"]:
            hit_str = "[green]✓[/green]" if pq["hit"] else "[red]✗[/red]"
            retrieved_str = "\n".join(
                f"{d} p{p}" for d, p in pq["top_k_docs"]
            ) or "—"
            table.add_row(
                pq["id"],
                pq["question"][:45] + ("…" if len(pq["question"]) > 45 else ""),
                hit_str,
                f"{pq['expected_doc']} p{pq['expected_page']}",
                retrieved_str,
            )

        console.print(table)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------

    def save_report(self, metrics: dict, output_path: str | Path) -> None:
        """Save evaluation metrics to a JSON file."""
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        console.print(f"\n[green]✓[/green] Report saved to [cyan]{output_path}[/cyan]")
