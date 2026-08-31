"""DRUM Metrology Benchmark (M-Eval) Evaluation & Grading Engine.

Evaluates local or frontier models against the held-out DRUM benchmark suite,
performing dual-track verification (deterministic MCQ extraction + symbolic physics equivalence)
and generating stratified metrology scorecards across all 6 sub-disciplines.
"""

import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from drum_ml.benchmark.models import (
    BenchmarkFormat,
    BenchmarkScorecard,
    TaskScore,
)
from drum_ml.symbolic.latex_parser import sanitize_latex_units
from drum_ml.symbolic.pint_engine import check_unit_conversion_equivalence


class MEvalBenchmark:
    """Evaluator and grader for the DRUM Metrology Benchmark suite."""

    def __init__(self, benchmark_file: str = "./dataset/drum_benchmark_mcq.jsonl"):
        self.benchmark_file = Path(benchmark_file)

    def load_benchmark_records(self) -> list[dict[str, Any]]:
        """Load benchmark samples from JSONL."""
        records = []
        if not self.benchmark_file.exists():
            return records
        with open(self.benchmark_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def extract_mcq_answer(self, response_text: str) -> str | None:
        """Extracts the predicted MCQ option letter (A, B, C, D) from an LLM response."""
        text = response_text.strip()
        if not text:
            return None

        # Pattern 1: Exact standalone letter "A", "B", "C", "D"
        if text.upper() in ["A", "B", "C", "D"]:
            return text.upper()

        # Pattern 2: "Option A", "Option (A)", "(A)", "[A]", "Answer: A", "**A**"
        patterns = [
            r"(?:the\s+correct\s+answer\s+is|correct\s+option\s+is|answer\s*[:=]?)\s*[\*\_]*\(?([A-Da-d])\)?",
            r"[\*\_]*\(([A-Da-d])\)[\*\_]*",
            r"[\*\_]*\[([A-Da-d])\][\*\_]*",
            r"option\s+([A-Da-d])\b",
            r"^\s*([A-Da-d])[\.\:\)]\s*",
            r"\b([A-Da-d])\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        return None

    def grade_response(
        self,
        record: dict[str, Any],
        predicted_text: str,
    ) -> dict[str, Any]:
        """Grades a response against ground truth using MCQ extraction or symbolic equivalence."""
        fmt = record.get("format", BenchmarkFormat.MCQ.value)
        correct_key = record.get("correct_option_key")
        gt_answer = record.get("ground_truth_answer", "")

        # Case 1: Standard Chat JSONL compatibility (fallback)
        if not correct_key and "messages" in record:
            messages = record.get("messages", [])
            gt_answer = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
            fmt = BenchmarkFormat.FREE_FORM.value

        # Track A: MCQ Grading
        if fmt == BenchmarkFormat.MCQ.value or correct_key:
            pred_key = self.extract_mcq_answer(predicted_text)
            passed = (pred_key == correct_key) if (pred_key and correct_key) else False
            return {
                "passed": passed,
                "format": BenchmarkFormat.MCQ.value,
                "predicted_key": pred_key,
                "expected_key": correct_key,
                "predicted_raw": predicted_text[:200],
                "error": None if passed else f"Expected option {correct_key}, got {pred_key}",
            }

        # Track B: Free-Form / Symbolic Physics Equivalence
        gt_clean = sanitize_latex_units(gt_answer)
        pred_clean = sanitize_latex_units(predicted_text)

        exact_match = (gt_clean.lower() == pred_clean.lower()) or (
            gt_answer.strip().lower() == predicted_text.strip().lower()
        )
        is_sym_eq, sym_err = False, None

        if not exact_match:
            is_sym_eq, sym_err = check_unit_conversion_equivalence(gt_clean, pred_clean)

        passed = exact_match or is_sym_eq
        return {
            "passed": passed,
            "format": BenchmarkFormat.FREE_FORM.value,
            "exact_match": exact_match,
            "symbolic_match": is_sym_eq,
            "error": None if passed else sym_err,
        }

    def compute_scorecard(
        self,
        model_name: str,
        results: list[dict[str, Any]],
    ) -> BenchmarkScorecard:
        """Computes stratified scores across all 6 tasks, formats, and difficulties."""
        total = len(results)
        passed = sum(1 for r in results if r["grade"]["passed"])
        overall_pct = (passed / total * 100.0) if total > 0 else 0.0

        # Sub-breakdowns
        task_stats: dict[str, dict[str, int]] = {}
        fmt_stats: dict[str, dict[str, int]] = {}
        diff_stats: dict[str, dict[str, int]] = {}

        for r in results:
            t = r.get("task", "unknown")
            f = r.get("format", "unknown")
            d = r.get("difficulty", "intermediate")
            is_p = 1 if r["grade"]["passed"] else 0

            # Task
            if t not in task_stats:
                task_stats[t] = {"total": 0, "passed": 0}
            task_stats[t]["total"] += 1
            task_stats[t]["passed"] += is_p

            # Format
            if f not in fmt_stats:
                fmt_stats[f] = {"total": 0, "passed": 0}
            fmt_stats[f]["total"] += 1
            fmt_stats[f]["passed"] += is_p

            # Difficulty
            if d not in diff_stats:
                diff_stats[d] = {"total": 0, "passed": 0}
            diff_stats[d]["total"] += 1
            diff_stats[d]["passed"] += is_p

        def to_task_score_dict(d_dict: dict[str, dict[str, int]]) -> dict[str, TaskScore]:
            out = {}
            for k, v in d_dict.items():
                pct = (v["passed"] / v["total"] * 100.0) if v["total"] > 0 else 0.0
                out[k] = TaskScore(task=k, total=v["total"], passed=v["passed"], accuracy_pct=pct)
            return out

        return BenchmarkScorecard(
            model_name=model_name,
            total_samples=total,
            passed_samples=passed,
            overall_accuracy_pct=overall_pct,
            task_breakdown=to_task_score_dict(task_stats),
            format_breakdown=to_task_score_dict(fmt_stats),
            difficulty_breakdown=to_task_score_dict(diff_stats),
            detailed_results=results,
        )

    def print_scorecard(
        self, scorecard: BenchmarkScorecard, console: Console | None = None
    ) -> None:
        """Prints a rich, formatted evaluation scorecard to the terminal."""
        con = console or Console()
        table = Table(
            title=f"DRUM Metrology Benchmark Scorecard: {scorecard.model_name}",
            title_style="bold magenta",
            header_style="bold cyan",
        )
        table.add_column("Category / Sub-Discipline", style="bold")
        table.add_column("Samples", justify="right")
        table.add_column("Passed", justify="right")
        table.add_column("Accuracy", justify="right")

        # Task breakdown
        task_names = {
            "constants": "1. Fundamental Constants & SI 2019",
            "dimensions": "2. Dimensional Decomposition & Base SI",
            "conversions": "3. Unit Conversions & Affine Offsets",
            "homogeneity": "4. Error Detection & Homogeneity",
            "conventions": "5. SI Typography & Metrological Rules",
            "uncertainty": "6. Metrological Uncertainty (GUM)",
        }

        for task_key, task_label in task_names.items():
            if task_key in scorecard.task_breakdown:
                ts = scorecard.task_breakdown[task_key]
                color = (
                    "green"
                    if ts.accuracy_pct >= 80
                    else "yellow"
                    if ts.accuracy_pct >= 50
                    else "red"
                )
                table.add_row(
                    task_label,
                    str(ts.total),
                    str(ts.passed),
                    f"[{color}]{ts.accuracy_pct:.1f}%[/{color}]",
                )

        table.add_section()
        ov_color = (
            "bold green"
            if scorecard.overall_accuracy_pct >= 80
            else "bold yellow"
            if scorecard.overall_accuracy_pct >= 50
            else "bold red"
        )
        table.add_row(
            "OVERALL DRUM BENCHMARK SCORE",
            str(scorecard.total_samples),
            str(scorecard.passed_samples),
            f"[{ov_color}]{scorecard.overall_accuracy_pct:.2f}%[/{ov_color}]",
        )

        con.print("\n")
        con.print(table)
        con.print("\n")
