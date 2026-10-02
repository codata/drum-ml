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

    @staticmethod
    def format_prompt(record: dict[str, Any]) -> str:
        """Format a benchmark record into prompt text for LLM inference."""
        question = record.get("question", "").strip()
        options = record.get("options", [])
        if options:
            formatted_opts = []
            for opt in options:
                k = opt.get("key", "")
                t = opt.get("text", "").strip()
                formatted_opts.append(f"{k}. {t}")
            opts_str = "\n".join(formatted_opts)
            return (
                f"Question: {question}\n\n"
                f"Options:\n{opts_str}\n\n"
                f"Instructions: Analyze the options carefully and select the single correct letter (A, B, C, or D).\n"
                f"Respond in valid JSON format with keys \"answer\" (the single uppercase letter) and \"explanation\" (brief justification).\n\n"
                f"Example response format:\n"
                f"```json\n"
                f'{{\n  "answer": "A",\n  "explanation": "Brief explanation of the metrological rationale."\n}}\n'
                f"```"
            )
        return f"Question: {question}\n\nAnswer:"

    @staticmethod
    def query_model_api(
        endpoint: str,
        model_name: str,
        prompt: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        """Queries an LLM via LiteLLM or an OpenAI-compatible endpoint."""
        # 1. Try LiteLLM first for robust multi-provider handling (Ollama, OpenAI, Claude, vLLM)
        try:
            import litellm

            litellm.suppress_debug_info = True

            model_target = model_name
            api_base = endpoint.rstrip("/")
            if "11434" in endpoint and not model_target.startswith("ollama"):
                model_target = f"ollama/{model_name}"
                api_base = "http://localhost:11434"

            resp = litellm.completion(
                model=model_target,
                api_base=api_base,
                api_key=api_key or "sk-local",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert metrologist and physicist. Answer with extreme precision.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
                timeout=timeout,
            )
            choices = resp.choices if hasattr(resp, "choices") else []
            if choices:
                msg = choices[0].message
                content = getattr(msg, "content", "") or ""
                reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "thinking_content", None)
                if not content and reasoning:
                    return str(reasoning).strip()
                if reasoning and reasoning not in content:
                    return f"<think>\n{reasoning}\n</think>\n{content}".strip()
                return str(content).strip()
        except Exception:
            pass

        # 2. Fallback to direct HTTP request
        import urllib.error
        import urllib.request

        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions") and not url.endswith("/completions"):
            url = f"{url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert metrologist and physicist. Answer metrological questions with extreme precision.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "") or ""
                    reasoning = msg.get("reasoning_content") or msg.get("thinking_content")
                    if not content and reasoning:
                        return str(reasoning).strip()
                    if reasoning and reasoning not in content:
                        return f"<think>\n{reasoning}\n</think>\n{content}".strip()
                    return str(content).strip()
                return ""
        except Exception as e:
            return f"ERROR_CALLING_MODEL: {e}"

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

    @staticmethod
    def parse_mcq_response(response_text: str) -> tuple[str | None, str | None]:
        """Extracts (predicted_key, explanation) from an LLM response.

        Attempts multi-tier parsing:
        1. JSON parsing (fenced ```json ... ``` blocks and raw JSON objects).
        2. Thought-tag stripping (<think>...</think>).
        3. Heuristic regex pattern matching for non-compliant models.
        """
        text = response_text.strip()
        if not text:
            return None, None

        # Strip reasoning tags if present
        cleaned_text = text
        if "</think>" in cleaned_text:
            cleaned_text = cleaned_text.split("</think>")[-1].strip()
        elif "<think>" in cleaned_text:
            cleaned_text = re.sub(r"<think>[\s\S]*?</think>", "", cleaned_text).strip()

        # Helper to extract letter from value
        def extract_letter_from_val(val: Any) -> str | None:
            if not isinstance(val, str):
                return None
            v = val.strip()
            if v.upper() in ["A", "B", "C", "D"]:
                return v.upper()
            m = re.search(r"\b([A-Da-d])\b", v)
            if m:
                return m.group(1).upper()
            return None

        # Tier 1: Try JSON extraction
        # 1a. Markdown fenced blocks
        json_candidates = re.findall(
            r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned_text, flags=re.IGNORECASE
        )
        # 1b. Outermost/balanced JSON objects
        raw_objs = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned_text)
        json_candidates.extend(raw_objs)

        if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            json_candidates.insert(0, cleaned_text)

        key_aliases = [
            "answer",
            "option",
            "choice",
            "selected_option",
            "correct_option",
            "key",
            "selected",
            "response",
        ]
        exp_aliases = [
            "explanation",
            "reasoning",
            "rationale",
            "justification",
            "details",
            "notes",
        ]

        for cand in json_candidates:
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    pred_key = None
                    for k in key_aliases:
                        if k in data:
                            pred_key = extract_letter_from_val(data[k])
                            if pred_key:
                                break

                    pred_exp = None
                    for e in exp_aliases:
                        if e in data and isinstance(data[e], str):
                            pred_exp = data[e].strip()
                            break

                    if pred_key:
                        return pred_key, pred_exp
            except Exception:
                continue

        # Tier 2: Standalone Letter
        if cleaned_text.upper() in ["A", "B", "C", "D"]:
            return cleaned_text.upper(), None

        # Tier 3: Regex heuristics
        patterns = [
            r"(?:the\s+correct\s+answer\s+is|correct\s+option\s+is|answer\s*[:=]?)\s*[\*\_]*\(?([A-Da-d])\)?",
            r'"answer"\s*:\s*"([A-Da-d])"',
            r'"option"\s*:\s*"([A-Da-d])"',
            r"[\*\_]*\(([A-Da-d])\)[\*\_]*",
            r"[\*\_]*\[([A-Da-d])\][\*\_]*",
            r"option\s+([A-Da-d])\b",
            r"^\s*([A-Da-d])[\.\:\)]\s*",
            r"\b([A-Da-d])\b",
        ]
        for pat in patterns:
            m = re.search(pat, cleaned_text, re.IGNORECASE)
            if m:
                return m.group(1).upper(), None

        return None, None

    def extract_mcq_answer(self, response_text: str) -> str | None:
        """Extracts the predicted MCQ option letter (A, B, C, D) from an LLM response."""
        pred_key, _ = self.parse_mcq_response(response_text)
        return pred_key

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
            pred_key, pred_exp = self.parse_mcq_response(predicted_text)
            passed = (pred_key == correct_key) if (pred_key and correct_key) else False
            grade_dict: dict[str, Any] = {
                "passed": passed,
                "format": BenchmarkFormat.MCQ.value,
                "predicted_key": pred_key,
                "expected_key": correct_key,
                "predicted_raw": predicted_text,
                "error": None if passed else f"Expected option {correct_key}, got {pred_key}",
            }
            if pred_exp:
                grade_dict["predicted_explanation"] = pred_exp
            return grade_dict

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
