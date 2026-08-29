"""M-Eval Metrology Benchmark Evaluation Runner."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from drum_ml.symbolic.latex_parser import sanitize_latex_units
from drum_ml.symbolic.pint_engine import check_unit_conversion_equivalence


class MEvalBenchmark:
    """Evaluates local or frontier models against the held-out M-Eval metrology test set."""

    def __init__(self, benchmark_file: str = "./dataset/test.jsonl"):
        self.benchmark_file = Path(benchmark_file)

    def load_benchmark_records(self) -> List[Dict[str, Any]]:
        records = []
        if not self.benchmark_file.exists():
            return records
        with open(self.benchmark_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def grade_response(self, ground_truth: str, predicted: str) -> Dict[str, Any]:
        """Dual-format grading: deterministic string check + symbolic equivalence check."""
        gt_clean = sanitize_latex_units(ground_truth)
        pred_clean = sanitize_latex_units(predicted)

        # 1. Exact string match
        exact_match = (gt_clean == pred_clean) or (ground_truth.strip() == predicted.strip())

        # 2. Symbolic equivalence check (if containing numbers/units)
        is_sym_eq, err = check_unit_conversion_equivalence(gt_clean, pred_clean) if not exact_match else (True, None)

        passed = exact_match or is_sym_eq
        return {
            "passed": passed,
            "exact_match": exact_match,
            "symbolic_match": is_sym_eq,
            "error": err,
        }
