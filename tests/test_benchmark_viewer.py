"""Unit tests for the DRUM Benchmark HTML Interactive Viewer."""

import json
import tempfile
import unittest
from pathlib import Path

from drum_ml.benchmark.viewer import generate_benchmark_html, save_benchmark_viewer


class TestBenchmarkViewer(unittest.TestCase):
    def test_generate_benchmark_html_basic(self):
        sample_records = [
            {
                "id": "drum_bench_test_001",
                "task": "dimensions",
                "format": "mcq",
                "difficulty": "intermediate",
                "question": "What is the dimensional formula for velocity?",
                "options": [
                    {
                        "key": "A",
                        "text": "\\text{m}\\cdot\\text{s}^{-1}",
                        "is_correct": True,
                        "distractor_rationale": "Ground truth",
                    },
                    {
                        "key": "B",
                        "text": "\\text{m}\\cdot\\text{s}^{-2}",
                        "is_correct": False,
                        "distractor_rationale": "Acceleration error",
                    },
                    {
                        "key": "C",
                        "text": "\\text{kg}\\cdot\\text{m}",
                        "is_correct": False,
                        "distractor_rationale": "Momentum error",
                    },
                    {
                        "key": "D",
                        "text": "\\text{s}^{-1}",
                        "is_correct": False,
                        "distractor_rationale": "Frequency error",
                    },
                ],
                "correct_option_key": "A",
                "ground_truth_answer": "\\text{m}\\cdot\\text{s}^{-1}",
                "entity_uri": "http://qudt.org/vocab/unit/M-PER-SEC",
                "explanation": "Velocity is length divided by time: L T^-1.",
                "metadata": {
                    "unit_symbol": "m/s",
                    "dim_vector": {"L": 1, "M": 0, "T": -1, "I": 0, "Theta": 0, "N": 0, "J": 0},
                },
            }
        ]
        sample_scorecard = {
            "model_name": "test_baseline",
            "total_samples": 1,
            "passed_samples": 1,
            "overall_accuracy_pct": 100.0,
            "task_breakdown": {
                "dimensions": {"task": "dimensions", "total": 1, "passed": 1, "accuracy_pct": 100.0}
            },
        }

        html = generate_benchmark_html(sample_records, sample_scorecard, "Test Suite")

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("DRUM Metrology Benchmark", html)
        self.assertIn("drum_bench_test_001", html)
        self.assertIn("test_baseline", html)
        self.assertIn("Velocity is length divided by time", html)
        self.assertIn("katex", html)

    def test_save_benchmark_viewer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bench_file = tmp_path / "test_benchmark.jsonl"
            report_file = tmp_path / "test_report.json"
            out_html = tmp_path / "output_viewer.html"

            record = {
                "id": "drum_bench_conv_001",
                "task": "conversions",
                "format": "free_form",
                "difficulty": "introductory",
                "question": "Convert 1 inch to millimeters.",
                "ground_truth_answer": "25.4 mm",
                "entity_uri": "http://qudt.org/vocab/unit/IN",
                "explanation": "1 inch is defined as exactly 25.4 mm.",
            }
            with open(bench_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            report = {
                "model_name": "ground_truth_baseline",
                "total_samples": 1,
                "passed_samples": 1,
                "overall_accuracy_pct": 100.0,
                "task_breakdown": {
                    "conversions": {
                        "task": "conversions",
                        "total": 1,
                        "passed": 1,
                        "accuracy_pct": 100.0,
                    }
                },
            }
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f)

            p = save_benchmark_viewer(
                output_html_path=out_html,
                benchmark_file=bench_file,
                scorecard_file=report_file,
                open_browser=False,
            )

            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 1000)
            content = p.read_text(encoding="utf-8")
            self.assertIn("drum_bench_conv_001", content)
            self.assertIn("25.4 mm", content)


if __name__ == "__main__":
    unittest.main()
