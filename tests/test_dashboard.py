"""Unit tests for the DRUM Benchmark Multi-Model Leaderboard & Profiles Dashboard."""

import json
import tempfile
import unittest
from pathlib import Path

from drum_ml.benchmark.dashboard import (
    collect_scorecards_from_dir,
    generate_leaderboard_html,
    save_leaderboard_dashboard,
)


class TestLeaderboardDashboard(unittest.TestCase):
    def setUp(self):
        self.sample_scorecard_a = {
            "model_name": "gemma4:12b",
            "total_samples": 100,
            "passed_samples": 75,
            "overall_accuracy_pct": 75.0,
            "total_duration_seconds": 12.5,
            "avg_tokens_per_second": 86.4,
            "environment_profile": {
                "execution_mode": "local",
                "hardware_device": "Apple M4 Max",
                "os_platform": "macOS 15.0",
                "python_version": "3.13.0",
                "api_endpoint": "http://localhost:11434/v1",
            },
            "task_breakdown": {
                "constants": {"task": "constants", "total": 20, "passed": 18, "accuracy_pct": 90.0},
                "dimensions": {
                    "task": "dimensions",
                    "total": 20,
                    "passed": 16,
                    "accuracy_pct": 80.0,
                },
                "conversions": {
                    "task": "conversions",
                    "total": 20,
                    "passed": 14,
                    "accuracy_pct": 70.0,
                },
                "homogeneity": {
                    "task": "homogeneity",
                    "total": 20,
                    "passed": 15,
                    "accuracy_pct": 75.0,
                },
                "conventions": {
                    "task": "conventions",
                    "total": 10,
                    "passed": 8,
                    "accuracy_pct": 80.0,
                },
                "uncertainty": {
                    "task": "uncertainty",
                    "total": 10,
                    "passed": 4,
                    "accuracy_pct": 40.0,
                },
            },
            "detailed_results": [
                {
                    "id": "bench_001",
                    "task": "constants",
                    "format": "mcq",
                    "difficulty": "intermediate",
                    "grade": {"passed": True, "predicted_option": "A", "predicted_text": "A"},
                },
                {
                    "id": "bench_002",
                    "task": "dimensions",
                    "format": "mcq",
                    "difficulty": "intermediate",
                    "grade": {"passed": False, "predicted_option": "C", "predicted_text": "C"},
                },
            ],
        }

        self.sample_scorecard_b = {
            "model_name": "gpt-4o",
            "total_samples": 100,
            "passed_samples": 92,
            "overall_accuracy_pct": 92.0,
            "total_duration_seconds": 18.2,
            "avg_tokens_per_second": 65.0,
            "environment_profile": {
                "execution_mode": "cloud",
                "cloud_provider": "OpenAI",
                "api_endpoint": "https://api.openai.com/v1",
                "python_version": "3.13.0",
            },
            "task_breakdown": {
                "constants": {
                    "task": "constants",
                    "total": 20,
                    "passed": 20,
                    "accuracy_pct": 100.0,
                },
                "dimensions": {
                    "task": "dimensions",
                    "total": 20,
                    "passed": 19,
                    "accuracy_pct": 95.0,
                },
                "conversions": {
                    "task": "conversions",
                    "total": 20,
                    "passed": 18,
                    "accuracy_pct": 90.0,
                },
                "homogeneity": {
                    "task": "homogeneity",
                    "total": 20,
                    "passed": 18,
                    "accuracy_pct": 90.0,
                },
                "conventions": {
                    "task": "conventions",
                    "total": 10,
                    "passed": 9,
                    "accuracy_pct": 90.0,
                },
                "uncertainty": {
                    "task": "uncertainty",
                    "total": 10,
                    "passed": 8,
                    "accuracy_pct": 80.0,
                },
            },
            "detailed_results": [
                {
                    "id": "bench_001",
                    "task": "constants",
                    "format": "mcq",
                    "difficulty": "intermediate",
                    "grade": {"passed": True, "predicted_option": "A", "predicted_text": "A"},
                },
                {
                    "id": "bench_002",
                    "task": "dimensions",
                    "format": "mcq",
                    "difficulty": "intermediate",
                    "grade": {"passed": True, "predicted_option": "B", "predicted_text": "B"},
                },
            ],
        }

        self.sample_benchmark_items = [
            {
                "id": "bench_001",
                "task": "constants",
                "format": "mcq",
                "difficulty": "intermediate",
                "question": "What is the speed of light $c$ in SI base units?",
                "correct_option_key": "A",
                "ground_truth_answer": "299792458 m/s",
                "options": [
                    {"key": "A", "text": "299792458 m/s", "is_correct": True},
                    {"key": "B", "text": "300000000 m/s", "is_correct": False},
                ],
            },
            {
                "id": "bench_002",
                "task": "dimensions",
                "format": "mcq",
                "difficulty": "intermediate",
                "question": "What is the dimensional formula of force?",
                "correct_option_key": "B",
                "ground_truth_answer": "\\text{L}\\cdot\\text{M}\\cdot\\text{T}^{-2}",
                "options": [
                    {
                        "key": "A",
                        "text": "\\text{L}^2\\cdot\\text{M}\\cdot\\text{T}^{-2}",
                        "is_correct": False,
                    },
                    {
                        "key": "B",
                        "text": "\\text{L}\\cdot\\text{M}\\cdot\\text{T}^{-2}",
                        "is_correct": True,
                    },
                ],
            },
        ]

    def test_collect_scorecards_from_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write scorecard files
            (tmp_path / "model_a_scorecard.json").write_text(
                json.dumps(self.sample_scorecard_a), encoding="utf-8"
            )
            (tmp_path / "model_b_scorecard.json").write_text(
                json.dumps(self.sample_scorecard_b), encoding="utf-8"
            )
            # Write a non-scorecard json file that should be skipped
            (tmp_path / "manifest.json").write_text(json.dumps({"count": 2}), encoding="utf-8")

            scorecards = collect_scorecards_from_dir(tmp_path)
            self.assertEqual(len(scorecards), 2)
            names = [s["model_name"] for s in scorecards]
            self.assertIn("gemma4:12b", names)
            self.assertIn("gpt-4o", names)

    def test_collect_scorecards_nonexistent_dir(self):
        scorecards = collect_scorecards_from_dir("/nonexistent/path/for/tests")
        self.assertEqual(scorecards, [])

    def test_generate_leaderboard_html(self):
        scorecards = [self.sample_scorecard_a, self.sample_scorecard_b]
        html = generate_leaderboard_html(
            scorecards=scorecards,
            benchmark_samples=self.sample_benchmark_items,
            title="Custom Test Leaderboard",
        )

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Custom Test Leaderboard", html)
        self.assertIn("gemma4:12b", html)
        self.assertIn("gpt-4o", html)
        self.assertIn("Apple M4 Max", html)
        self.assertIn("OpenAI", html)
        self.assertIn("Leaderboard Matrix", html)
        self.assertIn("Model Arena", html)
        self.assertIn("Environment Profiles", html)
        self.assertIn("katex", html)
        self.assertIn("Average Model Accuracy", html)
        self.assertIn("diff-pill", html)
        self.assertIn("table-tab-pill", html)
        self.assertIn("Sub-Discipline Matrix", html)
        self.assertIn("Difficulty Tier Breakdown", html)
        self.assertIn("Track A (MCQ) vs. Track B", html)

    def test_save_leaderboard_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write scorecards
            (tmp_path / "gemma_scorecard.json").write_text(
                json.dumps(self.sample_scorecard_a), encoding="utf-8"
            )
            (tmp_path / "gpt4o_scorecard.json").write_text(
                json.dumps(self.sample_scorecard_b), encoding="utf-8"
            )

            # Write benchmark questions
            bench_file = tmp_path / "drum_benchmark_mcq.jsonl"
            with open(bench_file, "w", encoding="utf-8") as f:
                for item in self.sample_benchmark_items:
                    f.write(json.dumps(item) + "\n")

            out_html = tmp_path / "leaderboard.html"
            result_path = save_leaderboard_dashboard(
                output_html_path=out_html,
                benchmark_dir=tmp_path,
                benchmark_file=bench_file,
                open_browser=False,
            )

            self.assertTrue(result_path.exists())
            content = result_path.read_text(encoding="utf-8")
            self.assertIn("DRUM Metrology Benchmark", content)
            self.assertIn("gemma4:12b", content)
            self.assertIn("gpt-4o", content)


if __name__ == "__main__":
    unittest.main()
