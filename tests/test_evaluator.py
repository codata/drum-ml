"""Tests for DRUM Benchmark Evaluator and Grading Engine."""

import pytest

from drum_ml.benchmark.models import (
    BenchmarkFormat,
    BenchmarkTask,
)
from drum_ml.pipeline.evaluator import MEvalBenchmark


def test_mcq_answer_extraction():
    evaluator = MEvalBenchmark()

    # Standalone letter
    assert evaluator.extract_mcq_answer("A") == "A"
    assert evaluator.extract_mcq_answer(" c ") == "C"

    # Prefix / Parentheses
    assert evaluator.extract_mcq_answer("(B)") == "B"
    assert evaluator.extract_mcq_answer("[D]") == "D"
    assert evaluator.extract_mcq_answer("Option C") == "C"
    assert evaluator.extract_mcq_answer("The correct answer is (A)") == "A"
    assert evaluator.extract_mcq_answer("Answer: **B**") == "B"
    assert evaluator.extract_mcq_answer("D. Some description here") == "D"


def test_mcq_grading():
    evaluator = MEvalBenchmark()
    record = {
        "id": "sample_001",
        "format": BenchmarkFormat.MCQ.value,
        "correct_option_key": "B",
        "ground_truth_answer": "Option B text",
    }

    # Correct predictions
    res_correct = evaluator.grade_response(record, "The correct option is (B)")
    assert res_correct["passed"] is True
    assert res_correct["predicted_key"] == "B"

    # Wrong predictions
    res_wrong = evaluator.grade_response(record, "Option (C) is correct")
    assert res_wrong["passed"] is False
    assert res_wrong["predicted_key"] == "C"


def test_scorecard_computation():
    evaluator = MEvalBenchmark()
    results = [
        {
            "id": "c1",
            "task": BenchmarkTask.CONSTANTS.value,
            "format": BenchmarkFormat.MCQ.value,
            "difficulty": "introductory",
            "grade": {"passed": True},
        },
        {
            "id": "c2",
            "task": BenchmarkTask.CONSTANTS.value,
            "format": BenchmarkFormat.MCQ.value,
            "difficulty": "intermediate",
            "grade": {"passed": False},
        },
        {
            "id": "d1",
            "task": BenchmarkTask.DIMENSIONS.value,
            "format": BenchmarkFormat.MCQ.value,
            "difficulty": "intermediate",
            "grade": {"passed": True},
        },
    ]

    card = evaluator.compute_scorecard("test_model", results)
    assert card.total_samples == 3
    assert card.passed_samples == 2
    assert pytest.approx(card.overall_accuracy_pct, 0.1) == 66.67
    assert card.task_breakdown[BenchmarkTask.CONSTANTS.value].accuracy_pct == 50.0
    assert card.task_breakdown[BenchmarkTask.DIMENSIONS.value].accuracy_pct == 100.0
