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


def test_mcq_json_extraction():
    evaluator = MEvalBenchmark()

    # Pure JSON
    pure_json = '{"answer": "B", "explanation": "By the 2019 SI definition, c is exact."}'
    key, exp = evaluator.parse_mcq_response(pure_json)
    assert key == "B"
    assert exp == "By the 2019 SI definition, c is exact."

    # Markdown fenced JSON
    fenced_json = """Here is my answer:
```json
{
  "answer": "C",
  "explanation": "Energy has dimensions L^2 M T^-2."
}
```
"""
    key, exp = evaluator.parse_mcq_response(fenced_json)
    assert key == "C"
    assert "Energy has dimensions" in (exp or "")

    # Reasoning tag with JSON
    reasoning_json = """<think>
Evaluating options:
A is false. B is false. C is false. D is true.
</think>
```json
{
  "selected_option": "D",
  "reasoning": "Standard uncertainty is zero for exact defining constants."
}
```"""
    key, exp = evaluator.parse_mcq_response(reasoning_json)
    assert key == "D"
    assert "Standard uncertainty is zero" in (exp or "")


def test_format_prompt_json():
    record = {
        "question": "What is the exact value of c?",
        "options": [
            {"key": "A", "text": "299 792 458 m/s"},
            {"key": "B", "text": "3.00 x 10^8 m/s"},
        ],
    }
    prompt = MEvalBenchmark.format_prompt(record)
    assert "What is the exact value of c?" in prompt
    assert "A. 299 792 458 m/s" in prompt
    assert "B. 3.00 x 10^8 m/s" in prompt
    assert "json" in prompt.lower()
    assert '"answer"' in prompt


def test_mcq_grading():
    evaluator = MEvalBenchmark()
    record = {
        "id": "sample_001",
        "format": BenchmarkFormat.MCQ.value,
        "correct_option_key": "B",
        "ground_truth_answer": "Option B text",
    }

    # Correct predictions via plain text
    res_correct = evaluator.grade_response(record, "The correct option is (B)")
    assert res_correct["passed"] is True
    assert res_correct["predicted_key"] == "B"

    # Correct predictions via JSON with explanation
    res_json = evaluator.grade_response(
        record,
        '```json\n{"answer": "B", "explanation": "Correct SI multiplier."}\n```',
    )
    assert res_json["passed"] is True
    assert res_json["predicted_key"] == "B"
    assert res_json.get("predicted_explanation") == "Correct SI multiplier."

    # Wrong predictions
    res_wrong = evaluator.grade_response(record, '{"answer": "C", "explanation": "wrong"}')
    assert res_wrong["passed"] is False
    assert res_wrong["predicted_key"] == "C"
    assert res_wrong.get("predicted_explanation") == "wrong"


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
