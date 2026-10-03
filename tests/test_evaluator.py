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


def test_freeform_grading():
    evaluator = MEvalBenchmark()
    record = {
        "id": "sample_ff_001",
        "format": BenchmarkFormat.FREE_FORM.value,
        "ground_truth_answer": "1.054571817e-34 J s",
    }

    # Exact string match
    res1 = evaluator.grade_response(record, "1.054571817e-34 J s")
    assert res1["passed"] is True
    assert res1["exact_match"] is True

    # JSON formatted response with explanation
    res2 = evaluator.grade_response(
        record,
        '```json\n{"answer": "1.054571817e-34 J s", "explanation": "Exact by 2019 SI definition."}\n```',
    )
    assert res2["passed"] is True
    assert res2.get("extracted_answer") == "1.054571817e-34 J s"
    assert "Exact by 2019" in res2.get("predicted_explanation", "")

    # Symbolic unit equivalent (kg m^2 / s)
    res3 = evaluator.grade_response(
        record,
        '```json\n{"answer": "1.054571817e-34 kg * m^2 / s"}\n```',
    )
    assert res3["passed"] is True
    assert res3["symbolic_match"] is True

    # Boxed LaTeX expression
    res4 = evaluator.grade_response(
        record,
        r"The value of \hbar is \boxed{1.054571817 \times 10^{-34} \text{ J s}}",
    )
    assert res4["passed"] is True

    # Conversational text containing equation
    res5 = evaluator.grade_response(
        record,
        "The atomic unit of action is defined as 1.054571817e-34 J s.",
    )
    assert res5["passed"] is True


def test_scorecard_computation():
    evaluator = MEvalBenchmark()
    results = [
        {
            "id": "c1",
            "task": BenchmarkTask.CONSTANTS.value,
            "format": BenchmarkFormat.MCQ.value,
            "difficulty": "introductory",
            "grade": {"passed": True},
            "metrics": {
                "latency_seconds": 1.0,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "tokens_per_second": 50.0,
            },
        },
        {
            "id": "c2",
            "task": BenchmarkTask.CONSTANTS.value,
            "format": BenchmarkFormat.MCQ.value,
            "difficulty": "intermediate",
            "grade": {"passed": False},
            "metrics": {
                "latency_seconds": 2.0,
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "total_tokens": 160,
                "tokens_per_second": 20.0,
            },
        },
        {
            "id": "d1",
            "task": BenchmarkTask.DIMENSIONS.value,
            "format": BenchmarkFormat.MCQ.value,
            "difficulty": "intermediate",
            "grade": {"passed": True},
            "metrics": {
                "latency_seconds": 1.5,
                "prompt_tokens": 80,
                "completion_tokens": 30,
                "total_tokens": 110,
                "tokens_per_second": 20.0,
            },
        },
    ]

    card = evaluator.compute_scorecard("test_model", results, total_duration_seconds=4.5)
    assert card.total_samples == 3
    assert card.passed_samples == 2
    assert pytest.approx(card.overall_accuracy_pct, 0.1) == 66.67
    assert card.task_breakdown[BenchmarkTask.CONSTANTS.value].accuracy_pct == 50.0
    assert card.task_breakdown[BenchmarkTask.DIMENSIONS.value].accuracy_pct == 100.0

    # Environment & Version Metadata
    assert card.benchmark_version is not None
    assert card.timestamp is not None
    assert card.environment is not None
    assert card.environment.os != ""
    assert card.environment.architecture != ""

    # Performance Metrics
    assert card.metrics is not None
    assert card.metrics.total_prompt_tokens == 300
    assert card.metrics.total_completion_tokens == 120
    assert card.metrics.total_tokens == 420
    assert pytest.approx(card.metrics.avg_latency_seconds, 0.01) == 1.5
    assert card.metrics.p50_latency_seconds == 1.5
    assert pytest.approx(card.metrics.avg_tokens_per_second, 0.01) == 26.67


def test_anonymous_environment():
    from drum_ml.pipeline.evaluator import detect_gpu_info, get_anonymous_environment

    env = get_anonymous_environment(endpoint="http://localhost:11434/v1", model_name="gemma4:e2b-mlx")
    assert env.os != ""
    assert env.architecture != ""
    assert env.python_version != ""
    assert env.cpu_count is not None and env.cpu_count >= 1
    assert env.execution_type == "local"
    assert "Ollama" in env.provider
    assert env.is_local_inference is True

    gpu_name, cnt, _vram = detect_gpu_info()
    if gpu_name:
        assert isinstance(gpu_name, str)
        assert cnt is None or cnt >= 1


def test_classify_endpoint():
    from drum_ml.pipeline.evaluator import classify_endpoint

    # Local Ollama (local model)
    t, prov, _ep, is_loc = classify_endpoint("http://localhost:11434/v1", "gemma4:e2b-mlx")
    assert t == "local"
    assert is_loc is True
    assert "Ollama" in prov

    # Ollama proxying a cloud-hosted model (e.g. gemma4:31b-cloud)
    t, prov, _ep, is_loc = classify_endpoint("http://localhost:11434/v1", "gemma4:31b-cloud")
    assert t == "cloud"
    assert is_loc is False
    assert "Ollama Cloud" in prov

    # Ollama with tagged cloud model (e.g. deepseek-r1:cloud)
    t, prov, _ep, is_loc = classify_endpoint("http://localhost:11434/v1", "deepseek-r1:cloud")
    assert t == "cloud"
    assert is_loc is False
    assert "Ollama Cloud" in prov

    # Explicit override via execution_type='cloud'
    t, prov, _ep, is_loc = classify_endpoint("http://localhost:11434/v1", "gemma4:e2b", execution_type="cloud")
    assert t == "cloud"
    assert is_loc is False

    # Local vLLM
    t, prov, _ep, is_loc = classify_endpoint("http://127.0.0.1:8000/v1", "llama3")
    assert t == "local"
    assert is_loc is True
    assert "vLLM" in prov

    # Remote Cloud OpenAI
    t, prov, _ep, is_loc = classify_endpoint("https://api.openai.com/v1", "gpt-4o")
    assert t == "cloud"
    assert is_loc is False
    assert "OpenAI" in prov

    # Remote Cloud Groq
    t, prov, _ep, is_loc = classify_endpoint("https://api.groq.com/openai/v1", "llama-3.3-70b")
    assert t == "cloud"
    assert is_loc is False
    assert "Groq" in prov

    # In-memory baseline
    t, prov, _ep, is_loc = classify_endpoint(None, "ground_truth_baseline")
    assert t == "baseline"
    assert is_loc is True
    assert "Baseline" in prov


def test_checkpoint_save_and_load(tmp_path):
    evaluator = MEvalBenchmark()
    target_scorecard = tmp_path / "gemma4_12b_scorecard.json"
    ckpt_path = evaluator.get_checkpoint_path(target_scorecard)

    assert ckpt_path.name == ".gemma4_12b_scorecard.checkpoint.json"

    # Initially empty
    loaded, acc_time = evaluator.load_checkpoint(ckpt_path)
    assert loaded == {}
    assert acc_time == 0.0

    sample_results = [
        {
            "id": "item_001",
            "task": "constants",
            "format": "mcq",
            "grade": {"passed": True},
            "metrics": {"latency_seconds": 1.2},
        },
        {
            "id": "item_002",
            "task": "dimensions",
            "format": "mcq",
            "grade": {"passed": False},
            "metrics": {"latency_seconds": 2.1},
        },
    ]

    # Save checkpoint
    saved_path = evaluator.save_checkpoint(
        checkpoint_path=ckpt_path,
        model_name="gemma4:12b-mlx",
        benchmark_file="./benchmark.jsonl",
        total_records=300,
        results_list=sample_results,
        accumulated_duration_seconds=123.45,
    )
    assert saved_path.exists()

    # Load with matching model name
    loaded, acc_time = evaluator.load_checkpoint(ckpt_path, expected_model_name="gemma4:12b-mlx")
    assert len(loaded) == 2
    assert "item_001" in loaded
    assert "item_002" in loaded
    assert loaded["item_001"]["grade"]["passed"] is True
    assert pytest.approx(acc_time, 0.01) == 123.45

    # Model name mismatch returns empty
    loaded_mismatch, _ = evaluator.load_checkpoint(ckpt_path, expected_model_name="different_model")
    assert loaded_mismatch == {}

    # Remove checkpoint
    assert evaluator.remove_checkpoint(ckpt_path) is True
    assert not ckpt_path.exists()
    assert evaluator.remove_checkpoint(ckpt_path) is False

