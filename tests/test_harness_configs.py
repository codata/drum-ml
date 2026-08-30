"""Tests for lm-evaluation-harness task configurations and utilities."""

import sys
from pathlib import Path
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.drum_benchmark.utils import (
    format_mcq_doc,
)


def test_harness_yaml_files_exist_and_valid():
    tasks_dir = Path("tasks/drum_benchmark")
    assert tasks_dir.exists()

    expected_yamls = [
        "drum_benchmark.yaml",
        "drum_constants.yaml",
        "drum_dimensions.yaml",
        "drum_conversions.yaml",
        "drum_homogeneity.yaml",
        "drum_conventions.yaml",
        "drum_uncertainty.yaml",
    ]

    for yml_name in expected_yamls:
        yml_path = tasks_dir / yml_name
        assert yml_path.exists(), f"Missing YAML: {yml_name}"
        
        # Read content and parse YAML
        content = yml_path.read_text(encoding="utf-8")
        # Replace !function tags for basic yaml parsing test
        clean_content = content.replace("!function", "")
        data = yaml.safe_load(clean_content)
        assert data is not None

        if yml_name == "drum_benchmark.yaml":
            assert "group" in data
            assert "task" in data
            assert len(data["task"]) == 6
        else:
            assert "task" in data
            assert "group" in data
            assert data["output_type"] == "multiple_choice"
            assert data["doc_to_choice"] == ["A", "B", "C", "D"]
            assert "metric_list" in data


def test_harness_format_mcq_doc():
    doc = {
        "question": "What is the speed of light in vacuum?",
        "options": [
            {"key": "A", "text": "299792458 m/s"},
            {"key": "B", "text": "300000000 m/s"},
            {"key": "C", "text": "150000000 m/s"},
            {"key": "D", "text": "30000 m/s"},
        ],
        "correct_option_key": "A",
    }
    prompt = format_mcq_doc(doc)
    assert "Question: What is the speed of light in vacuum?" in prompt
    assert "A. 299792458 m/s" in prompt
    assert "B. 300000000 m/s" in prompt
    assert "Answer:" in prompt
