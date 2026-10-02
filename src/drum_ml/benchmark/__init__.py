"""DRUM Metrology Benchmark (M-Eval) Suite."""

from drum_ml.benchmark.viewer import generate_benchmark_html, save_benchmark_viewer

try:
    from drum_ml.benchmark.models import (
        BenchmarkFormat,
        BenchmarkSample,
        BenchmarkScorecard,
        BenchmarkTask,
        DifficultyTier,
        MCQOption,
        TaskScore,
    )

    __all__ = [
        "BenchmarkFormat",
        "BenchmarkSample",
        "BenchmarkScorecard",
        "BenchmarkTask",
        "DifficultyTier",
        "MCQOption",
        "TaskScore",
        "generate_benchmark_html",
        "save_benchmark_viewer",
    ]
except ImportError:
    __all__ = [
        "generate_benchmark_html",
        "save_benchmark_viewer",
    ]
