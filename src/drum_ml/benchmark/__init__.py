"""DRUM Metrology Benchmark (M-Eval) Suite."""

from drum_ml.benchmark.dashboard import (
    collect_scorecards_from_dir,
    generate_leaderboard_html,
    save_leaderboard_dashboard,
)
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
        "collect_scorecards_from_dir",
        "generate_benchmark_html",
        "generate_leaderboard_html",
        "save_benchmark_viewer",
        "save_leaderboard_dashboard",
    ]
except ImportError:
    __all__ = [
        "collect_scorecards_from_dir",
        "generate_benchmark_html",
        "generate_leaderboard_html",
        "save_benchmark_viewer",
        "save_leaderboard_dashboard",
    ]
