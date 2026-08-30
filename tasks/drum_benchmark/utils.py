"""Document processing utilities and filters for lm-evaluation-harness DRUM tasks."""

from typing import Any, Dict, List


def filter_by_task(dataset: Any, task_name: str) -> Any:
    """Filter dataset items by their metrological task name."""
    return dataset.filter(lambda example: example.get("task") == task_name)


def filter_constants(dataset: Any) -> Any:
    return filter_by_task(dataset, "constants")


def filter_dimensions(dataset: Any) -> Any:
    return filter_by_task(dataset, "dimensions")


def filter_conversions(dataset: Any) -> Any:
    return filter_by_task(dataset, "conversions")


def filter_homogeneity(dataset: Any) -> Any:
    return filter_by_task(dataset, "homogeneity")


def filter_conventions(dataset: Any) -> Any:
    return filter_by_task(dataset, "conventions")


def filter_uncertainty(dataset: Any) -> Any:
    return filter_by_task(dataset, "uncertainty")


def format_mcq_doc(doc: Dict[str, Any]) -> str:
    """Formats a BenchmarkSample document into standard multiple-choice prompt text."""
    question = doc.get("question", "").strip()
    options = doc.get("options", [])
    
    formatted_opts = []
    for opt in options:
        k = opt.get("key", "")
        t = opt.get("text", "").strip()
        formatted_opts.append(f"{k}. {t}")
    
    opts_str = "\n".join(formatted_opts)
    return f"Question: {question}\n{opts_str}\nAnswer:"
