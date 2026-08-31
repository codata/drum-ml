"""Data Models for the DRUM Metrology Benchmark (M-Eval) Suite."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkTask(StrEnum):
    """The 6 Core Metrology Benchmark Tasks."""

    CONSTANTS = "constants"  # Task 1: Fundamental Physical Constants & SI 2019
    DIMENSIONS = "dimensions"  # Task 2: Dimensional Decomposition & Base SI
    CONVERSIONS = "conversions"  # Task 3: Unit Conversions & Affine Transformations
    HOMOGENEITY = "homogeneity"  # Task 4: Error Detection & Dimensional Homogeneity
    CONVENTIONS = "conventions"  # Task 5: SI Typography & Metrological Conventions
    UNCERTAINTY = "uncertainty"  # Task 6: Metrological Uncertainty (GUM) & Sig-Figs


class BenchmarkFormat(StrEnum):
    """Evaluation formats."""

    MCQ = "mcq"  # 4-option multiple choice (A, B, C, D)
    FREE_FORM = "free_form"  # Open-ended reasoning and exact symbolic calculation


class DifficultyTier(StrEnum):
    INTRODUCTORY = "introductory"  # High school / basic scientific lookup
    INTERMEDIATE = "intermediate"  # Undergraduate STEM / applied engineering
    ADVANCED = "advanced"  # Metrology lab / standards / GUM level


class MCQOption(BaseModel):
    key: str  # "A", "B", "C", "D"
    text: str  # Option description / formula
    is_correct: bool = False
    distractor_rationale: str | None = None  # Why this distractor was crafted


class BenchmarkSample(BaseModel):
    id: str
    task: BenchmarkTask
    format: BenchmarkFormat
    difficulty: DifficultyTier = DifficultyTier.INTERMEDIATE
    question: str
    context: str | None = None
    options: list[MCQOption] | None = None  # For MCQ
    correct_option_key: str | None = None  # "A", "B", "C", "D"
    ground_truth_answer: str  # Canonical target string or formula
    entity_uri: str | None = None  # Provenance URI (BIPM, CODATA, QUDT)
    explanation: str | None = None  # Complete derivation / metrological explanation
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskScore(BaseModel):
    task: str
    total: int = 0
    passed: int = 0
    accuracy_pct: float = 0.0


class BenchmarkScorecard(BaseModel):
    model_name: str
    total_samples: int = 0
    passed_samples: int = 0
    overall_accuracy_pct: float = 0.0
    task_breakdown: dict[str, TaskScore] = Field(default_factory=dict)
    format_breakdown: dict[str, TaskScore] = Field(default_factory=dict)
    difficulty_breakdown: dict[str, TaskScore] = Field(default_factory=dict)
    detailed_results: list[dict[str, Any]] = Field(default_factory=list)
