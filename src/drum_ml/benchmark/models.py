"""Data Models for the DRUM Metrology Benchmark (M-Eval) Suite."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkTask(str, Enum):
    """The 6 Core Metrology Benchmark Tasks."""
    CONSTANTS = "constants"               # Task 1: Fundamental Physical Constants & SI 2019
    DIMENSIONS = "dimensions"             # Task 2: Dimensional Decomposition & Base SI
    CONVERSIONS = "conversions"           # Task 3: Unit Conversions & Affine Transformations
    HOMOGENEITY = "homogeneity"           # Task 4: Error Detection & Dimensional Homogeneity
    CONVENTIONS = "conventions"           # Task 5: SI Typography & Metrological Conventions
    UNCERTAINTY = "uncertainty"           # Task 6: Metrological Uncertainty (GUM) & Sig-Figs


class BenchmarkFormat(str, Enum):
    """Evaluation formats."""
    MCQ = "mcq"                           # 4-option multiple choice (A, B, C, D)
    FREE_FORM = "free_form"               # Open-ended reasoning and exact symbolic calculation


class DifficultyTier(str, Enum):
    INTRODUCTORY = "introductory"         # High school / basic scientific lookup
    INTERMEDIATE = "intermediate"         # Undergraduate STEM / applied engineering
    ADVANCED = "advanced"                 # Metrology lab / standards / GUM level


class MCQOption(BaseModel):
    key: str                              # "A", "B", "C", "D"
    text: str                             # Option description / formula
    is_correct: bool = False
    distractor_rationale: Optional[str] = None # Why this distractor was crafted


class BenchmarkSample(BaseModel):
    id: str
    task: BenchmarkTask
    format: BenchmarkFormat
    difficulty: DifficultyTier = DifficultyTier.INTERMEDIATE
    question: str
    context: Optional[str] = None
    options: Optional[List[MCQOption]] = None # For MCQ
    correct_option_key: Optional[str] = None  # "A", "B", "C", "D"
    ground_truth_answer: str                  # Canonical target string or formula
    entity_uri: Optional[str] = None          # Provenance URI (BIPM, CODATA, QUDT)
    explanation: Optional[str] = None         # Complete derivation / metrological explanation
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    task_breakdown: Dict[str, TaskScore] = Field(default_factory=dict)
    format_breakdown: Dict[str, TaskScore] = Field(default_factory=dict)
    difficulty_breakdown: Dict[str, TaskScore] = Field(default_factory=dict)
    detailed_results: List[Dict[str, Any]] = Field(default_factory=list)
