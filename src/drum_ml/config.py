"""Configuration loader and Pydantic Settings for DRUM-ML."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class AugmenterConfig(BaseModel):
    enabled: bool = True
    provider: str = "ollama"
    model: str = "qwen3.8:27b-mlx"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    variations_per_archetype: int = 2
    concurrency_limit: int = 5
    timeout_seconds: int = 60
    max_retries: int = 3
    personas: List[str] = Field(default_factory=lambda: ["general_user"])


class ValidatorConfig(BaseModel):
    strict_mode: bool = True
    verify_latex: bool = True
    verify_dimensions_sympy: bool = True
    verify_quantity_kinds: bool = True
    verify_numerical_precision: bool = True
    verify_sparql_syntax: bool = True
    generate_dpo_pairs: bool = True


class ExporterConfig(BaseModel):
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    dedup_jaccard_threshold: float = 0.85
    formats: List[str] = Field(default_factory=lambda: ["openai", "sharegpt", "dpo"])


class HuggingFaceConfig(BaseModel):
    repo_id: str = "codata/drum-metrology-instruct"
    token: Optional[str] = None
    private: bool = False


class PipelineConfig(BaseModel):
    output_dir: str = "./dataset"
    cache_db: str = "./data/cache/llm_cache.sqlite"
    augmenter: AugmenterConfig = Field(default_factory=AugmenterConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)

    @classmethod
    def load_from_yaml(cls, path: str = "./config.yaml") -> "PipelineConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        aug_data = data.get("augmenter", {})
        val_data = data.get("validator", {})
        exp_data = data.get("exporter", {})
        hf_data = data.get("huggingface", {})
        proj_data = data.get("project", {})

        return cls(
            output_dir=proj_data.get("output_dir", "./dataset"),
            cache_db=proj_data.get("cache_db", "./data/cache/llm_cache.sqlite"),
            augmenter=AugmenterConfig(**aug_data),
            validator=ValidatorConfig(**val_data),
            exporter=ExporterConfig(**exp_data),
            huggingface=HuggingFaceConfig(**hf_data),
        )
