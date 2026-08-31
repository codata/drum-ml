"""Prompts and templates package for DRUM-ML."""

from drum_ml.prompts.templates_archetypes import (
    generate_scaffolds_for_constant,
    generate_scaffolds_for_unit,
)
from drum_ml.prompts.templates_personas import PERSONA_SYSTEM_PROMPTS

__all__ = [
    "PERSONA_SYSTEM_PROMPTS",
    "generate_scaffolds_for_constant",
    "generate_scaffolds_for_unit",
]
