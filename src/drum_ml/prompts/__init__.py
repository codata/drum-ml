"""Prompts and templates package for DRUM-ML."""

from drum_ml.prompts.templates_archetypes import (
    generate_scaffolds_for_unit,
    generate_scaffolds_for_constant,
)
from drum_ml.prompts.templates_personas import PERSONA_SYSTEM_PROMPTS

__all__ = [
    "generate_scaffolds_for_unit",
    "generate_scaffolds_for_constant",
    "PERSONA_SYSTEM_PROMPTS",
]
