"""Agent 2: Pedagogical Archetype Generator."""

import json
from pathlib import Path

from drum_ml.models.entities import CanonicalEntityStore
from drum_ml.models.scaffolds import ScaffoldRecord
from drum_ml.prompts.templates_archetypes import (
    generate_scaffolds_for_constant,
    generate_scaffolds_for_unit,
)


class MetrologyScaffolder:
    """Agent 2: Generates canonical ground-truth instruction-response scaffolds across all 6 archetypes."""

    def __init__(self, entities_path: str | None = None):
        self.entities_path = Path(entities_path) if entities_path else None

    def generate_all(
        self,
        store: CanonicalEntityStore,
        limit_per_category: int | None = None,
    ) -> list[ScaffoldRecord]:
        """Generates canonical scaffolds for units and constants in the store.

        Args:
            store: The canonical entity store.
            limit_per_category: Optional maximum number of units and constants to scaffold (for quick sample generation).
        """
        scaffolds: list[ScaffoldRecord] = []

        # 1. Generate scaffolds for units
        unit_items = list(store.units.values())
        if limit_per_category is not None:
            unit_items = unit_items[:limit_per_category]

        for unit in unit_items:
            scaffolds.extend(generate_scaffolds_for_unit(unit, store))

        # 2. Generate scaffolds for constants
        const_items = list(store.constants.values())
        if limit_per_category is not None:
            const_items = const_items[:limit_per_category]

        for const in const_items:
            scaffolds.extend(generate_scaffolds_for_constant(const))

        return scaffolds

    def save_to_json(
        self, scaffolds: list[ScaffoldRecord], output_path: str = "./data/scaffolds.json"
    ) -> Path:
        """Serializes scaffolds list to JSON with UTF-8 encoding."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            data = [s.model_dump() for s in scaffolds]
            json.dump(data, f, indent=2)
        return out
