"""Tests for DRUM Benchmark Generator."""

import pytest

from drum_ml.benchmark.generator import DRUMBenchmarkGenerator
from drum_ml.benchmark.models import (
    BenchmarkFormat,
    BenchmarkTask,
)
from drum_ml.models.entities import (
    CanonicalEntityStore,
    ConstantCategory,
    ConversionRelation,
    DimensionVector,
    PhysicalConstantEntity,
    UnitEntity,
)


@pytest.fixture
def mock_entity_store():
    store = CanonicalEntityStore()

    # 1. Exact Defining Constant: Planck constant
    store.constants["c_h"] = PhysicalConstantEntity(
        uri="http://qudt.org/vocab/constant/PlanckConstant",
        name="Planck constant",
        symbol="h",
        latex_symbol="h",
        category=ConstantCategory.EXACT_SI_DEFINING,
        numeric_value="6.62607015e-34",
        unit_symbol="J s",
        dimension_vector=DimensionVector(L=2, M=1, T=-1),
    )

    # 2. Recommended Constant: Newtonian gravitation
    store.constants["c_G"] = PhysicalConstantEntity(
        uri="http://qudt.org/vocab/constant/NewtonianConstantOfGravitation",
        name="Newtonian constant of gravitation",
        symbol="G",
        latex_symbol="G",
        category=ConstantCategory.CODATA_RECOMMENDED,
        numeric_value="6.67430e-11",
        standard_uncertainty="0.00015e-11",
        relative_uncertainty="2.2e-5",
        unit_symbol="m^3 kg^-1 s^-2",
        dimension_vector=DimensionVector(L=3, M=-1, T=-2),
    )

    # 3. Units
    store.units["u_joule"] = UnitEntity(
        uri="http://qudt.org/vocab/unit/J",
        symbol="J",
        label="joule",
        dimension_vector=DimensionVector(L=2, M=1, T=-2),
        conversion=ConversionRelation(multiplier=1.0, exact=True),
    )
    store.units["u_kwh"] = UnitEntity(
        uri="http://qudt.org/vocab/unit/KiloW-HR",
        symbol="kW*h",
        label="kilowatt-hour",
        dimension_vector=DimensionVector(L=2, M=1, T=-2),
        conversion=ConversionRelation(multiplier=3600000.0, exact=True),
    )

    return store


def test_benchmark_generator_all_tasks(mock_entity_store):
    generator = DRUMBenchmarkGenerator(entities=mock_entity_store, seed=42)
    samples = generator.generate_all(samples_per_task=5)

    assert len(samples) > 0

    tasks_found = set(s.task for s in samples)
    assert BenchmarkTask.CONSTANTS in tasks_found
    assert BenchmarkTask.DIMENSIONS in tasks_found
    assert BenchmarkTask.CONVERSIONS in tasks_found
    assert BenchmarkTask.HOMOGENEITY in tasks_found
    assert BenchmarkTask.CONVENTIONS in tasks_found
    assert BenchmarkTask.UNCERTAINTY in tasks_found


def test_benchmark_mcq_options_integrity(mock_entity_store):
    generator = DRUMBenchmarkGenerator(entities=mock_entity_store, seed=42)
    c_samples = generator.generate_constants_task(count=4)

    mcq_samples = [s for s in c_samples if s.format == BenchmarkFormat.MCQ]
    assert len(mcq_samples) > 0

    for s in mcq_samples:
        assert s.options is not None
        assert len(s.options) == 4
        keys = [opt.key for opt in s.options]
        assert sorted(keys) == ["A", "B", "C", "D"]
        assert s.correct_option_key in ["A", "B", "C", "D"]
        correct_opts = [opt for opt in s.options if opt.is_correct]
        assert len(correct_opts) == 1
        assert correct_opts[0].key == s.correct_option_key
