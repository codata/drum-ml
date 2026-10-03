"""Unit tests for MetrologyScaffolder and the 6 Pedagogical Archetypes."""

from drum_ml.models.entities import (
    CanonicalEntityStore,
    ConstantCategory,
    ConversionRelation,
    DimensionVector,
    PhysicalConstantEntity,
    QuantityKindEntity,
    UnitEntity,
)
from drum_ml.models.scaffolds import ArchetypeType
from drum_ml.pipeline.scaffolder import MetrologyScaffolder
from drum_ml.prompts.templates_archetypes import (
    generate_scaffolds_for_constant,
    generate_scaffolds_for_unit,
)


def create_test_store() -> CanonicalEntityStore:
    store = CanonicalEntityStore()

    # Add Quantity Kind
    force_qk = QuantityKindEntity(
        uri="http://qudt.org/vocab/quantitykind/Force",
        label="Force",
        symbol="F",
        dimension_vector=DimensionVector(L=1, M=1, T=-2),
    )
    store.quantity_kinds[force_qk.uri] = force_qk

    # Add Unit with conversion
    newton = UnitEntity(
        uri="http://qudt.org/vocab/unit/N",
        symbol="N",
        label="Newton",
        is_si_derived=True,
        is_coherent=True,
        dimension_vector=DimensionVector(L=1, M=1, T=-2),
        has_quantity_kinds=[force_qk.uri],
        conversion=ConversionRelation(multiplier=1.0, offset=0.0, exact=True),
    )
    store.units[newton.uri] = newton

    # Add Exact Defining Constant
    c = PhysicalConstantEntity(
        uri="https://si-digital-framework.org/SI/constants/speed_of_light",
        name="Speed of light in vacuum",
        symbol="c",
        latex_symbol="c",
        category=ConstantCategory.EXACT_SI_DEFINING,
        numeric_value="299792458",
        unit_symbol="m/s",
        dimension_vector=DimensionVector(L=1, T=-1),
        defining_year=2019,
    )
    store.constants[c.uri] = c

    # Add Measured Constant with uncertainty
    g = PhysicalConstantEntity(
        uri="https://codata.org/constants/2022/G",
        name="Newtonian constant of gravitation",
        symbol="G",
        latex_symbol="G",
        category=ConstantCategory.CODATA_RECOMMENDED,
        numeric_value="6.67430e-11",
        standard_uncertainty="0.00015e-11",
        relative_uncertainty="2.2e-5",
        unit_symbol="m^3/(kg*s^2)",
        dimension_vector=DimensionVector(L=3, M=-1, T=-2),
        defining_year=2022,
    )
    store.constants[g.uri] = g

    return store


def test_generate_scaffolds_for_unit():
    store = create_test_store()
    unit = store.units["http://qudt.org/vocab/unit/N"]
    scaffolds = generate_scaffolds_for_unit(unit, store)

    archetypes = {s.archetype for s in scaffolds}
    assert ArchetypeType.DIRECT_IDENTIFICATION in archetypes
    assert ArchetypeType.DIMENSIONAL_DECOMPOSITION in archetypes
    assert ArchetypeType.CONVERSION_SCALING in archetypes
    assert ArchetypeType.DIMENSIONAL_ERROR_DETECTION in archetypes
    assert ArchetypeType.SEMANTIC_TOOL_USE in archetypes

    for s in scaffolds:
        assert s.entity_uri == unit.uri
        assert len(s.canonical_query) > 10
        assert len(s.ground_truth_answer) > 20


def test_generate_scaffolds_for_constant():
    store = create_test_store()

    # Exact Constant
    c = store.constants["https://si-digital-framework.org/SI/constants/speed_of_light"]
    c_scaffolds = generate_scaffolds_for_constant(c)
    c_archetypes = {s.archetype for s in c_scaffolds}

    assert ArchetypeType.DIRECT_IDENTIFICATION in c_archetypes
    assert ArchetypeType.DIMENSIONAL_DECOMPOSITION in c_archetypes
    assert ArchetypeType.SEMANTIC_TOOL_USE in c_archetypes
    assert ArchetypeType.METROLOGICAL_UNCERTAINTY in c_archetypes

    # Uncertainty should state exact
    unc_scaffold = next(
        s for s in c_scaffolds if s.archetype == ArchetypeType.METROLOGICAL_UNCERTAINTY
    )
    assert "exact" in unc_scaffold.ground_truth_answer.lower()

    # Measured Constant
    g = store.constants["https://codata.org/constants/2022/G"]
    g_scaffolds = generate_scaffolds_for_constant(g)
    g_unc = next(s for s in g_scaffolds if s.archetype == ArchetypeType.METROLOGICAL_UNCERTAINTY)
    assert "relative uncertainty" in g_unc.ground_truth_answer.lower()
    assert "expanded uncertainty" in g_unc.ground_truth_answer.lower()
    assert "jcgm 100:2008" in g_unc.ground_truth_answer.lower()


def test_scaffolder_generate_all(tmp_path):
    store = create_test_store()
    scaffolder = MetrologyScaffolder()
    all_scaffolds = scaffolder.generate_all(store)

    assert len(all_scaffolds) > 0
    all_archetypes = {s.archetype for s in all_scaffolds}
    # All 6 archetypes must be represented across units and constants
    assert len(all_archetypes) == 6

    json_path = tmp_path / "scaffolds.json"
    scaffolder.save_to_json(all_scaffolds, str(json_path))
    assert json_path.exists()
