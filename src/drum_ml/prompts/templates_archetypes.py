"""Templates and Generators for the 6 Metrological Archetypes."""

from drum_ml.models.entities import CanonicalEntityStore, PhysicalConstantEntity, UnitEntity
from drum_ml.models.scaffolds import ArchetypeType, ScaffoldRecord


def generate_scaffolds_for_unit(
    unit: UnitEntity, store: CanonicalEntityStore
) -> list[ScaffoldRecord]:
    """Generates pedagogical scaffolds for a unit across archetypes."""
    scaffolds = []

    # Archetype 1: Direct Identification
    qk_labels = []
    for qk_uri in unit.has_quantity_kinds:
        if qk_uri in store.quantity_kinds:
            qk_labels.append(store.quantity_kinds[qk_uri].label)
    qk_str = ", ".join(qk_labels) if qk_labels else "unspecified quantity kind"

    q1 = f"What is a {unit.label} ({unit.latex_symbol or unit.symbol}) and what does it measure?"
    a1 = (
        f"The **{unit.label}** (symbol: ${unit.latex_symbol or unit.symbol}$) is a unit of measurement "
        f"associated with the quantity kind(s): **{qk_str}**.\n\n"
        f"- **Official Symbol:** ${unit.latex_symbol or unit.symbol}$\n"
        f"- **Base SI Dimensional Formula:** $${unit.dimension_vector.to_latex()}$$\n"
        f"- **SI Base Unit Decomposition:** $${unit.dimension_vector.to_si_base_unit_latex()}$$."
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{unit.symbol}_arch1",
            archetype=ArchetypeType.DIRECT_IDENTIFICATION,
            entity_uri=unit.uri,
            canonical_query=q1,
            ground_truth_answer=a1,
        )
    )

    # Archetype 2: Dimensional Decomposition
    q2 = f"Derive the SI base unit decomposition of the unit '{unit.label}' (${unit.latex_symbol or unit.symbol}$)."
    a2 = (
        f"To derive the base SI decomposition of the **{unit.label}** (${unit.latex_symbol or unit.symbol}$):\n\n"
        f"1. Identify its base dimensional powers: $$[{unit.latex_symbol or unit.symbol}] = {unit.dimension_vector.to_latex()}$$\n"
        f"2. Map each dimensional component to its corresponding SI base unit (kg for mass, m for length, s for time, etc.):\n"
        f"$$1\\text{{ {unit.symbol}}} = {unit.dimension_vector.to_si_base_unit_latex()}$$\n\n"
        f"Coherent status: {'Coherent derived SI unit' if unit.is_coherent else 'Non-coherent unit'}."
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{unit.symbol}_arch2",
            archetype=ArchetypeType.DIMENSIONAL_DECOMPOSITION,
            entity_uri=unit.uri,
            canonical_query=q2,
            ground_truth_answer=a2,
        )
    )

    # Archetype 3: Conversion Scaling (if non-coherent or derived)
    if unit.conversion:
        q3 = f"What is the exact conversion relation from '{unit.label}' (${unit.latex_symbol or unit.symbol}$) to coherent SI base units?"
        a3 = (
            f"The conversion relation for **{unit.label}** (${unit.latex_symbol or unit.symbol}$) is:\n\n"
            f"$$\\text{{SI Value}} = (\\text{{Value in {unit.symbol}}} \\times {unit.conversion.multiplier}) + {unit.conversion.offset}$$\n"
            f"Exact conversion factor: **{unit.conversion.exact}**."
        )
        scaffolds.append(
            ScaffoldRecord(
                id=f"{unit.symbol}_arch3",
                archetype=ArchetypeType.CONVERSION_SCALING,
                entity_uri=unit.uri,
                canonical_query=q3,
                ground_truth_answer=a3,
            )
        )

    return scaffolds


def generate_scaffolds_for_constant(constant: PhysicalConstantEntity) -> list[ScaffoldRecord]:
    """Generates pedagogical scaffolds for a fundamental physical constant."""
    q = f"What is the value, standard uncertainty, and SI unit of the physical constant '{constant.name}' (${constant.latex_symbol}$)?"
    unc_text = (
        "Exact defining value with zero uncertainty ($u=0$) under the BIPM 2019 SI revision."
        if constant.category.value == "exact_si_defining"
        else f"Standard uncertainty: $u = {constant.standard_uncertainty}$, Relative uncertainty: $u_r = {constant.relative_uncertainty}$."
    )
    a = (
        f"**{constant.name}** (${constant.latex_symbol}$):\n\n"
        f"- **Value:** ${constant.numeric_value}\\text{{ {constant.unit_symbol}}}$\n"
        f"- **Category:** {constant.category.value}\n"
        f"- **Uncertainty:** {unc_text}\n"
        f"- **Defining Year:** {constant.defining_year}\n"
        f"- **Base SI Dimensions:** $${constant.dimension_vector.to_latex()}$$"
    )
    return [
        ScaffoldRecord(
            id=f"{constant.symbol}_const_arch1",
            archetype=ArchetypeType.DIRECT_IDENTIFICATION,
            entity_uri=constant.uri,
            canonical_query=q,
            ground_truth_answer=a,
        )
    ]
