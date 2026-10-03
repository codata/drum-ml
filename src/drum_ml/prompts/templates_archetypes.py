"""Templates and Generators for the 6 Metrological Archetypes."""

from drum_ml.models.entities import CanonicalEntityStore, PhysicalConstantEntity, UnitEntity
from drum_ml.models.scaffolds import ArchetypeType, ScaffoldRecord


def generate_scaffolds_for_unit(
    unit: UnitEntity, store: CanonicalEntityStore
) -> list[ScaffoldRecord]:
    """Generates pedagogical scaffolds for a unit across all applicable archetypes."""
    scaffolds: list[ScaffoldRecord] = []
    symbol_str = unit.latex_symbol or unit.symbol

    # ---------------------------------------------------------
    # Archetype 1: Direct Identification & Symbol Mapping
    # ---------------------------------------------------------
    qk_labels = []
    for qk_uri in unit.has_quantity_kinds:
        if qk_uri in store.quantity_kinds:
            label = store.quantity_kinds[qk_uri].label
            if label and label.lower() != "unknown" and "unknown" not in qk_uri.lower():
                qk_labels.append(label)
    qk_str = ", ".join(qk_labels) if qk_labels else "unspecified / general physical quantity"

    q1 = f"What is a {unit.label} ({symbol_str}) and what does it measure?"
    a1 = (
        f"The **{unit.label}** (symbol: ${symbol_str}$) is a unit of measurement "
        f"associated with the quantity kind(s): **{qk_str}**.\n\n"
        f"- **Official Symbol:** ${symbol_str}$\n"
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

    # ---------------------------------------------------------
    # Archetype 2: Dimensional Decomposition & Base SI
    # ---------------------------------------------------------
    q2 = f"Derive the SI base unit decomposition of the unit '{unit.label}' (${symbol_str}$)."
    a2 = (
        f"To derive the base SI decomposition of the **{unit.label}** (${symbol_str}$):\n\n"
        f"1. Identify its base dimensional powers: $$[{symbol_str}] = {unit.dimension_vector.to_latex()}$$\n"
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

    # ---------------------------------------------------------
    # Archetype 3: Conversion Scaling & Multipliers
    # ---------------------------------------------------------
    if unit.conversion:
        q3 = f"What is the exact conversion relation from '{unit.label}' (${symbol_str}$) to coherent SI base units?"
        a3 = (
            f"The conversion relation for **{unit.label}** (${symbol_str}$) is:\n\n"
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

    # ---------------------------------------------------------
    # Archetype 4: Dimensional Error Detection & Homogeneity
    # ---------------------------------------------------------
    # Pick a mismatched dimension for homogeneity violation check
    dim_vec = unit.dimension_vector
    if dim_vec.L == 0 and dim_vec.M == 0 and dim_vec.T == 0:
        mismatch_name = "length"
        mismatch_symbol = "m"
        mismatch_dim = "\\text{L}"
    elif dim_vec.M != 0:
        mismatch_name = "length"
        mismatch_symbol = "m"
        mismatch_dim = "\\text{L}"
    else:
        mismatch_name = "mass"
        mismatch_symbol = "kg"
        mismatch_dim = "\\text{M}"

    q4 = (
        f"Evaluate whether the physical expression $X = Y + Z$ is dimensionally valid, where $Y$ is "
        f"measured in {unit.label} (${symbol_str}$) and $Z$ is a {mismatch_name} measured in {mismatch_symbol} (${mismatch_dim}$)."
    )
    a4 = (
        f"The expression $X = Y + Z$ is **dimensionally invalid and physically undefined**.\n\n"
        f"1. **Dimensional Analysis:**\n"
        f"- Dimension of $Y$ ({unit.label}): $$[Y] = {dim_vec.to_latex()}$$\n"
        f"- Dimension of $Z$ ({mismatch_name}): $$[Z] = {mismatch_dim}$$\n\n"
        f"2. **Fourier's Principle of Dimensional Homogeneity:**\n"
        f"In physics and metrology (ISO 80000-1 / VIM3), two quantities can only be summed, subtracted, or compared if they possess the identical dimension vector ($[Y] = [Z]$).\n\n"
        f"3. **Conclusion:**\n"
        f"Because $$[{dim_vec.to_latex()}] \\neq [{mismatch_dim}]$$, the addition violates dimensional homogeneity."
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{unit.symbol}_arch4",
            archetype=ArchetypeType.DIMENSIONAL_ERROR_DETECTION,
            entity_uri=unit.uri,
            canonical_query=q4,
            ground_truth_answer=a4,
        )
    )

    # ---------------------------------------------------------
    # Archetype 5: Semantic Tool Use & Serialization
    # ---------------------------------------------------------
    q5 = (
        f"Write an executable Python snippet using `pint` to define a quantity in '{unit.label}' (${unit.symbol}$), "
        f"convert it to SI base units, and provide the SPARQL 1.1 query to retrieve its definition from the QUDT ontology."
    )
    a5 = (
        f"### 1. Python `pint` Execution\n\n"
        f"```python\n"
        f"import pint\n\n"
        f"ureg = pint.UnitRegistry()\n\n"
        f"# Define quantity in {unit.label}\n"
        f"try:\n"
        f'    quantity = 1.0 * ureg("{unit.symbol}")\n'
        f"    si_quantity = quantity.to_base_units()\n"
        f'    print(f"Original: {{quantity}}")\n'
        f'    print(f"SI Base: {{si_quantity}}")\n'
        f'    print(f"Dimensionality: {{quantity.dimensionality}}")\n'
        f"except Exception:\n"
        f'    print(f"Unit: {unit.label} ({unit.symbol})")\n'
        f"```\n\n"
        f"### 2. SPARQL 1.1 Query (QUDT Ontology)\n\n"
        f"```sparql\n"
        f"PREFIX qudt: <http://qudt.org/schema/qudt/>\n"
        f"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n\n"
        f"SELECT ?label ?symbol ?multiplier ?offset ?dimensionVector WHERE {{\n"
        f"  VALUES ?unit {{ <{unit.uri}> }}\n"
        f"  OPTIONAL {{ ?unit rdfs:label ?label . }}\n"
        f"  OPTIONAL {{ ?unit qudt:symbol ?symbol . }}\n"
        f"  OPTIONAL {{ ?unit qudt:conversionMultiplier ?multiplier . }}\n"
        f"  OPTIONAL {{ ?unit qudt:conversionOffset ?offset . }}\n"
        f"  OPTIONAL {{ ?unit qudt:hasDimensionVector ?dimensionVector . }}\n"
        f"}}\n"
        f"```"
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{unit.symbol}_arch5",
            archetype=ArchetypeType.SEMANTIC_TOOL_USE,
            entity_uri=unit.uri,
            canonical_query=q5,
            ground_truth_answer=a5,
        )
    )

    # ---------------------------------------------------------
    # Archetype 6: Metrological Uncertainty & GUM
    # ---------------------------------------------------------
    q6 = (
        f"Explain how measurement uncertainty is expressed for experimental quantities measured in "
        f"'{unit.label}' (${symbol_str}$) in accordance with the GUM (JCGM 100:2008)."
    )
    a6 = (
        f"Under JCGM 100:2008 (Guide to the Expression of Uncertainty in Measurement - GUM), any experimental value measured in **{unit.label}** (${symbol_str}$) is reported with its associated uncertainty:\n\n"
        f"1. **Standard Uncertainty Representation:**\n"
        f"$$y = (y_0 \\pm u(y))\\text{{ {unit.symbol}}}$$\n"
        f"where $y_0$ is the best estimate and $u(y)$ is the standard uncertainty (one standard deviation).\n\n"
        f"2. **Expanded Uncertainty ($U$ with coverage factor $k=2$ for ~95.45% confidence):**\n"
        f"$$U = k \\cdot u_c(y) = 2 \\cdot u_c(y)$$\n\n"
        f"3. **Dimensional Homogeneity of Uncertainty:**\n"
        f"The uncertainty $u(y)$ and expanded uncertainty $U$ must share the exact same physical dimension as the measured quantity: $${unit.dimension_vector.to_latex()}$$."
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{unit.symbol}_arch6",
            archetype=ArchetypeType.METROLOGICAL_UNCERTAINTY,
            entity_uri=unit.uri,
            canonical_query=q6,
            ground_truth_answer=a6,
        )
    )

    return scaffolds


def generate_scaffolds_for_constant(constant: PhysicalConstantEntity) -> list[ScaffoldRecord]:
    """Generates pedagogical scaffolds for a fundamental physical constant across applicable archetypes."""
    scaffolds: list[ScaffoldRecord] = []
    is_exact = constant.category.value == "exact_si_defining"

    # ---------------------------------------------------------
    # Archetype 1: Direct Identification & Value
    # ---------------------------------------------------------
    q1 = f"What is the value, standard uncertainty, and SI unit of the physical constant '{constant.name}' (${constant.latex_symbol}$)?"
    unc_text = (
        "Exact defining value with zero uncertainty ($u=0$) under the BIPM 2019 SI revision."
        if is_exact
        else f"Standard uncertainty: $u = {constant.standard_uncertainty}$, Relative uncertainty: $u_r = {constant.relative_uncertainty}$."
    )
    a1 = (
        f"**{constant.name}** (${constant.latex_symbol}$):\n\n"
        f"- **Value:** ${constant.numeric_value}\\text{{ {constant.unit_symbol}}}$\n"
        f"- **Category:** {constant.category.value}\n"
        f"- **Uncertainty:** {unc_text}\n"
        f"- **Defining Year:** {constant.defining_year}\n"
        f"- **Base SI Dimensions:** $${constant.dimension_vector.to_latex()}$$"
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{constant.symbol}_const_arch1",
            archetype=ArchetypeType.DIRECT_IDENTIFICATION,
            entity_uri=constant.uri,
            canonical_query=q1,
            ground_truth_answer=a1,
        )
    )

    # ---------------------------------------------------------
    # Archetype 2: Dimensional Decomposition of Constant
    # ---------------------------------------------------------
    q2 = f"Derive the base SI dimension vector and coherent base units for the fundamental constant '{constant.name}' (${constant.latex_symbol}$)."
    a2 = (
        f"The fundamental constant **{constant.name}** (${constant.latex_symbol}$) has the following base SI structure:\n\n"
        f"1. **Base SI Dimension Vector:** $$[{constant.latex_symbol}] = {constant.dimension_vector.to_latex()}$$\n"
        f"2. **SI Base Unit Decomposition:** $$1\\text{{ {constant.unit_symbol}}} = {constant.dimension_vector.to_si_base_unit_latex()}$$\n"
        f"3. **Value in SI Base Units:** $${constant.numeric_value} \\cdot ({constant.dimension_vector.to_si_base_unit_latex()})$$"
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{constant.symbol}_const_arch2",
            archetype=ArchetypeType.DIMENSIONAL_DECOMPOSITION,
            entity_uri=constant.uri,
            canonical_query=q2,
            ground_truth_answer=a2,
        )
    )

    # ---------------------------------------------------------
    # Archetype 5: Semantic Tool Use for Physical Constant
    # ---------------------------------------------------------
    q5 = (
        f"Provide a SPARQL 1.1 query to retrieve the value, uncertainty, and unit of the constant "
        f"'{constant.name}' (${constant.symbol}$) from the CODATA / BIPM ontology."
    )
    a5 = (
        f"### SPARQL 1.1 Query for {constant.name}\n\n"
        f"```sparql\n"
        f"PREFIX qudt: <http://qudt.org/schema/qudt/>\n"
        f"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        f"PREFIX drum: <https://codata.org/constants/ontology/>\n\n"
        f"SELECT ?label ?symbol ?numericValue ?standardUncertainty ?unitSymbol WHERE {{\n"
        f"  VALUES ?constant {{ <{constant.uri}> }}\n"
        f"  OPTIONAL {{ ?constant rdfs:label ?label . }}\n"
        f"  OPTIONAL {{ ?constant qudt:symbol ?symbol . }}\n"
        f"  OPTIONAL {{ ?constant qudt:numericValue ?numericValue . }}\n"
        f"  OPTIONAL {{ ?constant qudt:standardUncertainty ?standardUncertainty . }}\n"
        f"  OPTIONAL {{ ?constant qudt:unitSymbol ?unitSymbol . }}\n"
        f"}}\n"
        f"```"
    )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{constant.symbol}_const_arch5",
            archetype=ArchetypeType.SEMANTIC_TOOL_USE,
            entity_uri=constant.uri,
            canonical_query=q5,
            ground_truth_answer=a5,
        )
    )

    # ---------------------------------------------------------
    # Archetype 6: Metrological Uncertainty & GUM for Constant
    # ---------------------------------------------------------
    q6 = (
        f"Explain the uncertainty budget and coverage factor for the fundamental constant "
        f"'{constant.name}' (${constant.latex_symbol}$) under CODATA recommendations and the 2019 SI redefinition."
    )
    if is_exact:
        a6 = (
            f"Under the 2019 BIPM SI revision (9th Edition), **{constant.name}** (${constant.latex_symbol}$) is an **exact SI defining constant**:\n\n"
            f"- **Exact Defined Value:** $${constant.latex_symbol} = {constant.numeric_value}\\text{{ {constant.unit_symbol}}}$$\n"
            f"- **Standard Uncertainty ($u$):** $$u({constant.latex_symbol}) = 0$$\n"
            f"- **Relative Standard Uncertainty ($u_r$):** $$u_r({constant.latex_symbol}) = 0$$\n"
            f"- **Metrological Rule:** By international treaty (CGPM), defining constants have zero uncertainty by definition and contribute zero variance to derived measurement budgets."
        )
    else:
        a6 = (
            f"Under CODATA recommendations and JCGM 100:2008 (GUM), **{constant.name}** (${constant.latex_symbol}$) is an experimentally determined constant:\n\n"
            f"- **Recommended Value:** $${constant.latex_symbol} = {constant.numeric_value}\\text{{ {constant.unit_symbol}}}$$\n"
            f"- **Standard Uncertainty ($u$):** $$u({constant.latex_symbol}) = {constant.standard_uncertainty or 'N/A'}\\text{{ {constant.unit_symbol}}}$$\n"
            f"- **Relative Uncertainty ($u_r$):** $$u_r({constant.latex_symbol}) = {constant.relative_uncertainty or 'N/A'}$$\n"
            f"- **Expanded Uncertainty ($U$, $k=2$ for ~95.45% confidence):**\n"
            f"$$U = 2 \\cdot u({constant.latex_symbol}) = 2 \\times ({constant.standard_uncertainty or '0'})\\text{{ {constant.unit_symbol}}}$$"
        )
    scaffolds.append(
        ScaffoldRecord(
            id=f"{constant.symbol}_const_arch6",
            archetype=ArchetypeType.METROLOGICAL_UNCERTAINTY,
            entity_uri=constant.uri,
            canonical_query=q6,
            ground_truth_answer=a6,
        )
    )

    return scaffolds
