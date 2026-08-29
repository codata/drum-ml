Scientific Assumptions & Invariants
====================================

The DRUM-ML pipeline is built upon explicit metrological and computational invariants:

1. Standards Hierarchy Invariant
--------------------------------
When definitions, conversion multipliers, or constant values overlap, the pipeline strictly adheres to:

.. math::

   \textbf{Tier 1 (BIPM SI Framework)} \succ \textbf{Tier 2 (CODATA DRUM Constants)} \succ \textbf{Tier 3 (QUDT 2.1)}

- BIPM SI 9th Edition (2019) is the ultimate source of truth for the 7 SI base units and the 7 exact defining constants.
- CODATA recommended values (2018/2022) provide authoritative values and uncertainty budgets for empirical physical constants.
- QUDT provides the broader semantic graph for non-SI, legacy, and domain-specific units.

2. Exact SI 2019 Defining Constant Invariant
--------------------------------------------
Under the 2019 SI redefinition:

- Speed of light :math:`c = 299\,792\,458\text{ m/s}` (exact, :math:`u=0`).
- Planck constant :math:`h = 6.626\,070\,15 \times 10^{-34}\text{ J}\cdot\text{s}` (exact, :math:`u=0`).
- Elementary charge :math:`e = 1.602\,176\,634 \times 10^{-19}\text{ C}` (exact, :math:`u=0`).
- Boltzmann constant :math:`k = 1.380\,649 \times 10^{-23}\text{ J/K}` (exact, :math:`u=0`).
- Avogadro constant :math:`N_{\text{A}} = 6.022\,140\,76 \times 10^{23}\text{ mol}^{-1}` (exact, :math:`u=0`).
- Hyperfine transition frequency of :math:`^{133}\text{Cs}` :math:`\Delta\nu_{\text{Cs}} = 9\,192\,631\,770\text{ Hz}` (exact, :math:`u=0`).
- Luminous efficacy :math:`K_{\text{cd}} = 683\text{ lm/W}` (exact, :math:`u=0`).

No floating-point rounding drift or truncation is permitted for these values.

3. Affine vs. Differential Temperature Invariant
------------------------------------------------
Temperature conversions must distinguish between **absolute gauge points** and **temperature intervals**:

- Absolute point conversion: :math:`T_{\text{K}} = T_{^{\circ}\text{C}} + 273.15`
- Differential interval conversion: :math:`\Delta T_{\text{K}} = \Delta T_{^{\circ}\text{C}}` (no offset applied).

4. Zero Hardcoded Data Assumption
---------------------------------
All units, quantity kinds, dimensions, and physical constants are parsed dynamically from RDF Turtle ontologies or versioned JSON files via SPARQL. The codebase contains no static dictionary tables of units or constants.

5. Entity Isolation Guard (Train/Test Partitioning)
---------------------------------------------------
All augmented variants of an entity and its associated scaffolds are strictly isolated within a single split (Train 85%, Val 10%, or Test 5%) to prevent data leakage into the held-out benchmark.
