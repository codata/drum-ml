"""DRUM Metrology Benchmark Dataset Generator.

Synthesizes standardized, held-out gold benchmark samples across the 6 Core Metrology Tasks
in both Multiple-Choice Question (MCQ) and Free-Form Symbolic reasoning formats.
"""

import json
import random
from pathlib import Path

from drum_ml.benchmark.models import (
    BenchmarkFormat,
    BenchmarkSample,
    BenchmarkTask,
    DifficultyTier,
    MCQOption,
)
from drum_ml.models.entities import (
    CanonicalEntityStore,
    ConstantCategory,
)


class DRUMBenchmarkGenerator:
    """Generates the DRUM Metrology Benchmark suite from canonical RDF entities."""

    def __init__(self, entities: CanonicalEntityStore, seed: int = 42):
        self.entities = entities
        self.rng = random.Random(seed)

    def generate_all(
        self,
        samples_per_task: int = 50,
    ) -> list[BenchmarkSample]:
        """Generate a balanced benchmark dataset across all 6 tasks and formats."""
        samples: list[BenchmarkSample] = []
        samples.extend(self.generate_constants_task(count=samples_per_task))
        samples.extend(self.generate_dimensions_task(count=samples_per_task))
        samples.extend(self.generate_conversions_task(count=samples_per_task))
        samples.extend(self.generate_homogeneity_task(count=samples_per_task))
        samples.extend(self.generate_conventions_task(count=samples_per_task))
        samples.extend(self.generate_uncertainty_task(count=samples_per_task))
        return samples

    # =========================================================================
    # Task 1: Fundamental Physical Constants & SI 2019
    # =========================================================================
    def generate_constants_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        constants_list = list(self.entities.constants.values())
        if not constants_list:
            return samples

        # 1. Exact defining SI 2019 constants identification
        exact_constants = [
            c for c in constants_list if c.category == ConstantCategory.EXACT_SI_DEFINING
        ]
        recommended_constants = [
            c for c in constants_list if c.category == ConstantCategory.CODATA_RECOMMENDED
        ]

        for i, c in enumerate(exact_constants):
            # MCQ: Value and exactness
            c_val = c.numeric_value
            unit_str = c.unit_symbol or ""

            q_text = (
                f"Under the 2019 SI redefinition, what is the exact defined numerical value "
                f"of the {c.name} ({c.symbol})?"
            )
            correct_opt = f"{c_val} {unit_str} (Exact, standard uncertainty u = 0)"

            # Distractors
            try:
                val_float = float(c_val)
                distractor_1 = (
                    f"{val_float * 1.0001:.7e} {unit_str} (Experimental with u_r = 1.2e-8)"
                )
                distractor_2 = f"{val_float * 0.9990:.7e} {unit_str} (Pre-2019 value)"
                distractor_3 = f"{c_val} {unit_str} (Recommended value with u = 4.5e-9)"
            except Exception:
                distractor_1 = f"{c_val}01 {unit_str} (Uncertainty u_r = 1.0e-7)"
                distractor_2 = f"{c_val} (Approximate value)"
                distractor_3 = f"{c_val} {unit_str} (Fixed prior to 1983)"

            opts, correct_key = self._build_mcq_options(
                correct_text=correct_opt,
                distractors=[
                    (
                        distractor_1,
                        "Incorrectly marks defining constant as having experimental uncertainty",
                    ),
                    (distractor_2, "Outdated pre-2019 value"),
                    (distractor_3, "Incorrect claim of non-zero uncertainty budget"),
                ],
            )

            sample_id = f"drum_bench_const_mcq_{i + 1:03d}"
            samples.append(
                BenchmarkSample(
                    id=sample_id,
                    task=BenchmarkTask.CONSTANTS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTRODUCTORY
                    if "speed of light" in c.name.lower() or "planck" in c.name.lower()
                    else DifficultyTier.INTERMEDIATE,
                    question=q_text,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct_opt,
                    entity_uri=c.uri,
                    explanation=(
                        f"In the 2019 revision of the SI, the {c.name} ({c.symbol}) was given the exact value "
                        f"{c_val} {unit_str} by definition, fixing its standard uncertainty to exactly zero."
                    ),
                    metadata={"constant_symbol": c.symbol, "category": c.category.value},
                )
            )

            # Free-form variant
            ff_id = f"drum_bench_const_ff_{i + 1:03d}"
            samples.append(
                BenchmarkSample(
                    id=ff_id,
                    task=BenchmarkTask.CONSTANTS,
                    format=BenchmarkFormat.FREE_FORM,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=f"State the exact defined numerical value and unit of {c.name} ({c.symbol}) under the 2019 SI definition.",
                    ground_truth_answer=f"{c_val} {unit_str}",
                    entity_uri=c.uri,
                    explanation=f"{c.name} ({c.symbol}) = {c_val} {unit_str} (exact by 2019 SI definition).",
                    metadata={"constant_symbol": c.symbol},
                )
            )

        # 2. Non-exact CODATA constants with uncertainty
        for j, c in enumerate(recommended_constants[:count]):
            unit_str = c.unit_symbol or ""
            u_str = f" ± {c.standard_uncertainty}" if c.standard_uncertainty else ""
            q_text = (
                f"Which statement correctly describes the Newtonian constant of gravitation ({c.symbol})"
                if "gravitation" in c.name.lower()
                else f"Which statement correctly reflects the current CODATA status of the physical constant '{c.name}' ({c.symbol})?"
            )
            correct_opt = (
                f"{c.name} ({c.symbol}) is an experimentally determined constant with standard uncertainty "
                f"u = {c.standard_uncertainty or 'non-zero'} {unit_str}."
            )
            distractor_1 = f"{c.name} was defined as an exact fixed constant with u = 0 in the 2019 SI redefinition."
            distractor_2 = f"{c.name} is a dimensionless mathematical constant like pi and e."
            distractor_3 = f"{c.name} has a relative standard uncertainty exceeding 50% in current CODATA evaluations."

            opts, correct_key = self._build_mcq_options(
                correct_text=correct_opt,
                distractors=[
                    (
                        distractor_1,
                        "Confuses experimental CODATA constants with the 7 exact defining constants",
                    ),
                    (
                        distractor_2,
                        "Falsely identifies dimensional physical constant as mathematical constant",
                    ),
                    (distractor_3, "Grossly overstates experimental uncertainty"),
                ],
            )

            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_const_rec_{j + 1:03d}",
                    task=BenchmarkTask.CONSTANTS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.ADVANCED
                    if c.relative_uncertainty
                    else DifficultyTier.INTERMEDIATE,
                    question=q_text,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct_opt,
                    entity_uri=c.uri,
                    explanation=(
                        f"{c.name} is not one of the seven exact SI defining constants; it is subject to experimental "
                        f"determination recommended by CODATA with value {c.numeric_value}{u_str} {unit_str}."
                    ),
                    metadata={"constant_symbol": c.symbol},
                )
            )

        return samples[:count]

    # =========================================================================
    # Task 2: Dimensional Decomposition & Base SI
    # =========================================================================
    def generate_dimensions_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        derived_units = [
            u for u in self.entities.units.values() if not u.dimension_vector.is_dimensionless()
        ]

        selected_units = self.rng.sample(derived_units, min(len(derived_units), count))
        for idx, unit in enumerate(selected_units):
            dim_latex = unit.dimension_vector.to_latex()
            base_si_latex = unit.dimension_vector.to_si_base_unit_latex()
            sym = unit.symbol or unit.label

            # MCQ Question: Base SI decomposition
            q_text = f"What is the coherent SI base unit decomposition and dimensional formula for the unit '{unit.label}' ({sym})?"
            correct_opt = f"SI Base: {base_si_latex} | Dimension: {dim_latex}"

            # Craft distinct dimensional distractors
            # Distractor 1: Flip time power sign
            d1_vec = unit.dimension_vector.model_copy()
            if d1_vec.T != 0:
                d1_vec.T = -d1_vec.T
            else:
                d1_vec.L = d1_vec.L + 1
            distractor_1 = (
                f"SI Base: {d1_vec.to_si_base_unit_latex()} | Dimension: {d1_vec.to_latex()}"
            )

            # Distractor 2: Off-by-one on length power
            d2_vec = unit.dimension_vector.model_copy()
            d2_vec.L = d2_vec.L + 1 if d2_vec.L >= 0 else d2_vec.L - 1
            distractor_2 = (
                f"SI Base: {d2_vec.to_si_base_unit_latex()} | Dimension: {d2_vec.to_latex()}"
            )

            # Distractor 3: Missing mass or current dimension
            d3_vec = unit.dimension_vector.model_copy()
            if d3_vec.M != 0:
                d3_vec.M = 0
            elif d3_vec.I != 0:
                d3_vec.I = 0
            else:
                d3_vec.T = d3_vec.T - 1
            distractor_3 = (
                f"SI Base: {d3_vec.to_si_base_unit_latex()} | Dimension: {d3_vec.to_latex()}"
            )

            opts, correct_key = self._build_mcq_options(
                correct_text=correct_opt,
                distractors=[
                    (distractor_1, "Sign error in time or length dimension exponent"),
                    (distractor_2, "Off-by-one error in length dimension power"),
                    (distractor_3, "Omitted mass or electric current base dimension"),
                ],
            )

            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_dim_mcq_{idx + 1:03d}",
                    task=BenchmarkTask.DIMENSIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q_text,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct_opt,
                    entity_uri=unit.uri,
                    explanation=(
                        f"The unit {unit.label} ({sym}) has the 7-base ISQ dimension vector "
                        f"[L={unit.dimension_vector.L}, M={unit.dimension_vector.M}, T={unit.dimension_vector.T}, "
                        f"I={unit.dimension_vector.I}, Theta={unit.dimension_vector.Theta}, N={unit.dimension_vector.N}, J={unit.dimension_vector.J}], "
                        f"which corresponds to {base_si_latex}."
                    ),
                    metadata={"unit_symbol": sym, "dim_vector": unit.dimension_vector.model_dump()},
                )
            )

            # Free-form decomposition
            if idx % 2 == 0:
                samples.append(
                    BenchmarkSample(
                        id=f"drum_bench_dim_ff_{idx + 1:03d}",
                        task=BenchmarkTask.DIMENSIONS,
                        format=BenchmarkFormat.FREE_FORM,
                        difficulty=DifficultyTier.INTERMEDIATE,
                        question=f"Derive the SI base unit expression and ISQ dimension vector for the derived unit '{unit.label}' ({sym}).",
                        ground_truth_answer=f"{base_si_latex} ({dim_latex})",
                        entity_uri=unit.uri,
                        explanation=f"{sym} = {base_si_latex} in SI base units, with dimension {dim_latex}.",
                        metadata={"unit_symbol": sym},
                    )
                )

        return samples[:count]

    # =========================================================================
    # Task 3: Unit Conversions & Affine Transformations
    # =========================================================================
    def generate_conversions_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []

        # 1. Affine Temperature conversions (Special Metrological Class)
        temp_cases = [
            ("25.0 °C", "298.15 K", "T_K = T_C + 273.15 = 25.0 + 273.15 = 298.15 K", 25.0),
            ("100.0 °C", "373.15 K", "T_K = T_C + 273.15 = 100.0 + 273.15 = 373.15 K", 100.0),
            ("-40.0 °F", "233.15 K", "T_K = (-40 - 32) * 5/9 + 273.15 = 233.15 K", -40.0),
            ("68.0 °F", "293.15 K", "T_K = (68 - 32) * 5/9 + 273.15 = 293.15 K", 68.0),
            (
                "0.0 °C (interval Delta T = 10 °C)",
                "Delta T = 10 K",
                "Temperature differences Delta T in °C are equal to Delta T in K without offset (1 °C interval = 1 K interval).",
                10.0,
            ),
        ]

        for i, (val_in, val_out, expl, _num) in enumerate(temp_cases):
            q_text = f"Convert the temperature {val_in} to its exact thermodynamic temperature representation in Kelvin (K)."
            if "interval" in val_in:
                q_text = "A temperature change of Delta T = 10.0 °C occurs in an experiment. What is this temperature interval in Kelvin?"
                correct_opt = "10.0 K (Temperature intervals do not apply the 273.15 offset)"
                distractor_1 = "283.15 K (Applying T + 273.15 to a differential interval)"
                distractor_2 = "263.15 K"
                distractor_3 = "18.0 K"
            else:
                correct_opt = val_out
                try:
                    f_val = float(val_out.split()[0])
                    distractor_1 = f"{f_val - 273.15:.2f} K (Omitted 273.15 absolute offset)"
                    distractor_2 = f"{f_val * 1.8:.2f} K (Incorrect Rankine multiplier)"
                    distractor_3 = f"{f_val + 32.0:.2f} K (Erroneous Fahrenheit offset)"
                except Exception:
                    distractor_1 = "273.15 K"
                    distractor_2 = "0.0 K"
                    distractor_3 = "300.0 K"

            opts, correct_key = self._build_mcq_options(
                correct_text=correct_opt,
                distractors=[
                    (
                        distractor_1,
                        "Failure to apply or misapplying the affine 273.15 K zero-point offset",
                    ),
                    (distractor_2, "Incorrect dimensional scaling multiplier"),
                    (distractor_3, "Confusing Fahrenheit and Celsius conversion constants"),
                ],
            )

            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_conv_temp_{i + 1:03d}",
                    task=BenchmarkTask.CONVERSIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q_text,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct_opt,
                    explanation=expl,
                    metadata={"conversion_type": "affine_temperature"},
                )
            )

        # 2. Multiplicative unit conversions from QUDT/BIPM
        conv_units = [
            u
            for u in self.entities.units.values()
            if u.conversion and u.conversion.multiplier != 1.0
        ]
        selected_conv = self.rng.sample(conv_units, min(len(conv_units), count - len(temp_cases)))

        for j, u in enumerate(selected_conv):
            mult = u.conversion.multiplier
            sym = u.symbol or u.label
            q_text = f"Convert a measured quantity of 1.0 {sym} ({u.label}) to coherent SI units."
            correct_opt = f"{mult:.6e} SI base units (exact multiplier: {mult})"

            distractor_1 = f"{1.0 / mult:.6e} SI base units (Inverted conversion factor)"
            distractor_2 = f"{mult * 1000.0:.6e} SI base units (Order of magnitude error 10^3)"
            distractor_3 = f"{mult * 0.001:.6e} SI base units (Order of magnitude error 10^-3)"

            opts, correct_key = self._build_mcq_options(
                correct_text=correct_opt,
                distractors=[
                    (distractor_1, "Inverted ratio multiplier (dividing instead of multiplying)"),
                    (distractor_2, "Erroneous prefix factor of 1000x"),
                    (distractor_3, "Erroneous prefix factor of 1/1000x"),
                ],
            )

            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_conv_mult_{j + 1:03d}",
                    task=BenchmarkTask.CONVERSIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q_text,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct_opt,
                    entity_uri=u.uri,
                    explanation=f"According to BIPM/QUDT tables, 1 {sym} is defined as exactly {mult} coherent SI units.",
                    metadata={"unit_symbol": sym, "multiplier": mult},
                )
            )

        return samples[:count]

    # =========================================================================
    # Task 4: Error Detection & Dimensional Homogeneity
    # =========================================================================
    def generate_homogeneity_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []

        scenarios = [
            (
                "An engineering equation is proposed: X = 15 kg + 4.2 m. What is the fundamental metrological issue with this expression?",
                "Dimensional Inhomogeneity: Quantities of different physical dimensions (Mass [M] and Length [L]) cannot be added or subtracted.",
                [
                    (
                        "The coefficients must be converted to scientific notation first.",
                        "Misunderstanding of physical dimension rules",
                    ),
                    (
                        "Kilograms must be replaced by Newtons before addition.",
                        "Incorrect assumption that force resolves mass addition",
                    ),
                    (
                        "The expression is valid if 15 kg is divided by standard gravity.",
                        "Misapplication of gravitational conversion",
                    ),
                ],
                DifficultyTier.INTRODUCTORY,
            ),
            (
                "A sensor reports a capacitance value as '50 kmuF' (kilo-microfarads). According to BIPM SI rules, why is this notation prohibited?",
                "Compound prefixes are not permitted in SI; 'kmuF' must be written as 50 mF (millifarads) or 5e-2 F.",
                [
                    ("Farads cannot take metric prefixes.", "False rule regarding Farad prefixes"),
                    (
                        "Capacitance must only be reported in electrostatic units (statfarad).",
                        "Obsolete CGS unit preference",
                    ),
                    ("The symbol F must be lowercase in SI.", "Incorrect unit capitalization rule"),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "A mechanical torque sensor records 50 N*m. A software system converts this value directly into 50 Joules (J). Why is this metrologically problematic?",
                "Although Torque and Energy share identical dimensions [L^2 M T^-2], Torque is a vector cross-product quantity (N*m) and must never be expressed in Joules (J) to avoid confusing energy with rotational moment.",
                [
                    (
                        "Torque has dimension [L M T^-1] whereas Joule is [L^2 M T^-2].",
                        "Factually incorrect dimension claim",
                    ),
                    ("The Joule is not an SI derived unit.", "False claim about Joule SI status"),
                    (
                        "Torque can only be measured in foot-pounds.",
                        "Imperial bias against SI metric",
                    ),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "In the formula v^2 = u^2 + 2 * a * t, where v, u are velocities, a is acceleration, and t is time, check the dimensional homogeneity of the terms.",
                "Inhomogeneous: v^2 and u^2 have dimensions [L^2 T^-2], but the term 2*a*t has dimension [L T^-2 * T] = [L T^-1], which violates dimensional homogeneity.",
                [
                    (
                        "Homogeneous: All terms have dimension [L T^-1].",
                        "Failure to compute power of velocity",
                    ),
                    (
                        "Homogeneous: All terms have dimension [L^2 T^-2].",
                        "Failure to compute dimension of acceleration * time",
                    ),
                    (
                        "Inhomogeneous: The constant 2 carries a dimension of Length.",
                        "Incorrectly assigning dimension to pure scalar",
                    ),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
        ]

        for i, (q, correct, dists, diff) in enumerate(scenarios * (count // len(scenarios) + 1)):
            if len(samples) >= count:
                break
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_homogeneity_{i + 1:03d}",
                    task=BenchmarkTask.HOMOGENEITY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=diff,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=f"Metrological principle: {correct}",
                    metadata={"task_type": "homogeneity_audit"},
                )
            )

        return samples[:count]

    # =========================================================================
    # Task 5: SI Typography & Metrological Conventions
    # =========================================================================
    def generate_conventions_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []

        rules = [
            (
                "According to the BIPM SI Brochure (9th Edition), which of the following represents the correct formatting for a numerical value and unit symbol?",
                "25.4 mm (with a non-breaking space separating the number and the unit symbol)",
                [
                    (
                        "25.4mm (without space between number and unit)",
                        "Violates BIPM Section 5.4.3 spacing rule",
                    ),
                    (
                        "25.4 MM (capitalized millimeter)",
                        "Violates prefix and unit case conventions",
                    ),
                    (
                        "25.4-mm (hyphenated value and unit in normal prose)",
                        "Violates SI style guide",
                    ),
                ],
                "BIPM SI Brochure Section 5.4.3 requires a space between the numerical value and the unit symbol (e.g. 25.4 mm), except for superscript degree, minute, second angles.",
            ),
            (
                "When writing the full English name of an SI unit named after a scientist (e.g. Newton, Kelvin, Watt), what is the mandatory capitalization rule?",
                "The unit name is written in all lowercase (e.g. newton, kelvin, watt), but the symbol is capitalized (N, K, W).",
                [
                    (
                        "Both the unit name and symbol are capitalized (e.g. Newton, N).",
                        "Confuses person name with unit name",
                    ),
                    (
                        "Both the unit name and symbol are lowercase (e.g. newton, n).",
                        "Violates symbol capitalization rule for person names",
                    ),
                    (
                        "Unit names are capitalized only at the beginning of a sentence or formula.",
                        "Incomplete rule statement",
                    ),
                ],
                "BIPM SI Brochure Section 5.2 establishes that unit names are common nouns and written in lowercase (newton, pascal), while symbols for units named after persons are capitalized (N, Pa).",
            ),
            (
                "How should physical quantities vs. unit symbols be typeset in LaTeX according to ISO 80000-1?",
                "Physical quantity symbols are typeset in italic (e.g., $m$ for mass, $v$ for velocity), whereas unit symbols are typeset in upright Roman (e.g., \\text{kg}, \\text{m/s}).",
                [
                    (
                        "Both quantities and units must be typeset in italic font.",
                        "Violates ISO 80000 font distinction",
                    ),
                    (
                        "Both quantities and units must be typeset in upright Roman font.",
                        "Lacks distinction between variable and unit",
                    ),
                    (
                        "Units are italicized and variables are bolded.",
                        "Incorrect typography mapping",
                    ),
                ],
                "ISO 80000-1 and BIPM mandate italic font for variables/quantities and upright Roman for unit symbols and mathematical operators.",
            ),
            (
                "What is the correct SI representation of the product of two units, such as newton and meter?",
                "N*m or N m (using a half-high dot or space to prevent confusion with prefixes like nm for nanometer)",
                [
                    ("Nm without space or dot", "Ambiguous with nm (nanometer)"),
                    ("N/m", "Represents quotient instead of product"),
                    ("N_m with underscore", "Invalid typographical notation"),
                ],
                "A dot or space is required to distinguish compound products from metric prefixes.",
            ),
        ]

        for i, (q, correct, dists, expl) in enumerate(rules * (count // len(rules) + 1)):
            if len(samples) >= count:
                break
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_conv_rule_{i + 1:03d}",
                    task=BenchmarkTask.CONVENTIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=expl,
                    metadata={"standard": "BIPM_SI_9th_Ed"},
                )
            )

        return samples[:count]

    # =========================================================================
    # Task 6: Metrological Uncertainty (GUM) & Sig-Figs
    # =========================================================================
    def generate_uncertainty_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []

        scenarios = [
            (
                "A measurement is reported as y = (100.00 ± 0.05) g with coverage factor k = 2. What is the standard combined uncertainty u_c(y)?",
                "u_c(y) = 0.05 / 2 = 0.025 g",
                [
                    (
                        "u_c(y) = 0.05 * 2 = 0.10 g",
                        "Multiplying instead of dividing by coverage factor k",
                    ),
                    (
                        "u_c(y) = 0.05 g",
                        "Confusing expanded uncertainty U with standard uncertainty u_c",
                    ),
                    ("u_c(y) = 0.0005 g", "Arithmetic magnitude error"),
                ],
                "By JCGM 100:2008 (GUM), the expanded uncertainty U = k * u_c. Therefore, u_c = U / k = 0.05 / 2 = 0.025 g.",
            ),
            (
                "Two independent quantities A = (10.0 ± 0.3) m and B = (20.0 ± 0.4) m are summed: C = A + B. What is the combined standard uncertainty u_c(C)?",
                "u_c(C) = sqrt((0.3)^2 + (0.4)^2) = sqrt(0.09 + 0.16) = sqrt(0.25) = 0.5 m",
                [
                    (
                        "u_c(C) = 0.3 + 0.4 = 0.7 m",
                        "Linear sum instead of root-sum-square for independent variables",
                    ),
                    ("u_c(C) = (0.3 * 0.4) = 0.12 m", "Product of uncertainties"),
                    ("u_c(C) = sqrt(0.3 + 0.4) = 0.837 m", "Root sum without squaring variances"),
                ],
                "GUM uncertainty propagation for independent additive variables follows u_c(A+B) = sqrt(u(A)^2 + u(B)^2).",
            ),
            (
                "What is the relative standard uncertainty u_r for a length measurement L = 50.00 m with standard uncertainty u(L) = 0.02 m?",
                "u_r(L) = 0.02 / 50.00 = 0.0004 = 4.0e-4 (or 0.04%)",
                [
                    ("u_r(L) = 50.00 * 0.02 = 1.0 m", "Multiplying value and uncertainty"),
                    (
                        "u_r(L) = 0.02 m",
                        "Reporting absolute uncertainty instead of dimensionless relative ratio",
                    ),
                    ("u_r(L) = 4.0e-2 (4%)", "Order of magnitude conversion error"),
                ],
                "Relative standard uncertainty is defined as u_r(x) = u(x) / |x|.",
            ),
        ]

        for i, (q, correct, dists, expl) in enumerate(scenarios * (count // len(scenarios) + 1)):
            if len(samples) >= count:
                break
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_uncert_{i + 1:03d}",
                    task=BenchmarkTask.UNCERTAINTY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.ADVANCED,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=expl,
                    metadata={"standard": "JCGM_100_GUM"},
                )
            )

        return samples[:count]

    # =========================================================================
    # Helpers
    # =========================================================================
    def _build_mcq_options(
        self,
        correct_text: str,
        distractors: list[tuple[str, str]],
    ) -> tuple[list[MCQOption], str]:
        """Randomly places correct answer and 3 distractors among keys A, B, C, D."""
        all_items = [(correct_text, True, "Ground truth")] + [
            (text, False, rationale) for text, rationale in distractors[:3]
        ]
        self.rng.shuffle(all_items)

        keys = ["A", "B", "C", "D"]
        options = []
        correct_key = "A"
        for k, (txt, is_corr, rat) in zip(keys, all_items, strict=True):
            if is_corr:
                correct_key = k
            options.append(
                MCQOption(
                    key=k,
                    text=txt,
                    is_correct=is_corr,
                    distractor_rationale=rat,
                )
            )
        return options, correct_key

    def save_jsonl(self, samples: list[BenchmarkSample], filepath: str | Path) -> None:
        """Save benchmark samples to a JSONL file."""
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")
