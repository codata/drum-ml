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

        # Generate exact SI defining constant questions
        for _i, c in enumerate(exact_constants[:count]):
            c_val = c.numeric_value
            unit_str = c.unit_symbol or ""
            sym_display = f" ({c.symbol})" if c.symbol else ""

            q_text = (
                f"Under the 2019 SI redefinition, what is the exact defined numerical value "
                f"of the {c.name}{sym_display}?"
            )
            correct_opt = f"{c_val} {unit_str} (Exact, standard uncertainty u = 0)".strip()

            # Distractors
            try:
                val_float = float(c_val)
                distractor_1 = (
                    f"{val_float * 1.0001:.7e} {unit_str} (Experimental with u_r = 1.2e-8)".strip()
                )
                distractor_2 = f"{val_float * 0.9990:.7e} {unit_str} (Pre-2019 value)".strip()
                distractor_3 = f"{c_val} {unit_str} (Recommended value with u = 4.5e-9)".strip()
            except Exception:
                distractor_1 = f"{c_val}01 {unit_str} (Uncertainty u_r = 1.0e-7)".strip()
                distractor_2 = f"{c_val} (Approximate value)".strip()
                distractor_3 = f"{c_val} {unit_str} (Fixed prior to 1983)".strip()

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

            sample_id = f"drum_bench_const_mcq_{len(samples) + 1:03d}"
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
                        f"In the 2019 revision of the SI, the {c.name}{sym_display} was given the exact value "
                        f"{c_val} {unit_str} by definition, fixing its standard uncertainty to exactly zero."
                    ),
                    metadata={"constant_symbol": c.symbol, "category": c.category.value},
                )
            )

            # Free-form variant
            if len(samples) < count:
                ff_id = f"drum_bench_const_ff_{len(samples) + 1:03d}"
                samples.append(
                    BenchmarkSample(
                        id=ff_id,
                        task=BenchmarkTask.CONSTANTS,
                        format=BenchmarkFormat.FREE_FORM,
                        difficulty=DifficultyTier.INTERMEDIATE,
                        question=f"State the exact defined numerical value and unit of {c.name}{sym_display} under the 2019 SI definition.",
                        ground_truth_answer=f"{c_val} {unit_str}".strip(),
                        entity_uri=c.uri,
                        explanation=f"{c.name}{sym_display} = {c_val} {unit_str} (exact by 2019 SI definition).",
                        metadata={"constant_symbol": c.symbol},
                    )
                )

        # 2. Non-exact CODATA constants with uncertainty (fill up to count)
        remaining = max(0, count - len(samples))
        for j, c in enumerate(recommended_constants[:remaining]):
            unit_str = c.unit_symbol or ""
            sym_display = f" ({c.symbol})" if c.symbol else ""
            u_str = f" ± {c.standard_uncertainty}" if c.standard_uncertainty else ""
            q_text = (
                f"Which statement correctly describes the Newtonian constant of gravitation{sym_display}"
                if "gravitation" in c.name.lower()
                else f"Which statement correctly reflects the current CODATA status of the physical constant '{c.name}'{sym_display}?"
            )
            correct_opt = (
                f"{c.name}{sym_display} is an experimentally determined constant with standard uncertainty "
                f"u = {c.standard_uncertainty or 'non-zero'} {unit_str}.".strip()
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
                    metadata={
                        "constant_symbol": c.symbol,
                        "standard_uncertainty": c.standard_uncertainty,
                    },
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

        return samples  # =========================================================================

    # Task 4: Error Detection & Dimensional Homogeneity (100% Unique Procedural Generator)
    # =========================================================================
    def generate_homogeneity_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        used_questions: set[str] = set()

        # 1. Physics Formulas with Dimensional Violations / Homogeneity Checks
        physics_formulas = [
            (
                "Consider the proposed energy equation: E = m * c^3, where m is mass and c is the speed of light in vacuum. Evaluate its dimensional homogeneity.",
                "Inhomogeneous: Mass times velocity cubed has dimension [L^3 M T^-3], whereas Energy has dimension [L^2 M T^-2].",
                [
                    (
                        "Homogeneous: Both sides have dimension [L^2 M T^-2].",
                        "Miscalculating velocity power",
                    ),
                    (
                        "Homogeneous: c is dimensionless so [E] = [M].",
                        "Falsely claiming speed of light is dimensionless",
                    ),
                    (
                        "Inhomogeneous: Mass cannot be converted to energy in SI.",
                        "False claim rejecting mass-energy equivalence",
                    ),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "In kinematics, an engineer writes: v^2 = u^2 + 2 * a * t, where v, u are velocities, a is acceleration, and t is time. Is this equation dimensionally homogeneous?",
                "Inhomogeneous: v^2 and u^2 have dimension [L^2 T^-2], while 2*a*t has dimension [L T^-2 * T] = [L T^-1].",
                [
                    (
                        "Homogeneous: All terms have dimension [L^2 T^-2].",
                        "Failing to compute dimension of acceleration * time",
                    ),
                    (
                        "Homogeneous: All terms have dimension [L T^-1].",
                        "Confusing velocity squared with linear velocity",
                    ),
                    (
                        "Inhomogeneous: The factor 2 carries a dimension of Length [L].",
                        "Incorrectly assigning dimension to pure scalar",
                    ),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "A proposed gravitational force formula states: F = G * (m1 * m2) / r, where m1, m2 are masses, r is separation distance, and G is the gravitational constant. Is this dimensionally valid?",
                "Inhomogeneous: With denominator r (instead of r^2), the right side has dimension [L^2 M T^-2] (Energy), not [L M T^-2] (Force).",
                [
                    (
                        "Homogeneous: The equation represents Newtonian force correctly.",
                        "Failing to notice linear distance power",
                    ),
                    (
                        "Homogeneous: Both sides have dimension [L M T^-2].",
                        "Incorrect dimensional arithmetic",
                    ),
                    (
                        "Inhomogeneous: G is a dimensionless constant.",
                        "Falsely assuming G is dimensionless",
                    ),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "An electrical power formula is written as P = I^2 / R, where I is electric current and R is resistance. What is the dimensional error?",
                "Inhomogeneous: I^2 / R has dimension [I^2 / (L^2 M T^-3 I^-2)] = [L^-2 M^-1 T^3 I^4] (Conductance squared), whereas Power is [L^2 M T^-3] (correct formula is P = I^2 * R).",
                [
                    (
                        "Homogeneous: Power is always current squared divided by resistance.",
                        "Confusing I^2 * R with V^2 / R",
                    ),
                    (
                        "Homogeneous: Both sides have dimension [L^2 M T^-3].",
                        "Failing to divide dimensions correctly",
                    ),
                    (
                        "Inhomogeneous: Electric current is not an SI base dimension.",
                        "Rejecting ampere as SI base dimension",
                    ),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "A student calculates pendulum period as T = 2 * pi * sqrt(g / L), where g is gravitational acceleration and L is string length. Why is this expression dimensionally flawed?",
                "Inhomogeneous: sqrt(g / L) has dimension sqrt([L T^-2] / [L]) = [T^-1] (frequency), whereas Period T must have dimension [T] (time). The correct formula is 2*pi*sqrt(L / g).",
                [
                    (
                        "Homogeneous: Both sides have dimension of Time [T].",
                        "Failing to invert the dimensional ratio",
                    ),
                    (
                        "Homogeneous: 2*pi carries a dimension of Time.",
                        "Assigning dimension to pure angle/scalar",
                    ),
                    (
                        "Inhomogeneous: g is dimensionless at Earth sea level.",
                        "Claiming standard gravity is dimensionless",
                    ),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "A proposed acoustic wave intensity formula is I = 0.5 * rho * v * f, where rho is fluid density [M L^-3], v is wave speed [L T^-1], and f is frequency [T^-1]. What is the dimensional status?",
                "Inhomogeneous: The product gives dimension [M L^-2 T^-2], whereas acoustic Intensity (Power per unit Area) has dimension [M T^-3]. (Intensity scales with f^2 and amplitude squared).",
                [
                    (
                        "Homogeneous: Power per unit area is [M L^-2 T^-2].",
                        "Incorrect intensity dimension",
                    ),
                    (
                        "Homogeneous: All acoustic intensity formulas are dimensionless.",
                        "False claim regarding wave intensity",
                    ),
                    (
                        "Inhomogeneous: Density carries no time component.",
                        "Misunderstanding dimensional makeup",
                    ),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "A capacitor energy formula is proposed as E = 0.5 * C * V, where C is capacitance and V is electric potential. Evaluate this expression.",
                "Inhomogeneous: C * V has dimension of Electric Charge [I T] (Coulombs), not Energy [L^2 M T^-2] (Joules). The correct energy formula is 0.5 * C * V^2.",
                [
                    (
                        "Homogeneous: C * V represents energy in electrostatic units.",
                        "Confusing charge Q = CV with energy E = 1/2 CV^2",
                    ),
                    (
                        "Homogeneous: Both sides have dimension [L^2 M T^-2].",
                        "Failing to square voltage dimension",
                    ),
                    (
                        "Inhomogeneous: Capacitance is a dimensionless ratio.",
                        "False claim about capacitance",
                    ),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "An engineer calculates fluid Bernoulli head as H = p + 0.5 * rho * v + rho * g * h, where p is pressure, rho is density, v is velocity, g is gravity, and h is height. What is the dimensional error?",
                "Inhomogeneous: The dynamic pressure term 0.5*rho*v has dimension [M L^-2 T^-1] instead of Pressure [M L^-1 T^-2] because velocity must be squared (0.5*rho*v^2).",
                [
                    (
                        "Homogeneous: All terms have dimension of Pressure [M L^-1 T^-2].",
                        "Failing to square velocity",
                    ),
                    (
                        "Homogeneous: All terms have dimension of Energy [L^2 M T^-2].",
                        "Confusing pressure with total energy",
                    ),
                    (
                        "Inhomogeneous: Density times gravity is dimensionless.",
                        "False claim regarding specific weight",
                    ),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "A magnetic energy density equation is written as u_B = B^2 / (2 * mu_0), where B is magnetic flux density [M T^-2 I^-1] and mu_0 is magnetic permeability [L M T^-2 I^-2]. Is this dimensionally homogeneous?",
                "Homogeneous: [B^2 / mu_0] = [M^2 T^-4 I^-2] / [L M T^-2 I^-2] = [M L^-1 T^-2] = [L^2 M T^-2] / [L^3], which correctly represents Energy per unit Volume (Energy Density).",
                [
                    (
                        "Inhomogeneous: The result has dimension of Force [L M T^-2].",
                        "Incorrect dimensional division",
                    ),
                    (
                        "Inhomogeneous: mu_0 is dimensionless in SI.",
                        "Confusing SI permeability with relative permeability",
                    ),
                    (
                        "Inhomogeneous: B^2 carries dimension of Current squared.",
                        "Incorrect magnetic dimension power",
                    ),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "A thermal conduction heat rate formula is proposed: q = -k * A * (dT / dx)^2, where k is thermal conductivity, A is area, and dT/dx is temperature gradient. Evaluate its dimensional validity.",
                "Inhomogeneous: Squaring the temperature gradient produces extra [Theta L^-1] factor, yielding dimension [M T^-3 Theta], whereas Heat Rate (Power) is [L^2 M T^-3]. (Fourier's law is linear: q = -k A dT/dx).",
                [
                    (
                        "Homogeneous: Heat flow rate always scales quadratically with temperature gradient.",
                        "Falsely asserting quadratic Fourier law",
                    ),
                    (
                        "Homogeneous: Both sides have dimension of Power [L^2 M T^-3].",
                        "Failing to track temperature power",
                    ),
                    ("Inhomogeneous: Area carries no length dimension.", "False claim about area"),
                ],
                DifficultyTier.ADVANCED,
            ),
        ]

        for q, correct, dists, diff in physics_formulas:
            if q in used_questions:
                continue
            used_questions.add(q)
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_homogeneity_{len(samples) + 1:03d}",
                    task=BenchmarkTask.HOMOGENEITY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=diff,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=f"Dimensional analysis: {correct}",
                    metadata={"task_type": "formula_homogeneity"},
                )
            )

        # 2. Quantity Kind Discriminators (Same Dimension Vector, Different Physical Meaning)
        qk_discriminators = [
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
                "A radiation sensor reports an activity of 100 Bq (becquerel). A technician replaces the unit with 100 Hz (hertz) since both are 1/s. Why does BIPM forbid this interchange?",
                "BIPM specifies that Hertz (Hz) is reserved strictly for periodic phenomena, whereas Becquerel (Bq) is designated exclusively for stochastic radioactive decays, preventing hazardous ambiguity.",
                [
                    (
                        "Hertz has dimension [T^-2] while Becquerel is [T^-1].",
                        "Incorrect dimension for Hertz",
                    ),
                    ("Becquerel is not an official SI unit.", "False claim regarding SI status"),
                    ("1 Bq is equal to 3.7e10 Hz.", "Confusing Becquerel with Curie conversion"),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "In medical physics, a dose of ionizing radiation is measured as 2 Gy (gray). A report writes this as 2 Sv (sievert). What is the critical metrological and safety violation?",
                "Gray (Gy) measures Absorbed Dose (physical energy J/kg), whereas Sievert (Sv) measures Equivalent/Effective Dose weighted by biological damage factors (Q/W_R); substituting them conflates physical energy with biological risk.",
                [
                    (
                        "Gray has dimension [L^2 T^-2] while Sievert is dimensionless.",
                        "False claim that Sievert is dimensionless",
                    ),
                    (
                        "1 Gy is defined as exactly 100 Sv.",
                        "Confusing Gray with old Rad/Rem factors",
                    ),
                    (
                        "Sievert is only applicable to electromagnetic waves, not particles.",
                        "Incorrect radiation applicability",
                    ),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "A fluid mechanics software reports Dynamic Viscosity in m^2/s instead of Pa*s. What is the fundamental dimensional discrepancy?",
                "Dynamic Viscosity has dimension [M L^-1 T^-1] (Pa*s), whereas m^2/s is Kinematic Viscosity [L^2 T^-1] (dynamic viscosity divided by density).",
                [
                    (
                        "Both dynamic and kinematic viscosity have identical dimension [L^2 T^-1].",
                        "Failing to distinguish dynamic from kinematic viscosity",
                    ),
                    (
                        "Dynamic viscosity is dimensionless in SI.",
                        "False claim regarding viscosity dimension",
                    ),
                    ("Pa*s has dimension [M L T^-2].", "Incorrect dimensional formula for Pa*s"),
                ],
                DifficultyTier.ADVANCED,
            ),
            (
                "An optical system compares Luminous Flux (lumen, lm) and Illuminance (lux, lx). What distinguishes their dimensional composition?",
                "Luminous Flux (lm = cd * sr) measures total emitted visible light, whereas Illuminance (lx = lm / m^2) measures luminous flux incident per unit area [J L^-2].",
                [
                    (
                        "Lumen and lux are identical coherent units.",
                        "Confusing flux with flux density",
                    ),
                    (
                        "Lux has dimension [J], whereas Lumen is [J L^2].",
                        "Inverted area dependency",
                    ),
                    ("Candela is derived from lux and lumen.", "Inverting base and derived status"),
                ],
                DifficultyTier.INTERMEDIATE,
            ),
        ]

        for q, correct, dists, diff in qk_discriminators:
            if q in used_questions:
                continue
            used_questions.add(q)
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_homogeneity_{len(samples) + 1:03d}",
                    task=BenchmarkTask.HOMOGENEITY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=diff,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=f"Quantity kind discrimination: {correct}",
                    metadata={"task_type": "quantity_kind_discrimination"},
                )
            )

        # 3. Compound Metric Prefix Violations
        prefix_cases = [
            ("50 kmuF", "kilo-microfarads", "50 mF (millifarads) or 5e-2 F", "Farad (F)"),
            ("12 mkg", "milli-kilograms", "12 g (grams)", "Gram / Kilogram"),
            ("100 Mns", "mega-nanoseconds", "100 ms (milliseconds)", "Second (s)"),
            ("25 muA", "micro-kiloamperes", "25 mA (milliamperes)", "Ampere (A)"),
            ("8 GpF", "giga-picofarads", "8 mF (millifarads)", "Farad (F)"),
            ("150 kmm", "kilo-millimeters", "150 m (meters)", "Meter (m)"),
            ("40 cdm", "centi-decimeters", "4 mm (millimeters)", "Meter (m)"),
            ("500 mmuA", "milli-microamperes", "500 nA (nanoamperes)", "Ampere (A)"),
        ]

        for notation, full_name, valid_replacement, unit_type in prefix_cases:
            q = f"A laboratory instrument logs a measurement as '{notation}' ({full_name}). According to BIPM SI Brochure rules, why is this notation strictly prohibited?"
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = f"Compound metric prefixes are forbidden in SI; '{notation}' must be written using a single prefix as {valid_replacement}."
            dists = [
                (
                    f"The metric prefix '{notation.split()[1][:2]}' cannot be applied to {unit_type}.",
                    "False claim about unit prefix eligibility",
                ),
                (f"{unit_type} must only be expressed in CGS units.", "Obsolete CGS preference"),
                (
                    f"The notation is valid only if written with a hyphen: '{notation.replace(' ', '-')}'",
                    "Incorrect hyphenation rule",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_homogeneity_{len(samples) + 1:03d}",
                    task=BenchmarkTask.HOMOGENEITY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="BIPM SI Brochure Section 3.1 states: Compound prefix symbols, that is, prefix symbols formed by the juxtaposition of two or more prefix symbols, are not permitted.",
                    metadata={"task_type": "compound_prefix_violation", "notation": notation},
                )
            )

        # 4. Procedural Incompatible Dimension Addition/Subtraction pairs (fill to count)
        derived_units = [
            u
            for u in self.entities.units.values()
            if not u.dimension_vector.is_dimensionless() and u.symbol and len(u.symbol) < 15
        ]
        shuffled_units = list(derived_units)
        self.rng.shuffle(shuffled_units)

        for i in range(len(shuffled_units) - 1):
            if len(samples) >= count:
                break
            u1 = shuffled_units[i]
            u2 = shuffled_units[i + 1]
            if u1.dimension_vector.model_dump() == u2.dimension_vector.model_dump():
                continue

            c1 = round(self.rng.uniform(2.0, 50.0), 1)
            c2 = round(self.rng.uniform(1.5, 30.0), 1)
            op = self.rng.choice(["+", "-"])

            q = f"An engineering equation is proposed: X = {c1} {u1.symbol} {op} {c2} {u2.symbol}. What is the fundamental metrological issue with this expression?"
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = (
                f"Dimensional Inhomogeneity: Quantities of different physical dimensions "
                f"('{u1.label}' [{u1.dimension_vector.to_latex()}] and '{u2.label}' [{u2.dimension_vector.to_latex()}]) "
                f"cannot be added or subtracted."
            )
            dists = [
                (
                    f"The expression is valid, but the result must be expressed in {u1.symbol}.",
                    "Incorrectly assuming first term dictates dimension",
                ),
                (
                    f"The coefficients must be converted to base SI units first ({c1} and {c2} are non-SI).",
                    "Confusing scalar coefficients with dimensional units",
                ),
                (
                    f"The operation is permissible if {c1} is multiplied by standard gravity g.",
                    "Arbitrary gravitational conversion assumption",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_homogeneity_{len(samples) + 1:03d}",
                    task=BenchmarkTask.HOMOGENEITY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTRODUCTORY,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="Fourier's Principle of Dimensional Homogeneity mandates that all terms in a sum must share identical dimensions.",
                    metadata={
                        "task_type": "additive_inhomogeneity",
                        "u1": u1.symbol,
                        "u2": u2.symbol,
                    },
                )
            )

        return samples[:count]

    # =========================================================================
    # Task 5: SI Typography & Metrological Conventions (100% Unique Procedural Generator)
    # =========================================================================
    def generate_conventions_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        used_questions: set[str] = set()

        # 1. Person-Named Units (Capital Symbol vs Lowercase Name)
        person_units = [
            ("newton", "N", "force", "Sir Isaac Newton"),
            ("joule", "J", "energy", "James Prescott Joule"),
            ("pascal", "Pa", "pressure", "Blaise Pascal"),
            ("watt", "W", "power", "James Watt"),
            ("volt", "V", "electric potential", "Alessandro Volta"),
            ("ohm", "Ω", "electrical resistance", "Georg Simon Ohm"),
            ("kelvin", "K", "thermodynamic temperature", "Lord Kelvin"),
            ("hertz", "Hz", "frequency", "Heinrich Hertz"),
            ("coulomb", "C", "electric charge", "Charles-Augustin de Coulomb"),
            ("farad", "F", "capacitance", "Michael Faraday"),
            ("tesla", "T", "magnetic flux density", "Nikola Tesla"),
            ("weber", "Wb", "magnetic flux", "Wilhelm Eduard Weber"),
            ("henry", "H", "inductance", "Joseph Henry"),
            ("siemens", "S", "electrical conductance", "Werner von Siemens"),
            ("becquerel", "Bq", "radioactivity", "Henri Becquerel"),
            ("gray", "Gy", "absorbed dose", "Louis Harold Gray"),
            ("sievert", "Sv", "dose equivalent", "Rolf Sievert"),
            ("ampere", "A", "electric current", "André-Marie Ampère"),
        ]

        for name, sym, quantity_kind, person in person_units:
            q = f"When writing the SI unit for {quantity_kind} named after {person}, what is the mandatory capitalization rule for its full English name vs. its symbol?"
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = f"The full unit name is written in all lowercase ('{name}'), but its symbol is capitalized ('{sym}')."
            dists = [
                (
                    f"Both the full unit name and symbol are capitalized ('{name.capitalize()}', '{sym}').",
                    "Confuses person name with unit common noun",
                ),
                (
                    f"Both the full unit name and symbol are lowercase ('{name}', '{sym.lower()}').",
                    "Violates person-named symbol capitalization",
                ),
                (
                    f"The unit name is capitalized ('{name.capitalize()}'), but the symbol is lowercase ('{sym.lower()}').",
                    "Inverts standard convention",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_conventions_{len(samples) + 1:03d}",
                    task=BenchmarkTask.CONVENTIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTRODUCTORY,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="BIPM SI Brochure Section 5.2: Unit names are common nouns and start with a lowercase letter (e.g. newton), while symbols for units named after persons start with a capital letter (e.g. N).",
                    metadata={"unit_name": name, "symbol": sym},
                )
            )

        # 2. Metric Prefix Case Sensitivity & Ambiguity Rules
        prefix_pairs = [
            ("M", "mega (10^6)", "m", "milli (10^-3)", "MW vs mW (megawatt vs milliwatt)"),
            ("P", "peta (10^15)", "p", "pico (10^-12)", "PJ vs pJ (petajoule vs picojoule)"),
            ("G", "giga (10^9)", "g", "gram (base mass unit)", "GHz vs g (gigahertz vs gram)"),
            ("E", "exa (10^18)", "d", "deci (10^-1)", "EV vs dV (exavolt vs decivolt)"),
            ("Z", "zetta (10^21)", "z", "zepto (10^-21)", "ZB vs zB (zettabyte vs zeptobyte)"),
            ("Y", "yotta (10^24)", "y", "yocto (10^-24)", "Ym vs ym (yottameter vs yoctometer)"),
            (
                "T",
                "tera (10^12)",
                "t",
                "tonne (metric ton = 1000 kg)",
                "TH vs t (terahertz vs tonne)",
            ),
            (
                "K",
                "kelvin (unit)",
                "k",
                "kilo (10^3 prefix)",
                "K vs k (kelvin temperature vs kilo prefix)",
            ),
        ]

        for p_upper, u_name, p_lower, l_name, example in prefix_pairs:
            q = f"In SI metric prefixes, what is the critical distinction between the uppercase prefix symbol '{p_upper}' and the lowercase symbol '{p_lower}' ({example})?"
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = f"Uppercase '{p_upper}' represents {u_name}, whereas lowercase '{p_lower}' represents {l_name}; confusing case causes an astronomical factor error."
            dists = [
                (
                    f"'{p_upper}' and '{p_lower}' are interchangeable case variants of the same multiplier.",
                    "Ignoring case sensitivity in SI prefixes",
                ),
                (
                    f"'{p_upper}' is the American NIST spelling, while '{p_lower}' is the European BIPM spelling.",
                    "False geographic standard claim",
                ),
                (
                    f"'{p_upper}' is only used in computer science, whereas '{p_lower}' is used in physics.",
                    "Domain confusion",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_conventions_{len(samples) + 1:03d}",
                    task=BenchmarkTask.CONVENTIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="BIPM Section 3.1: SI prefix symbols are strictly case-sensitive to differentiate large multiplying factors from submultiple factors.",
                    metadata={"prefix_upper": p_upper, "prefix_lower": p_lower},
                )
            )

        # 3. ISO 80000 & BIPM Formatting & Typography Rules
        formatting_rules = [
            (
                "According to BIPM SI Brochure (9th Edition) Section 5.4.3, how should a numerical value and unit symbol be spaced in text?",
                "25.4 mm (with a non-breaking space separating the numerical value and the unit symbol)",
                [
                    (
                        "25.4mm (without any space between number and unit)",
                        "Violates mandatory spacing rule",
                    ),
                    (
                        "25.4-mm (with a hyphen connecting number and unit in prose)",
                        "Violates SI style guide against hyphenated units",
                    ),
                    ("25.4_mm (with an underscore)", "Invalid typographical character"),
                ],
                "BIPM Section 5.4.3: The numerical value always precedes the unit, and a space is always used to separate the unit from the number.",
                DifficultyTier.INTRODUCTORY,
            ),
            (
                "What is the sole exception to the BIPM rule requiring a space between a numerical value and its unit symbol?",
                "The plane angle units: degree (°), minute (′), and second (″) (e.g. 30°22′15″ written without spaces)",
                [
                    (
                        "Temperature in degrees Celsius (e.g. 25°C without space)",
                        "Degrees Celsius requires a space: 25 °C",
                    ),
                    (
                        "Percentage values (e.g. 50% without space)",
                        "BIPM/ISO treats % as dimensionless with space: 50 %",
                    ),
                    (
                        "Time in seconds (e.g. 10s without space)",
                        "Time units require a space: 10 s",
                    ),
                ],
                "BIPM Section 5.4.3: The only exceptions are the symbols for plane angle: degree, minute, and second.",
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "How must physical quantity variables (e.g. mass, length, velocity) vs. unit symbols (e.g. kg, m, s) be formatted in mathematical expressions according to ISO 80000-1?",
                "Quantity variables are formatted in italic type ($m, l, v$), whereas unit symbols and mathematical operators are formatted in upright Roman type (\\text{kg}, \\text{m}, \\text{s}).",
                [
                    (
                        "Both variables and units must be formatted in italic type ($m, kg, s$).",
                        "Violates font distinction rule",
                    ),
                    (
                        "Both variables and units must be formatted in upright Roman type.",
                        "Prevents distinguishing variable from unit",
                    ),
                    (
                        "Variables are in bold type and units are in italic type.",
                        "Incorrect font mapping",
                    ),
                ],
                "ISO 80000-1 & BIPM Chapter 5 mandate italic font for physical variables and upright Roman for units.",
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "What is the BIPM rule regarding the use of multiple solidi (slashes) in compound derived unit expressions (e.g. acceleration or thermal conductivity)?",
                "A solidus (/) must never be used more than once in an expression without parentheses (e.g. write m/s^2 or m*s^-2, never m/s/s).",
                [
                    (
                        "Multiple solidi are encouraged to show sequential divisions (e.g. J/kg/K).",
                        "Creates mathematical ambiguity",
                    ),
                    (
                        "A solidus is strictly forbidden anywhere in SI; only negative exponents are allowed.",
                        "Solidus is permitted if used at most once",
                    ),
                    (
                        "Up to three solidi are permitted in fluid mechanics expressions.",
                        "False discipline exemption",
                    ),
                ],
                "BIPM Section 5.4.2: A solidus must not be used more than once without parentheses to remove ambiguity.",
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "How should a range of physical quantities with units be formatted according to SI style guides?",
                "'10 m to 20 m' or '(10 to 20) m' (the unit must be explicit for each number or enclose the range in parentheses)",
                [
                    (
                        "'10-20 m' (attaching the unit only to the final number)",
                        "Leaves first number ambiguous without units",
                    ),
                    (
                        "'10m - 20m' (omitting spaces and using a hyphen)",
                        "Violates spacing and hyphen rules",
                    ),
                    (
                        "'10 to 20 meters' (mixing numbers with spelled-out names)",
                        "Inconsistent abbreviation style",
                    ),
                ],
                "BIPM Section 5.4.3: In expressing a range, either the unit must be stated after each number or parentheses used.",
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "How should measurement uncertainty and tolerance be formatted with its unit symbol?",
                "'(100.0 ± 0.2) kg' or '100.0 kg ± 0.2 kg' (ensuring the unit applies to both the value and the uncertainty)",
                [
                    (
                        "'100.0 ± 0.2 kg' (attaching unit only to uncertainty)",
                        "Leaves central value dimensionally undefined",
                    ),
                    (
                        "'100.0 kg ± 0.2' (attaching unit only to central value)",
                        "Leaves uncertainty undefined",
                    ),
                    ("'100.0 ± 0.2kg' (omitting space before unit)", "Violates spacing rule"),
                ],
                "ISO 80000-1: The unit must unambiguously qualify both the estimated value and the uncertainty.",
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "What is the correct representation for the multiplication of two units, such as newton and meter for torque or moment?",
                "N*m or N m (using a half-high dot or space to prevent confusion with metric prefixes like nm for nanometer)",
                [
                    ("Nm (without dot or space)", "Confusing with nm (nanometer)"),
                    ("N/m", "Represents quotient instead of product"),
                    ("N_m (with underscore)", "Non-standard character"),
                ],
                "BIPM Section 5.4.2: Multiplication must be indicated by a space or half-high (centred) dot.",
                DifficultyTier.INTRODUCTORY,
            ),
            (
                "When numbers with many digits are written in SI publications, how should digits be grouped?",
                "Grouped in sets of three with a thin space (e.g. 100 000 or 0.000 123), never using commas or periods as group separators",
                [
                    (
                        "Using commas as thousand separators (e.g. 100,000)",
                        "Commas are reserved as decimal markers in many countries",
                    ),
                    (
                        "Using periods as thousand separators (e.g. 100.000)",
                        "Periods are decimal markers in other countries",
                    ),
                    (
                        "No spaces or separators permitted under any circumstance (e.g. 100000000)",
                        "Impairs human readability",
                    ),
                ],
                "BIPM Section 5.4.4: Digits may be divided into groups of three separated by a thin space, but neither commas nor periods are used.",
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "How must mathematical constants like pi, Euler's number e, and the imaginary unit i be typeset in metrological publications?",
                "Upright Roman font (\\pi, \\text{e}, \\text{i}) because they have fixed invariant mathematical values, not variable physical quantities.",
                [
                    (
                        "Italic font ($e, i, \\pi$) like all mathematical symbols.",
                        "Confuses mathematical constants with physical variables",
                    ),
                    (
                        "Bold font (\\mathbf{e}, \\mathbf{i}) to indicate importance.",
                        "Bold is reserved for vectors and matrices",
                    ),
                    ("Sans-serif font to distinguish from Latin text.", "Non-standard font choice"),
                ],
                "ISO 80000-2: Universal mathematical constants are typeset in upright Roman font.",
                DifficultyTier.ADVANCED,
            ),
            (
                "In SI, can metric prefixes be attached to the base unit of mass, the kilogram (kg)?",
                "No; metric prefix names and symbols are attached to the gram (g), not the kilogram (e.g. write mg or μg, never μkg).",
                [
                    (
                        "Yes; prefixes can be attached directly to kilogram (e.g. kkg for 1000 kg).",
                        "Violates BIPM Section 3.2",
                    ),
                    (
                        "Kilogram cannot take any submultiples under any circumstances.",
                        "Gram takes prefixes instead",
                    ),
                    (
                        "Prefixes are attached to the tonne (t), not the gram.",
                        "Tonne is a non-SI unit accepted for use",
                    ),
                ],
                "BIPM Section 3.2: Names and symbols for decimal multiples and submultiples of the unit of mass are formed by attaching prefix names to 'gram'.",
                DifficultyTier.INTRODUCTORY,
            ),
            (
                "What is the official BIPM rule regarding the pluralization of SI unit symbols when expressing quantities greater than 1 (e.g. 5 kilograms)?",
                "Unit symbols are mathematical entities and are never pluralized with an 's' (write 5 kg, never 5 kgs).",
                [
                    (
                        "Unit symbols must take an 's' in plural form (e.g. 5 kgs, 10 ms).",
                        "Confuses unit symbol with English abbreviation",
                    ),
                    ("Unit symbols take an apostrophe-s (e.g. 5 kg's).", "Punctuation error"),
                    (
                        "Pluralization depends on whether the unit is named after a person.",
                        "False grammatical rule",
                    ),
                ],
                "BIPM Section 5.2: Unit symbols do not change in the plural.",
                DifficultyTier.INTRODUCTORY,
            ),
            (
                "How does an exponent attached to a prefixed unit symbol operate (for example, in the symbol km²)?",
                "The exponent applies to the entire prefixed unit as an indivisible unit: km² = (10³ m)² = 10⁶ m², not 10³ (m²).",
                [
                    (
                        "The exponent applies only to the base unit (km² = 10³ m² = 1000 m²).",
                        "Failing to square the prefix",
                    ),
                    ("Prefixes cannot be used with exponents in SI.", "False restriction"),
                    (
                        "The exponent is applied as km² = 10^(3^2) m² = 10⁹ m².",
                        "Incorrect double exponentiation",
                    ),
                ],
                "BIPM Section 3.1: An exponent attached to a symbol containing a prefix indicates that the multiple or submultiple is raised to the power.",
                DifficultyTier.INTERMEDIATE,
            ),
        ]

        for q, correct, dists, expl, diff in formatting_rules:
            if q in used_questions:
                continue
            used_questions.add(q)
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_conventions_{len(samples) + 1:03d}",
                    task=BenchmarkTask.CONVENTIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=diff,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=expl,
                    metadata={"standard": "BIPM_SI_9th_Ed_ISO_80000"},
                )
            )

        # 4. Procedural Prefix Multiplier Questions from SI Prefixes (fill to count)
        si_prefixes_data = [
            ("quetta", "Q", "10^30", "1 000 000 000 000 000 000 000 000 000 000"),
            ("ronna", "R", "10^27", "1 000 000 000 000 000 000 000 000 000"),
            ("yotta", "Y", "10^24", "1 000 000 000 000 000 000 000 000"),
            ("zetta", "Z", "10^21", "1 000 000 000 000 000 000 000"),
            ("exa", "E", "10^18", "1 000 000 000 000 000 000"),
            ("peta", "P", "10^15", "1 000 000 000 000 000"),
            ("tera", "T", "10^12", "1 000 000 000 000"),
            ("giga", "G", "10^9", "1 000 000 000"),
            ("mega", "M", "10^6", "1 000 000"),
            ("kilo", "k", "10^3", "1 000"),
            ("hecto", "h", "10^2", "100"),
            ("deca", "da", "10^1", "10"),
            ("deci", "d", "10^-1", "0.1"),
            ("centi", "c", "10^-2", "0.01"),
            ("milli", "m", "10^-3", "0.001"),
            ("micro", "μ", "10^-6", "0.000 001"),
            ("nano", "n", "10^-9", "0.000 000 001"),
            ("pico", "p", "10^-12", "0.000 000 000 001"),
            ("femto", "f", "10^-15", "0.000 000 000 000 001"),
            ("atto", "a", "10^-18", "0.000 000 000 000 000 001"),
            ("zepto", "z", "10^-21", "0.000 000 000 000 000 000 001"),
            ("yocto", "y", "10^-24", "0.000 000 000 000 000 000 000 001"),
            ("ronto", "r", "10^-27", "0.000 000 000 000 000 000 000 000 001"),
            ("quecto", "q", "10^-30", "0.000 000 000 000 000 000 000 000 000 001"),
        ]

        for p_name, p_sym, p_pow, _p_dec in si_prefixes_data:
            if len(samples) >= count:
                break
            q = f"Under official BIPM SI standards, what is the multiplying factor and symbol for the metric prefix '{p_name}'?"
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = f"Symbol: '{p_sym}' with multiplying factor {p_pow}"
            dists = [
                (
                    f"Symbol: '{p_sym.swapcase()}' with multiplying factor {p_pow}",
                    "Incorrect case for prefix symbol",
                ),
                (
                    f"Symbol: '{p_sym}' with multiplying factor 10^({-int(p_pow.replace('10^', '')) if '-' in p_pow else '-' + p_pow.replace('10^', '')})",
                    "Inverted sign of decimal exponent",
                ),
                (
                    f"Symbol: '{p_sym}' with multiplying factor 10^({int(p_pow.replace('10^', '')) + 3 if '-' not in p_pow else int(p_pow.replace('10^', '')) - 3})",
                    "Shifted prefix decade factor",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_conventions_{len(samples) + 1:03d}",
                    task=BenchmarkTask.CONVENTIONS,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTRODUCTORY
                    if abs(int(p_pow.replace("10^", ""))) <= 9
                    else DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=f"BIPM SI Brochure Table 7: The SI prefix '{p_name}' has symbol '{p_sym}' representing the factor {p_pow}.",
                    metadata={"prefix_name": p_name, "symbol": p_sym, "factor": p_pow},
                )
            )

        return samples[:count]

    # =========================================================================
    # Task 6: Metrological Uncertainty (GUM) & Sig-Figs (100% Unique Procedural Generator)
    # =========================================================================
    def generate_uncertainty_task(self, count: int = 50) -> list[BenchmarkSample]:
        samples: list[BenchmarkSample] = []
        used_questions: set[str] = set()

        unit_pool = [
            "m",
            "kg",
            "s",
            "A",
            "K",
            "V",
            "N",
            "J",
            "W",
            "Pa",
            "g",
            "mm",
            "mV",
            "kN",
            "kPa",
            "mA",
            "cm",
        ]

        # 1. Procedural Additive Combined Uncertainty in Quadrature (u_c = sqrt(u1^2 + u2^2))
        for _ in range(60):
            if len(samples) >= 18:
                break
            u_sym = self.rng.choice(unit_pool)
            v1 = round(self.rng.uniform(10.0, 100.0), 1)
            v2 = round(self.rng.uniform(5.0, 50.0), 1)

            u1_val = round(
                self.rng.choice([0.03, 0.04, 0.06, 0.08, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]),
                3,
            )
            u2_val = round(
                self.rng.choice([0.04, 0.03, 0.08, 0.06, 0.16, 0.20, 0.15, 0.40, 0.30, 0.25, 0.35]),
                3,
            )

            comb_u = round((u1_val**2 + u2_val**2) ** 0.5, 4)
            comb_v = round(v1 + v2, 1)

            q = (
                f"Two uncorrelated independent measurements A = ({v1} ± {u1_val}) {u_sym} and "
                f"B = ({v2} ± {u2_val}) {u_sym} are added: Y = A + B = {comb_v} {u_sym}. "
                f"According to JCGM 100:2008 (GUM), what is the combined standard uncertainty u_c(Y)?"
            )
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = (
                f"u_c(Y) = sqrt(({u1_val})^2 + ({u2_val})^2) = {comb_u:.4f} {u_sym}".rstrip(
                    "0"
                ).rstrip(".")
                + f" {u_sym}"
            )
            linear_sum = round(u1_val + u2_val, 3)
            product_u = round(u1_val * u2_val, 4)
            root_sum_nosq = round((u1_val + u2_val) ** 0.5, 4)

            dists = [
                (
                    f"u_c(Y) = {u1_val} + {u2_val} = {linear_sum} {u_sym}",
                    "Linear addition instead of root-sum-square quadrature",
                ),
                (
                    f"u_c(Y) = ({u1_val} * {u2_val}) = {product_u} {u_sym}",
                    "Multiplying uncertainties directly",
                ),
                (
                    f"u_c(Y) = sqrt({u1_val} + {u2_val}) = {root_sum_nosq} {u_sym}",
                    "Square root of sum without squaring variances",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_uncert_{len(samples) + 1:03d}",
                    task=BenchmarkTask.UNCERTAINTY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="JCGM 100:2008 (GUM) Section 5.1.2: For independent additive components Y = A + B, variances add: u_c^2(Y) = u^2(A) + u^2(B).",
                    metadata={"task_type": "additive_quadrature", "u1": u1_val, "u2": u2_val},
                )
            )

        # 2. Expanded Uncertainty and Coverage Factors (U = k * u_c and u_c = U / k)
        for _ in range(60):
            if len(samples) >= 32:
                break
            u_sym = self.rng.choice(unit_pool)
            v = round(self.rng.uniform(20.0, 500.0), 2)
            u_c = round(
                self.rng.choice([0.015, 0.025, 0.040, 0.050, 0.075, 0.120, 0.250, 0.350, 0.450]), 3
            )
            k_factor = self.rng.choice([2, 3])
            confidence = "95.45%" if k_factor == 2 else "99.73%"
            expanded_U = round(k_factor * u_c, 4)

            if self.rng.random() > 0.5:
                q = (
                    f"A laboratory calibration establishes a combined standard uncertainty u_c = {u_c} {u_sym} "
                    f"for a measurement of {v} {u_sym}. What is the expanded uncertainty U for coverage factor k = {k_factor} (~{confidence} level of confidence)?"
                )
                if q in used_questions:
                    continue
                used_questions.add(q)
                correct = f"U = k * u_c = {k_factor} * {u_c} = {expanded_U:.3f} {u_sym}"
                dists = [
                    (
                        f"U = u_c / {k_factor} = {u_c / k_factor:.4f} {u_sym}",
                        "Dividing instead of multiplying by coverage factor",
                    ),
                    (f"U = u_c^2 = {u_c**2:.6f} {u_sym}", "Squaring standard uncertainty"),
                    (
                        f"U = {u_c} {u_sym}",
                        "Confusing expanded uncertainty with standard uncertainty",
                    ),
                ]
            else:
                q = (
                    f"A measurement certificate reports y = ({v} ± {expanded_U}) {u_sym} with coverage factor k = {k_factor} (normal distribution). "
                    f"What is the combined standard uncertainty u_c(y)?"
                )
                if q in used_questions:
                    continue
                used_questions.add(q)
                correct = f"u_c(y) = U / k = {expanded_U} / {k_factor} = {u_c:.3f} {u_sym}"
                dists = [
                    (
                        f"u_c(y) = U * k = {expanded_U} * {k_factor} = {expanded_U * k_factor:.3f} {u_sym}",
                        "Multiplying instead of dividing by coverage factor",
                    ),
                    (
                        f"u_c(y) = {expanded_U} {u_sym}",
                        "Assuming expanded uncertainty is standard uncertainty",
                    ),
                    (
                        f"u_c(y) = sqrt({expanded_U}) = {expanded_U**0.5:.4f} {u_sym}",
                        "Square root of expanded uncertainty",
                    ),
                ]

            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_uncert_{len(samples) + 1:03d}",
                    task=BenchmarkTask.UNCERTAINTY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="GUM Section 6.2: Expanded uncertainty U is obtained by multiplying combined standard uncertainty u_c by coverage factor k: U = k * u_c.",
                    metadata={"task_type": "expanded_uncertainty", "k": k_factor},
                )
            )

        # 3. Procedural Multiplicative Relative Uncertainty in Quadrature
        relations = [
            ("Power P = V * I", "voltage V", "current I", "P"),
            ("Kinetic Energy E_k = 0.5 * m * v^2", "mass m", "velocity v", "E_k"),
            ("Electrical Resistance R = V / I", "voltage V", "current I", "R"),
            ("Density rho = m / V", "mass m", "volume V", "rho"),
            ("Force F = m * a", "mass m", "acceleration a", "F"),
            ("Pressure p = F / A", "force F", "area A", "p"),
            ("Electrical Work W = P * t", "power P", "time t", "W"),
            ("Momentum p = m * v", "mass m", "velocity v", "p"),
        ]
        for rel_name, var1, var2, target_sym in relations:
            ur1 = self.rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
            ur2 = self.rng.choice([1.0, 2.0, 3.0, 4.0])
            if "v^2" in rel_name:
                eff_ur2 = 2.0 * ur2
                comb_rel = round((ur1**2 + eff_ur2**2) ** 0.5, 2)
                formula_detail = f"sqrt(({ur1}%)^2 + (2 * {ur2}%)^2) = sqrt({ur1**2} + {eff_ur2**2})% ≈ {comb_rel}%"
            else:
                comb_rel = round((ur1**2 + ur2**2) ** 0.5, 2)
                formula_detail = (
                    f"sqrt(({ur1}%)^2 + ({ur2}%)^2) = sqrt({ur1**2 + ur2**2})% ≈ {comb_rel}%"
                )

            q = (
                f"In the relation {rel_name}, independent relative standard uncertainties are evaluated as "
                f"u_r({var1}) = {ur1}% and u_r({var2}) = {ur2}%. What is the combined relative standard uncertainty u_r({target_sym})?"
            )
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = f"u_r({target_sym}) = {formula_detail}"
            linear_sum = round(ur1 + (2.0 * ur2 if "v^2" in rel_name else ur2), 2)
            dists = [
                (
                    f"u_r({target_sym}) = {ur1}% + {ur2}% = {linear_sum}%",
                    "Linear summation of relative uncertainties",
                ),
                (f"u_r({target_sym}) = {ur1 * ur2:.2f}%", "Product of relative uncertainties"),
                (
                    f"u_r({target_sym}) = sqrt({ur1 + ur2:.1f})% = {(ur1 + ur2) ** 0.5:.2f}%",
                    "Square root of sum without squaring percentages",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_uncert_{len(samples) + 1:03d}",
                    task=BenchmarkTask.UNCERTAINTY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.ADVANCED
                    if "v^2" in rel_name
                    else DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="GUM Section 5.1.6: For power-law product/quotient models Y = c * X1^p1 * X2^p2, relative standard uncertainties combine in quadrature.",
                    metadata={"task_type": "relative_uncertainty_quadrature", "relation": rel_name},
                )
            )

        # 4. Procedural Type A Sample Mean & Resolution Uncertainty
        stats_cases = [
            ("length gauge block", "mm", 0.12, 9, 3.0),
            ("resistor batch", "Ω", 0.40, 16, 4.0),
            ("thermocouple temperature", "°C", 0.25, 25, 5.0),
            ("mass comparator", "mg", 0.18, 9, 3.0),
            ("voltage calibrator", "mV", 0.50, 25, 5.0),
            ("current shunt", "μA", 0.36, 36, 6.0),
            ("pressure transducer", "kPa", 0.49, 49, 7.0),
        ]
        for item_name, unit, s_val, n_val, sqrt_n in stats_cases:
            u_mean = round(s_val / sqrt_n, 4)
            q = (
                f"A series of n = {n_val} repeated independent measurements of a {item_name} gives a sample standard deviation "
                f"s = {s_val} {unit}. According to GUM Type A evaluation, what is the standard uncertainty of the arithmetic mean u(x̄)?"
            )
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = (
                f"u(x̄) = s / sqrt(n) = {s_val} / sqrt({n_val}) = {s_val} / {sqrt_n:.1f} = {u_mean:.4f} {unit}".rstrip(
                    "0"
                ).rstrip(".")
                + f" {unit}"
            )
            dists = [
                (
                    f"u(x̄) = s = {s_val} {unit}",
                    "Reporting individual sample standard deviation s instead of standard deviation of the mean",
                ),
                (f"u(x̄) = s / n = {s_val / n_val:.4f} {unit}", "Dividing by n instead of sqrt(n)"),
                (f"u(x̄) = s * sqrt(n) = {s_val * sqrt_n:.4f} {unit}", "Multiplying by sqrt(n)"),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_uncert_{len(samples) + 1:03d}",
                    task=BenchmarkTask.UNCERTAINTY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="GUM Section 4.2.3: The standard uncertainty of the arithmetic mean is u(x̄) = s / sqrt(n).",
                    metadata={"task_type": "type_a_mean", "n": n_val},
                )
            )

        # 5. Type B Distribution Shapes, GUM Concepts & Rounding Rules
        gum_concepts = [
            (
                "A digital micrometer has a display resolution of 0.001 mm. Assuming a rectangular (uniform) probability distribution with bounds a = ±0.0005 mm, what is the standard uncertainty u due to resolution?",
                "u = a / sqrt(3) = 0.0005 / sqrt(3) ≈ 0.000289 mm",
                [
                    (
                        "u = a / 2 = 0.000250 mm",
                        "Dividing by 2 instead of sqrt(3) for rectangular distribution",
                    ),
                    ("u = a * sqrt(3) ≈ 0.000866 mm", "Multiplying by sqrt(3)"),
                    (
                        "u = a / sqrt(6) ≈ 0.000204 mm",
                        "Using triangular distribution divisor sqrt(6) instead of sqrt(3)",
                    ),
                ],
                "GUM Section 4.3.7: For a rectangular distribution with half-width a, the variance is u^2 = a^2 / 3, so standard uncertainty is u = a / sqrt(3).",
                DifficultyTier.ADVANCED,
            ),
            (
                "A manufacturer specifies that a temperature probe has maximum error bounds ±0.6 °C with a triangular probability distribution. What is the Type B standard uncertainty?",
                "u = a / sqrt(6) = 0.6 / sqrt(6) ≈ 0.245 °C",
                [
                    (
                        "u = 0.6 / sqrt(3) ≈ 0.346 °C",
                        "Using rectangular divisor sqrt(3) instead of triangular divisor sqrt(6)",
                    ),
                    ("u = 0.6 / 2 = 0.300 °C", "Assuming normal distribution k=2"),
                    ("u = 0.6 °C", "Taking full bound as standard uncertainty"),
                ],
                "GUM Section 4.3.9: For a triangular distribution with half-width a, the variance is u^2 = a^2 / 6, hence u = a / sqrt(6).",
                DifficultyTier.ADVANCED,
            ),
            (
                "What fundamentally distinguishes a Type A evaluation of measurement uncertainty from a Type B evaluation according to the GUM (JCGM 100:2008)?",
                "Type A evaluation is based on statistical analysis of series of repeated observations, whereas Type B is based on scientific judgment using all non-statistical information (certificates, specifications, handbooks).",
                [
                    (
                        "Type A applies only to SI base units, while Type B applies to derived units.",
                        "False categorization by unit type",
                    ),
                    (
                        "Type A represents systematic errors, while Type B represents random errors.",
                        "Conflating evaluation method with error nature",
                    ),
                    (
                        "Type A has zero uncertainty, while Type B has experimental uncertainty.",
                        "False claim about uncertainty values",
                    ),
                ],
                "GUM Section 2.3.2: Type A: method of evaluation of uncertainty by the statistical analysis of series of observations; Type B: method of evaluation by means other than the statistical analysis of series of observations.",
                DifficultyTier.INTRODUCTORY,
            ),
            (
                "What is the relative standard uncertainty u_r(V) for a voltage measurement V = 230.0 V with standard uncertainty u(V) = 0.46 V?",
                "u_r(V) = u(V) / |V| = 0.46 / 230.0 = 0.0020 = 2.0e-3 (or 0.20%)",
                [
                    ("u_r(V) = 230.0 * 0.46 = 105.8 V", "Multiplying value and uncertainty"),
                    (
                        "u_r(V) = 0.46 V",
                        "Reporting absolute uncertainty instead of dimensionless relative ratio",
                    ),
                    ("u_r(V) = 2.0e-4 (0.02%)", "Order of magnitude calculation error"),
                ],
                "GUM Section 5.1.6: Relative standard uncertainty is defined as u_r(y) = u_c(y) / |y|.",
                DifficultyTier.INTRODUCTORY,
            ),
            (
                "According to metrological reporting standards, how many significant digits should generally be retained in a standard uncertainty value u, and how must the measured value be rounded?",
                "Uncertainty is generally rounded to 1 or at most 2 significant digits, and the measured value must be rounded to the same least significant decimal place as the uncertainty.",
                [
                    (
                        "Uncertainty must always have 5 significant digits to preserve calculation precision.",
                        "False precision in uncertainty reporting",
                    ),
                    (
                        "Measured value is rounded to nearest integer regardless of uncertainty.",
                        "Decoupling measurement value from uncertainty resolution",
                    ),
                    (
                        "Uncertainty is always rounded up to the nearest power of 10.",
                        "Excessive coarseness",
                    ),
                ],
                "GUM Section 7.2.6 & ISO 80000-1: Numerical values and their uncertainties must be rounded to match the decimal resolution of the uncertainty.",
                DifficultyTier.INTERMEDIATE,
            ),
            (
                "A measurement calculation yields x = 12.34567 m and combined uncertainty u = 0.0234 m. What is the correct reported format following GUM rounding rules?",
                "(12.346 ± 0.023) m (uncertainty to 2 significant digits, value rounded to thousandths place)",
                [
                    (
                        "(12.34567 ± 0.0234) m (full unrounded floating precision)",
                        "Unrounded raw calculator precision",
                    ),
                    (
                        "(12.3 ± 0.023) m (value rounded coarser than uncertainty)",
                        "Loss of significant figures in value",
                    ),
                    ("(12.0 ± 0.02) m", "Over-rounded central value"),
                ],
                "GUM Section 7.2.6: The number of decimal places in the result should be consistent with the uncertainty.",
                DifficultyTier.INTERMEDIATE,
            ),
        ]

        for q, correct, dists, expl, diff in gum_concepts:
            if len(samples) >= count:
                break
            if q in used_questions:
                continue
            used_questions.add(q)
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_uncert_{len(samples) + 1:03d}",
                    task=BenchmarkTask.UNCERTAINTY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=diff,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation=expl,
                    metadata={"standard": "JCGM_100_GUM"},
                )
            )

        # 6. Fallback Procedural Additive Quadrature (if count > samples)
        while len(samples) < count:
            u_sym = self.rng.choice(unit_pool)
            v1 = round(self.rng.uniform(10.0, 100.0), 1)
            v2 = round(self.rng.uniform(5.0, 50.0), 1)
            u1_val = round(self.rng.uniform(0.01, 0.80), 3)
            u2_val = round(self.rng.uniform(0.01, 0.80), 3)
            comb_u = round((u1_val**2 + u2_val**2) ** 0.5, 4)
            comb_v = round(v1 + v2, 1)

            q = (
                f"Two uncorrelated independent measurements A = ({v1} ± {u1_val}) {u_sym} and "
                f"B = ({v2} ± {u2_val}) {u_sym} are added: Y = A + B = {comb_v} {u_sym}. "
                f"According to JCGM 100:2008 (GUM), what is the combined standard uncertainty u_c(Y)?"
            )
            if q in used_questions:
                continue
            used_questions.add(q)

            correct = (
                f"u_c(Y) = sqrt(({u1_val})^2 + ({u2_val})^2) = {comb_u:.4f} {u_sym}".rstrip(
                    "0"
                ).rstrip(".")
                + f" {u_sym}"
            )
            linear_sum = round(u1_val + u2_val, 3)
            product_u = round(u1_val * u2_val, 4)
            root_sum_nosq = round((u1_val + u2_val) ** 0.5, 4)

            dists = [
                (
                    f"u_c(Y) = {u1_val} + {u2_val} = {linear_sum} {u_sym}",
                    "Linear addition instead of root-sum-square quadrature",
                ),
                (
                    f"u_c(Y) = ({u1_val} * {u2_val}) = {product_u} {u_sym}",
                    "Multiplying uncertainties directly",
                ),
                (
                    f"u_c(Y) = sqrt({u1_val} + {u2_val}) = {root_sum_nosq} {u_sym}",
                    "Square root of sum without squaring variances",
                ),
            ]
            opts, correct_key = self._build_mcq_options(correct_text=correct, distractors=dists)
            samples.append(
                BenchmarkSample(
                    id=f"drum_bench_uncert_{len(samples) + 1:03d}",
                    task=BenchmarkTask.UNCERTAINTY,
                    format=BenchmarkFormat.MCQ,
                    difficulty=DifficultyTier.INTERMEDIATE,
                    question=q,
                    options=opts,
                    correct_option_key=correct_key,
                    ground_truth_answer=correct,
                    explanation="JCGM 100:2008 (GUM) Section 5.1.2: For independent additive components Y = A + B, variances add: u_c^2(Y) = u^2(A) + u^2(B).",
                    metadata={"task_type": "additive_quadrature", "u1": u1_val, "u2": u2_val},
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
