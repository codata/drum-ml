"""Pint and SymPy Symbolic Equivalence and Unit Verification Engine."""

import pint

from drum_ml.symbolic.latex_parser import sanitize_latex_units

# Initialize global Pint UnitRegistry
ureg = pint.UnitRegistry()
# Define quantity shortcut
Q_ = ureg.Quantity


def check_unit_conversion_equivalence(
    expr_left_str: str,
    expr_right_str: str,
    is_temperature_delta: bool = False,
    rel_tol: float = 1e-4,
) -> tuple[bool, str | None]:
    """Checks whether two unit expressions (e.g. '1 N' and '1 kg * m / s**2', or '100 degC' and '373.15 K')
    are physically and numerically equivalent.

    Args:
        expr_left_str: Left-hand side string or LaTeX snippet.
        expr_right_str: Right-hand side string or LaTeX snippet.
        is_temperature_delta: If True, treats temperature units as differences (no affine offset).
        rel_tol: Relative numerical tolerance.

    Returns:
        (is_equivalent, error_message)
    """
    clean_left = sanitize_latex_units(expr_left_str)
    clean_right = sanitize_latex_units(expr_right_str)

    try:
        if is_temperature_delta:
            # Handle temperature differences (e.g. delta_degC -> delta_degK)
            clean_left = clean_left.replace("degC", "delta_degC").replace("degF", "delta_degF")
            clean_right = clean_right.replace("degC", "delta_degC").replace("degF", "delta_degF")

        q_left = ureg(clean_left)
        q_right = ureg(clean_right)

        # Check dimensional compatibility
        if not q_left.check(q_right.dimensionality):
            return (
                False,
                f"Dimensional incompatibility: [{q_left.dimensionality}] != [{q_right.dimensionality}]",
            )

        # Check numerical equality after conversion to base SI
        q_left_base = q_left.to_base_units()
        q_right_base = q_right.to_base_units()

        mag_l = float(q_left_base.magnitude)
        mag_r = float(q_right_base.magnitude)
        diff = abs(mag_l - mag_r)
        norm = max(abs(mag_l), abs(mag_r))

        if norm == 0.0:
            if diff > 1e-12:
                return False, f"Magnitude mismatch: {q_left_base} != {q_right_base}"
        elif diff / norm > rel_tol:
            return False, f"Magnitude mismatch: {q_left_base} != {q_right_base}"

        return True, None

    except Exception as e:
        return False, f"Pint evaluation error: {e!s}"
