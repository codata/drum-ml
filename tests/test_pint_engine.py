"""Tests for Pint / SymPy symbolic unit equivalence and LaTeX balance."""

from drum_ml.symbolic.latex_parser import check_latex_math_balance, sanitize_latex_units
from drum_ml.symbolic.pint_engine import check_unit_conversion_equivalence


def test_pint_basic_equivalence():
    # 1 N == 1 kg * m / s^2
    is_eq, err = check_unit_conversion_equivalence("1 N", "1 kg * m / s**2")
    assert is_eq
    assert err is None


def test_pint_dimensional_mismatch():
    # 1 N vs 1 J
    is_eq, err = check_unit_conversion_equivalence("1 N", "1 J")
    assert not is_eq
    assert "Dimensional incompatibility" in err


def test_latex_balance_checker():
    valid_latex = "The speed of light is $c = 299\\,792\\,458\\text{ m/s}$ in vacuum."
    is_bal, errs = check_latex_math_balance(valid_latex)
    assert is_bal
    assert len(errs) == 0

    invalid_latex = "The speed of light is $c = 299\\,792\\,458\\text{ m/s} in vacuum."
    is_bal2, errs2 = check_latex_math_balance(invalid_latex)
    assert not is_bal2
    assert len(errs2) > 0


def test_sanitize_latex_units():
    sanitized = sanitize_latex_units(r"\text{kg}\cdot\text{m}\cdot\text{s}^{-2}")
    assert sanitized == "kg * m * s**-2"

    # Implicit negative exponents on unit symbols
    assert sanitize_latex_units("6.02214076e23 mol-1") == "6.02214076e23 mol**-1"
    assert sanitize_latex_units("9.80665 m s-2") == "9.80665 m s**-2"
    assert sanitize_latex_units("101325 kg m-1 s-2") == "101325 kg m**-1 s**-2"


def test_pint_implicit_exponents():
    # mol^-1 vs mol-1
    is_eq, err = check_unit_conversion_equivalence("6.02214076e23 mol^-1", "6.02214076e23 mol-1")
    assert is_eq
    assert err is None

    # m/s^2 vs m s-2
    is_eq2, err2 = check_unit_conversion_equivalence("9.80665 m/s^2", "9.80665 m s-2")
    assert is_eq2
    assert err2 is None
