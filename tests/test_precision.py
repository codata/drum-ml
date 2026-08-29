"""Tests for arbitrary precision and CODATA string parsing."""

import pytest
from drum_ml.symbolic.precision import (
    parse_codata_value_uncertainty,
    verify_exact_numeric_precision,
)


def test_parse_codata_uncertainty():
    # Newtonian constant of gravitation: 6.67430(15)e-11
    val, unc = parse_codata_value_uncertainty("6.67430(15)e-11")
    assert val == "6.67430e-11"
    assert unc == "0.00015e-11"

    # Fine-structure constant: 0.0072973525693(11)
    val2, unc2 = parse_codata_value_uncertainty("0.0072973525693(11)")
    assert val2 == "0.0072973525693"
    assert unc2 == "0.0000000000011"


def test_verify_exact_numeric_precision():
    # Speed of light exact value
    assert verify_exact_numeric_precision("299792458", "299792458")
    assert not verify_exact_numeric_precision("299792458", "299792450")

    # Planck constant exact decimal precision
    assert verify_exact_numeric_precision("6.62607015e-34", "6.62607015e-34")
    assert not verify_exact_numeric_precision("6.62607015e-34", "6.626070e-34")

