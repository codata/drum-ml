"""Tests for DimensionVector arithmetic and LaTeX formatting."""

import pytest
from drum_ml.models.entities import DimensionVector
from drum_ml.symbolic.dimensions import parse_qudt_dimension_string


def test_dimension_vector_multiplication():
    # Force: [L=1, M=1, T=-2] * Distance: [L=1] -> Work: [L=2, M=1, T=-2]
    force = DimensionVector(L=1, M=1, T=-2)
    distance = DimensionVector(L=1)
    work = force * distance
    assert work == DimensionVector(L=2, M=1, T=-2)


def test_dimension_vector_division():
    # Work: [L=2, M=1, T=-2] / Area: [L=2] -> Pressure: [L=-1, M=1, T=-2] (same as Force/Area)
    work = DimensionVector(L=2, M=1, T=-2)
    volume = DimensionVector(L=3)
    pressure = work / volume
    assert pressure == DimensionVector(L=-1, M=1, T=-2)


def test_dimension_vector_latex():
    force = DimensionVector(L=1, M=1, T=-2, Theta=1)
    latex_dim = force.to_latex()
    assert "\\text{L}" in latex_dim
    assert "\\text{M}" in latex_dim
    assert "\\text{T}^{-2}" in latex_dim
    assert "\\Theta" in latex_dim
    assert "\\text{\\Theta}" not in latex_dim

    si_base = force.to_si_base_unit_latex()
    assert "\\text{kg}" in si_base
    assert "\\text{m}" in si_base
    assert "\\text{s}^{-2}" in si_base
    assert "\\text{K}" in si_base


def test_dimensionless():
    d = DimensionVector()
    assert d.is_dimensionless()
    assert d.to_latex() == "1"
    assert d.to_si_base_unit_latex() == "1"


def test_qudt_dimension_parsing():
    parsed = parse_qudt_dimension_string("L2M1T-2")
    assert parsed.L == 2
    assert parsed.M == 1
    assert parsed.T == -2
    assert parsed.Theta == 0

    ampere_vec = parse_qudt_dimension_string("A0E1L0I0M0H0T0D0")
    assert ampere_vec.I == 1
    assert ampere_vec.Theta == 0
    assert ampere_vec.L == 0
    assert ampere_vec.M == 0
    assert ampere_vec.T == 0
    assert ampere_vec.to_latex() == "\\text{I}"
    assert ampere_vec.to_si_base_unit_latex() == "\\text{A}"
