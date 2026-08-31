"""Tests for QuantityKind vs Unit separation and disambiguation."""

from drum_ml.symbolic.quantity_kinds import is_unit_compatible_with_quantity_kind


def test_torque_vs_energy_separation():
    # Torque with Joules should be rejected
    is_valid, reason = is_unit_compatible_with_quantity_kind("J", "Torque")
    assert not is_valid
    assert "Torque must be expressed in newton metres" in reason

    # Torque with N·m should be valid
    is_valid, reason = is_unit_compatible_with_quantity_kind("N·m", "Torque")
    assert is_valid
    assert reason is None


def test_frequency_vs_radioactivity_separation():
    # Activity with Hz should be rejected
    is_valid, reason = is_unit_compatible_with_quantity_kind("Hz", "Activity")
    assert not is_valid
    assert "Radioactive decay rate must be expressed in becquerels" in reason

    # Frequency with Hz should be valid
    is_valid, reason = is_unit_compatible_with_quantity_kind("Hz", "Frequency")
    assert is_valid
    assert reason is None


def test_absorbed_dose_vs_dose_equivalent():
    # AbsorbedDose with Sv should be rejected
    is_valid, reason = is_unit_compatible_with_quantity_kind("Sv", "AbsorbedDose")
    assert not is_valid
    assert "grays" in reason
