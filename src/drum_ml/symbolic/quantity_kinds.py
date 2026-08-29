"""QuantityKind Consistency & Disambiguation Engine.

Enforces metrological rules distinguishing between distinct Quantity Kinds
that share identical DimensionVectors (e.g., Torque vs. Energy, Frequency vs. Activity).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from drum_ml.models.entities import DimensionVector, QuantityKindEntity, UnitEntity


# Well-known pairs of QuantityKinds that share dimension vectors but must NOT be conflated
DIMENSIONALLY_DEGENERATE_FAMILIES: Dict[str, Dict[str, Any]] = {
    "L2_M1_T-2": {
        "description": "Energy / Work vs Torque / Moment of Force",
        "dimension": DimensionVector(L=2, M=1, T=-2),
        "kinds": {
            "Energy": ["J", "W·s", "cal", "eV", "BTU", "erg"],
            "Work": ["J", "W·s", "cal", "eV", "BTU", "erg"],
            "Torque": ["N·m", "kN·m", "lbf·ft", "lbf·in", "dyn·cm"],
            "MomentOfForce": ["N·m", "kN·m", "lbf·ft", "lbf·in"],
        },
        "invalid_assignments": [
            ("Torque", "J", "Torque must be expressed in newton metres (N·m), not joules (J)."),
            ("MomentOfForce", "J", "Moment of force must be expressed in newton metres (N·m), not joules (J)."),
            ("Energy", "N·m", "Energy is customarily expressed in joules (J), not newton metres (N·m)."),
        ],
    },
    "T-1": {
        "description": "Frequency vs Activity (Radioactivity) vs Angular Velocity",
        "dimension": DimensionVector(T=-1),
        "kinds": {
            "Frequency": ["Hz", "kHz", "MHz", "GHz"],
            "Activity": ["Bq", "Ci", "kBq", "MBq"],
            "AngularVelocity": ["rad/s", "deg/s", "rpm"],
        },
        "invalid_assignments": [
            ("Activity", "Hz", "Radioactive decay rate must be expressed in becquerels (Bq), not hertz (Hz)."),
            ("Frequency", "Bq", "Periodic frequency must be expressed in hertz (Hz), not becquerels (Bq)."),
            ("AngularVelocity", "Hz", "Angular velocity is expressed in radians per second (rad/s), not hertz (Hz)."),
        ],
    },
    "L2_T-2": {
        "description": "Absorbed Dose vs Dose Equivalent",
        "dimension": DimensionVector(L=2, T=-2),
        "kinds": {
            "AbsorbedDose": ["Gy", "mGy", "rad"],
            "DoseEquivalent": ["Sv", "mSv", "rem"],
        },
        "invalid_assignments": [
            ("AbsorbedDose", "Sv", "Absorbed radiation dose is expressed in grays (Gy), not sieverts (Sv)."),
            ("DoseEquivalent", "Gy", "Dose equivalent (biological risk) is expressed in sieverts (Sv), not grays (Gy)."),
        ],
    },
}


def is_unit_compatible_with_quantity_kind(
    unit_symbol: str,
    quantity_kind_name: str,
    unit_entity: Optional[UnitEntity] = None,
    quantity_kind_entity: Optional[QuantityKindEntity] = None,
) -> Tuple[bool, Optional[str]]:
    """Checks whether a given unit symbol is metrologically valid for a quantity kind.

    Returns:
        (is_valid, error_explanation)
    """
    clean_sym = unit_symbol.strip()
    clean_kind = quantity_kind_name.strip()

    # Check known degenerative families
    for fam_key, fam_data in DIMENSIONALLY_DEGENERATE_FAMILIES.items():
        for invalid_kind, invalid_unit, reason in fam_data["invalid_assignments"]:
            if invalid_kind.lower() in clean_kind.lower() and clean_sym == invalid_unit:
                return False, reason

    # If entity metadata is available, verify linkage
    if unit_entity and quantity_kind_entity:
        if quantity_kind_entity.uri in unit_entity.has_quantity_kinds:
            return True, None
        if unit_entity.uri in quantity_kind_entity.applicable_units:
            return True, None

    return True, None
