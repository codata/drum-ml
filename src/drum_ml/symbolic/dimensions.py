"""SI 7-Base Dimension Vector Arithmetic and Representation."""

from drum_ml.models.entities import DimensionVector


def parse_qudt_dimension_string(dim_str: str) -> DimensionVector:
    """Parses QUDT dimension string or symbol (e.g., 'A0E0L2I0M1H0T-2D0' or 'L2 M T-2')
    into a DimensionVector.
    """
    # Standard QUDT vector encoding: A (Angle), E (Current I), L (Length), I (Luminous J / Current), M (Mass), H (Theta), T (Time), D (Amount N)
    import re

    vec = DimensionVector()
    if not dim_str:
        return vec

    # Try matching QUDT encoded format: e.g. L1M1T-2 or A0E0L2I0M1H0T-2D0
    # Common letters in QUDT: L (Length), M (Mass), T (Time), E/I (Current), H/Theta (Temp), D/N (Amount), J (Luminous)
    patterns = [
        (r"L([+-]?\d+)", "L"),
        (r"M([+-]?\d+)", "M"),
        (r"T([+-]?\d+)", "T"),
        (r"(?:E|I)([+-]?\d+)", "I"),
        (r"(?:H|Theta|[\u0398\u03b8])([+-]?\d+)", "Theta"),
        (r"(?:D|N)([+-]?\d+)", "N"),
        (r"J([+-]?\d+)", "J"),
    ]

    for pat, dim_name in patterns:
        m = re.search(pat, dim_str)
        if m:
            val_str = m.group(1) if m.group(1) is not None else "1"
            try:
                setattr(vec, dim_name, int(val_str))
            except ValueError:
                pass

    return vec
