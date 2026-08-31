"""Arbitrary-Precision Decimal Verification and CODATA Uncertainty Parsing."""

import re
from decimal import Decimal, InvalidOperation


def parse_codata_value_uncertainty(val_str: str) -> tuple[str, str | None]:
    """Parses standard CODATA string representation with concise parenthetical uncertainty,
    e.g. '6.67430(15)e-11' -> ('6.67430e-11', '0.00015e-11') or ('6.67430e-11', '1.5e-15').

    Returns:
        (clean_value_string, standard_uncertainty_string)
    """
    cleaned = val_str.strip().replace(" ", "")

    # Check for concise form: e.g. 6.67430(15)e-11 or 0.0072973525693(11)
    m = re.match(r"^([+-]?\d+)(\.?\d*)\((\d+)\)(?:[eE]([+-]?\d+))?$", cleaned)
    if not m:
        # Check if already a pure number
        try:
            Decimal(cleaned)
            return cleaned, None
        except InvalidOperation:
            # Strip non-numeric suffixes or trailing ellipses
            clean_num = re.sub(r"[^\d\.eE\+\-]", "", cleaned)
            return clean_num, None

    int_part, frac_part, unc_digits, exp_part = m.groups()
    frac_len = len(frac_part) - 1 if frac_part and frac_part.startswith(".") else 0
    main_val = f"{int_part}{frac_part}"

    # Construct uncertainty value
    if frac_len > 0:
        unc_val = f"{int(unc_digits) * (10**-frac_len):.{frac_len}f}"
    else:
        unc_val = unc_digits

    if exp_part:
        main_val = f"{main_val}e{exp_part}"
        unc_val = f"{unc_val}e{exp_part}"

    return main_val, unc_val


def verify_exact_numeric_precision(expected_value_str: str, test_value_str: str) -> bool:
    """Verifies that a test constant value matches an expected exact value with zero decimal drift.

    Args:
        expected_value_str: Ground truth exact string (e.g. "299792458" or "6.62607015e-34").
        test_value_str: Candidate value string from model output.

    Returns:
        True if exact decimal match, False otherwise.
    """
    try:
        clean_exp = expected_value_str.strip().replace(" ", "")
        clean_test = test_value_str.strip().replace(" ", "")
        return Decimal(clean_exp) == Decimal(clean_test)
    except (InvalidOperation, TypeError):
        return False
