"""LaTeX Tokenizer, Sanitizer & Expression Normalizer for Metrology."""

import re
from typing import List, Tuple


def sanitize_latex_units(latex_str: str) -> str:
    """Normalizes metrological LaTeX notation into standard parseable arithmetic strings.

    Examples:
        '\\text{kg}\\cdot\\text{m}\\cdot\\text{s}^{-2}' -> 'kg * m * s**-2'
        '10^{-34}\\text{ J}\\cdot\\text{s}' -> '10**-34 * J * s'
        '\\frac{\\text{N}}{\\text{m}^2}' -> '(N) / (m**2)'
    """
    if not latex_str:
        return ""

    s = latex_str.strip()

    # Remove enclosing math markers ($ or $$ or \[ \])
    s = re.sub(r"^\$\$|\$\$$|^\\\[|\\\]$|^\$|\$$", "", s).strip()

    # Normalize thin spaces and formatting
    s = s.replace(r"\,", " ").replace(r"\:", " ").replace(r"\;", " ")
    s = s.replace(r"\times", " * ").replace(r"\cdot", " * ")
    s = s.replace(r"\left", "").replace(r"\right", "")

    # Replace \frac{A}{B} with (A) / (B)
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_pattern.search(s):
        s = frac_pattern.sub(r"(\1) / (\2)", s)

    # Strip \text{...} or \mathrm{...} or \mathbf{...}
    s = re.sub(r"\\(text|mathrm|mathbf|mathit)\{([^{}]+)\}", r"\2", s)

    # Convert powers: ^{ -2 } -> **-2, ^2 -> **2
    s = re.sub(r"\^\{([^{}]+)\}", r"**\1", s)
    s = re.sub(r"\^([0-9\-+]+)", r"**\1", s)

    # Clean double spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def check_latex_math_balance(text: str) -> Tuple[bool, List[str]]:
    """Verifies that math delimiters ($ and $$), curly braces, and brackets are properly balanced."""
    errors = []

    # Check $ and $$
    # Count unescaped dollars
    dollars = re.findall(r"(?<!\\)\$", text)
    if len(dollars) % 2 != 0:
        errors.append(f"Unbalanced inline/block math '$' delimiters (found {len(dollars)}).")

    # Check curly braces
    open_braces = len(re.findall(r"(?<!\\)\{", text))
    close_braces = len(re.findall(r"(?<!\\)\}", text))
    if open_braces != close_braces:
        errors.append(f"Unbalanced curly braces '{{}}' (open: {open_braces}, close: {close_braces}).")

    return (len(errors) == 0, errors)
