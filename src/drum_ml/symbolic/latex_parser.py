"""LaTeX Tokenizer, Sanitizer & Expression Normalizer for Metrology."""

import re


def sanitize_latex_units(latex_str: str) -> str:
    """Normalizes metrological LaTeX notation and plain text unit strings into standard parseable arithmetic strings.

    Examples:
        '\\text{kg}\\cdot\\text{m}\\cdot\\text{s}^{-2}' -> 'kg * m * s**-2'
        '10^{-34}\\text{ J}\\cdot\\text{s}' -> '10**-34 * J * s'
        '\\frac{\\text{N}}{\\text{m}^2}' -> '(N) / (m**2)'
        '6.02214076e23 mol-1' -> '6.02214076e23 mol**-1'
        '9.80665 m s-2' -> '9.80665 m * s**-2'
    """
    if not latex_str:
        return ""

    s = latex_str.strip()

    # Remove enclosing math markers ($ or $$ or \[ \])
    s = re.sub(r"^\$\$|\$\$$|^\\\[|\\\]$|^\$|\$$", "", s).strip()

    # Normalize unicode symbols
    s = s.replace("\u2212", "-").replace("×", " * ").replace("·", " * ").replace("•", " * ")

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

    # Convert LaTeX powers: ^{ -2 } -> **-2, ^2 -> **2
    s = re.sub(r"\^\{([^{}]+)\}", r"**\1", s)
    s = re.sub(r"\^([0-9\-+]+)", r"**\1", s)

    # Protect scientific notation (e.g. 1.05457e-34, 6.022e23, 1E+10)
    sci_matches: list[str] = []

    def _sci_sub(m: re.Match[str]) -> str:
        sci_matches.append(m.group(0))
        return f"__SCI_{len(sci_matches) - 1}__"

    s = re.sub(r"(?<=\d)[eE][+-]?\d+", _sci_sub, s)

    # Convert implicit powers on unit tokens: e.g. mol-1 -> mol**-1, s-2 -> s**-2, m2 -> m**2
    s = re.sub(r"([a-zA-Z]+)([-+]\d+)", r"\1**\2", s)

    # Restore scientific notation
    for i, orig in enumerate(sci_matches):
        s = s.replace(f"__SCI_{i}__", orig)

    # Clean double spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def check_latex_math_balance(text: str) -> tuple[bool, list[str]]:
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
        errors.append(
            f"Unbalanced curly braces '{{}}' (open: {open_braces}, close: {close_braces})."
        )

    return (len(errors) == 0, errors)
