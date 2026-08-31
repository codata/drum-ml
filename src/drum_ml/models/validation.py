"""Validation Gate and Audit Result Data Models."""

from pydantic import BaseModel, Field


class Tier1SyntaxReport(BaseModel):
    """LaTeX delimiters, bracket balance, JSON escaping check."""

    passed: bool = True
    balanced_delimiters: bool = True
    valid_latex_syntax: bool = True
    clean_json_escaping: bool = True
    errors: list[str] = Field(default_factory=list)


class Tier2SymbolicReport(BaseModel):
    """Pint / SymPy algebraic equivalence and dimensional homogeneity check."""

    passed: bool = True
    dimensions_homogeneous: bool = True
    quantity_kind_compatible: bool = True
    symbolic_equivalence_verified: bool = True
    errors: list[str] = Field(default_factory=list)


class Tier3PrecisionReport(BaseModel):
    """Arbitrary-precision SI constant and conversion multiplier verification."""

    passed: bool = True
    exact_constants_preserved: bool = True
    zero_rounding_drift: bool = True
    errors: list[str] = Field(default_factory=list)


class Tier4CodeReport(BaseModel):
    """SPARQL and Python unit code syntax & execution test."""

    passed: bool = True
    sparql_syntax_valid: bool = True
    python_ast_valid: bool = True
    sandbox_execution_success: bool = True
    errors: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Aggregated validation result across all 4 deterministic audit tiers."""

    record_id: str
    passed: bool
    is_dpo_eligible: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    tier1_syntax: Tier1SyntaxReport = Field(default_factory=Tier1SyntaxReport)
    tier2_symbolic: Tier2SymbolicReport = Field(default_factory=Tier2SymbolicReport)
    tier3_precision: Tier3PrecisionReport = Field(default_factory=Tier3PrecisionReport)
    tier4_code: Tier4CodeReport = Field(default_factory=Tier4CodeReport)
    corrected_response: str | None = None
