"""Canonical Metrological Entity Models.

Enforces strict separation between QuantityKind (physical property) and Unit (measurement reference convention),
alongside PhysicalConstant and DimensionVector according to VIM3, ISO 80000, and QUDT 2.1.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class BaseDimension(StrEnum):
    """The 7 base dimensions of the International System of Quantities (ISQ)."""

    LENGTH = "L"  # Length (meter, m)
    MASS = "M"  # Mass (kilogram, kg)
    TIME = "T"  # Time (second, s)
    ELECTRIC_CURRENT = "I"  # Electric Current (ampere, A)
    THERMODYNAMIC_TEMP = "Theta"  # Thermodynamic Temperature (kelvin, K)
    AMOUNT_OF_SUBSTANCE = "N"  # Amount of Substance (mole, mol)
    LUMINOUS_INTENSITY = "J"  # Luminous Intensity (candela, cd)


class DimensionVector(BaseModel):
    """SI base dimensional representation [L, M, T, I, Theta, N, J] with integer/rational exponents."""

    L: int = 0
    M: int = 0
    T: int = 0
    I: int = 0  # noqa: E741
    Theta: int = 0
    N: int = 0
    J: int = 0

    def is_dimensionless(self) -> bool:
        """Return True if all dimension powers are zero."""
        return all(v == 0 for v in self.model_dump().values())

    def to_latex(self) -> str:
        """Returns LaTeX dimension formula, e.g., \\text{L}\\cdot\\text{M}\\cdot\\text{T}^{-2}"""
        if self.is_dimensionless():
            return "1"
        terms = []
        for dim, power in self.model_dump().items():
            if power == 0:
                continue
            sym = "\\Theta" if dim == "Theta" else f"\\text{{{dim}}}"
            if power == 1:
                terms.append(sym)
            else:
                terms.append(f"{sym}^{{{power}}}")
        return " \\cdot ".join(terms) if terms else "1"

    def to_si_base_unit_latex(self) -> str:
        """Returns SI base unit formula in standard metrology order (kg, m, s, A, K, mol, cd)."""
        if self.is_dimensionless():
            return "1"
        unit_map = {"M": "kg", "L": "m", "T": "s", "I": "A", "Theta": "K", "N": "mol", "J": "cd"}
        order = ["M", "L", "T", "I", "Theta", "N", "J"]
        terms = []
        for dim in order:
            power = getattr(self, dim)
            if power == 1:
                terms.append(f"\\text{{{unit_map[dim]}}}")
            elif power != 0:
                terms.append(f"\\text{{{unit_map[dim]}}}^{{{power}}}")
        return " \\cdot ".join(terms) if terms else "1"

    def __mul__(self, other: "DimensionVector") -> "DimensionVector":
        return DimensionVector(
            L=self.L + other.L,
            M=self.M + other.M,
            T=self.T + other.T,
            I=self.I + other.I,
            Theta=self.Theta + other.Theta,
            N=self.N + other.N,
            J=self.J + other.J,
        )

    def __truediv__(self, other: "DimensionVector") -> "DimensionVector":
        return DimensionVector(
            L=self.L - other.L,
            M=self.M - other.M,
            T=self.T - other.T,
            I=self.I - other.I,
            Theta=self.Theta - other.Theta,
            N=self.N - other.N,
            J=self.J - other.J,
        )

    def __pow__(self, power: int) -> "DimensionVector":
        return DimensionVector(
            L=self.L * power,
            M=self.M * power,
            T=self.T * power,
            I=self.I * power,
            Theta=self.Theta * power,
            N=self.N * power,
            J=self.J * power,
        )


class QuantityKindEntity(BaseModel):
    """VIM3 Quantity Kind: An abstract physical property (e.g., Torque, Energy, Absorbed Dose).

    Note: Distinct QuantityKinds can share identical DimensionVectors (e.g. Torque vs Energy both [L^2 M T^-2]),
    making QuantityKind a mandatory discriminator in metrological reasoning.
    """

    uri: str
    label: str
    symbol: str | None = None
    description: str | None = None
    dimension_vector: DimensionVector
    applicable_units: list[str] = Field(default_factory=list)  # Unit URIs
    broader_quantity_kinds: list[str] = Field(default_factory=list)
    exact_match_uris: list[str] = Field(default_factory=list)


class ConversionRelation(BaseModel):
    """Conversion relationship from a unit to its SI coherent derived/base equivalent:
    SI_val = (val * multiplier) + offset.
    """

    multiplier: float
    offset: float = 0.0
    exact: bool = False
    conversion_formula: str | None = None


class UnitEntity(BaseModel):
    """VIM3 Unit: A real scalar quantity adopted by convention to express values of quantities of the same kind."""

    uri: str
    symbol: str
    label: str
    description: str | None = None
    is_si_base: bool = False
    is_si_derived: bool = False
    is_coherent: bool = True
    dimension_vector: DimensionVector
    has_quantity_kinds: list[str] = Field(default_factory=list)  # QuantityKind URIs
    ucum_code: str | None = None
    iec_symbol: str | None = None
    latex_symbol: str | None = None
    conversion: ConversionRelation | None = None
    exact_match_uris: list[str] = Field(default_factory=list)


class ConstantCategory(StrEnum):
    EXACT_SI_DEFINING = "exact_si_defining"  # e.g. c, h, e, k, N_A, Delta_nu_Cs, K_cd
    CODATA_RECOMMENDED = "codata_recommended"  # e.g. G, alpha, m_e, R_inf


class PhysicalConstantEntity(BaseModel):
    """CODATA / BIPM Fundamental Physical Constant representation with exact SI flags & uncertainty budgets."""

    uri: str
    name: str
    symbol: str
    latex_symbol: str
    category: ConstantCategory
    numeric_value: str  # String to preserve arbitrary precision without float truncation
    standard_uncertainty: str | None = None  # None for exact defining constants
    relative_uncertainty: str | None = None
    unit_symbol: str
    unit_uri: str | None = None
    quantity_kind_uri: str | None = None
    dimension_vector: DimensionVector
    defining_year: int = 2019
    description: str | None = None


class CanonicalEntityStore(BaseModel):
    """Container for all canonical ingested metrological entities."""

    units: dict[str, UnitEntity] = Field(default_factory=dict)
    quantity_kinds: dict[str, QuantityKindEntity] = Field(default_factory=dict)
    constants: dict[str, PhysicalConstantEntity] = Field(default_factory=dict)
