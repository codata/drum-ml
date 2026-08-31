"""Scaffold and Augmented Instruction-Response Data Models."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ArchetypeType(StrEnum):
    """The 6 Pedagogical Metrology Archetypes."""

    DIRECT_IDENTIFICATION = "direct_identification"  # Unit <-> Symbol, QuantityKind <-> SI Unit
    DIMENSIONAL_DECOMPOSITION = "dimensional_decomposition"  # Base SI decomposition & derivation
    CONVERSION_SCALING = "conversion_scaling"  # Multipliers, offsets, compound units
    DIMENSIONAL_ERROR_DETECTION = (
        "error_detection"  # Homogeneity violation, invalid sum, bad prefix
    )
    SEMANTIC_TOOL_USE = "semantic_tool_use"  # SPARQL, QUDT queries, Pint code execution
    METROLOGICAL_UNCERTAINTY = "metrological_uncertainty"  # GUM uncertainty propagation, sig-figs


class PersonaType(StrEnum):
    """Linguistic and domain persona profiles representing CODATA DRUM Scientific Unions & NMIs."""

    # General & Engineering Baseline
    GENERAL_USER = "general_user"
    PHYSICS_STUDENT = "physics_student"
    FIRMWARE_IOT_ENGINEER = "firmware_iot_engineer"
    DATA_SCIENTIST_ANALYST = "data_scientist_analyst"
    ISO_COMPLIANCE_AUDITOR = "iso_compliance_auditor"
    ACADEMIC_METROLOGIST = "academic_metrologist"

    # National Metrology Institutes & Global Science Council
    NIST_METROLOGIST = "nist_metrologist"  # National Institute of Standards and Technology (NIST)
    NRC_METROLOGIST = "nrc_metrologist"  # National Research Council Canada (NRC)
    ISC_SCIENCE_POLICY = "isc_science_policy"  # International Science Council (ISC)

    # Physical, Chemical & Mathematical Sciences
    IUPAP_PHYSICIST = "iupap_physicist"  # International Union of Pure and Applied Physics (IUPAP)
    IUPAC_CHEMIST = "iupac_chemist"  # International Union of Pure and Applied Chemistry (IUPAC)
    IAU_ASTRONOMER = "iau_astronomer"  # International Astronomical Union (IAU)
    IUCR_CRYSTALLOGRAPHER = "iucr_crystallographer"  # International Union of Crystallography (IUCr)
    IMU_MATHEMATICIAN = "imu_mathematician"  # International Mathematical Union (IMU)
    URSI_RADIO_SCIENTIST = "ursi_radio_scientist"  # Union Radio Scientifique Internationale (URSI)
    IUTAM_MECHANICS_ENGINEER = "iutam_mechanics_engineer"  # International Union of Theoretical and Applied Mechanics (IUTAM)
    AEROSPACE_PROPULSION_ENGINEER = "aerospace_propulsion_engineer"
    PARTICLE_PHYSICIST = "particle_physicist"

    # Earth, Space, Geo & Environmental Sciences
    IUGG_GEODESIST_GEOPHYSICIST = (
        "iugg_geodesist_geophysicist"  # International Union of Geodesy and Geophysics (IUGG)
    )
    IGU_GEOGRAPHER = "igu_geographer"  # International Geographical Union (IGU)
    ISPRS_PHOTOGRAMMETRIST = "isprs_photogrammetrist"  # International Society for Photogrammetry and Remote Sensing (ISPRS)
    ISDE_DIGITAL_EARTH = "isde_digital_earth"  # International Society for Digital Earth (ISDE)
    IUSS_SOIL_SCIENTIST = "iuss_soil_scientist"  # International Union of Soil Science (IUSS)
    ENERGY_ENVIRONMENTAL_SCIENTIST = "energy_environmental_scientist"

    # Biological, Medical & Health Sciences
    IUBS_BIOLOGIST = "iubs_biologist"  # International Union of Biological Sciences (IUBS)
    IUIS_IMMUNOLOGIST = "iuis_immunologist"  # International Union of Immunological Societies (IUIS)
    IUPHAR_PHARMACOLOGIST = (
        "iuphar_pharmacologist"  # International Union of Basic and Clinical Pharmacology (IUPHAR)
    )
    IUPS_PHYSIOLOGIST = "iups_physiologist"  # International Union of Physiological Sciences (IUPS)
    IUTOX_TOXICOLOGIST = "iutox_toxicologist"  # International Union of Toxicology (IUTOX)
    IUNS_NUTRITIONIST = "iuns_nutritionist"  # International Union of Nutritional Sciences (IUNS)
    IUFOST_FOOD_SCIENTIST = (
        "iufost_food_scientist"  # International Union of Food Science and Technology (IUFoST)
    )
    IOMP_MEDICAL_PHYSICIST = "iomp_medical_physicist"  # IUPESM / IOMP (Medical Physics)
    IFMBE_BIOMEDICAL_ENGINEER = (
        "ifmbe_biomedical_engineer"  # IUPESM / IFMBE (Biomedical Engineering)
    )

    # Social, Behavioral & Human Sciences
    IUPSYS_PSYCHOLOGIST = (
        "iupsys_psychologist"  # International Union of Psychological Science (IUPsyS)
    )
    ISA_SOCIOLOGIST = "isa_sociologist"  # International Sociological Association (ISA)
    IUSSP_DEMOGRAPHER = (
        "iussp_demographer"  # International Union for the Scientific Study of Population (IUSSP)
    )
    WAU_ANTHROPOLOGIST = "wau_anthropologist"  # World Anthropological Union (WAU)
    FOUR_S_SCIENCE_STUDIES = "four_s_science_studies"  # Society for Social Studies of Science (4S)
    IUHPST_HISTORIAN_PHILOSOPHER = "iuhpst_historian_philosopher"  # International Union for History and Philosophy of Science and Technology (IUHPST)


class DifficultyTier(StrEnum):
    INTRODUCTORY = "introductory"  # High school / basic lookup
    INTERMEDIATE = "intermediate"  # Undergraduate physics / engineering calculation
    ADVANCED = "advanced"  # Formal metrology / standards lab level


class ScaffoldRecord(BaseModel):
    """Canonical ground-truth template record generated from metrological entities."""

    id: str
    archetype: ArchetypeType
    entity_uri: str
    quantity_kind_uri: str | None = None
    canonical_query: str
    ground_truth_answer: str
    context_facts: dict[str, str] = Field(default_factory=dict)
    latex_expressions: list[str] = Field(default_factory=list)
    numerical_constants: dict[str, str] = Field(default_factory=dict)
    difficulty: DifficultyTier = DifficultyTier.INTERMEDIATE


class AugmentedRecord(BaseModel):
    """Augmented dataset candidate generated via LLM persona prompting."""

    id: str
    scaffold_id: str
    archetype: ArchetypeType
    persona: PersonaType
    user_query: str
    ground_truth_answer: str
    entity_uri: str
    quantity_kind_uri: str | None = None
    temperature: float = 0.7
    llm_generator: str
    metadata: dict[str, Any] = Field(default_factory=dict)
