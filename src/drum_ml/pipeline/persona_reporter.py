"""Persona Reporting and Analytics for Generated Datasets."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.table import Table
from drum_ml.models.scaffolds import AugmentedRecord, PersonaType


PERSONA_METADATA = {
    PersonaType.GENERAL_USER: {
        "organization": "General Public / Direct Inquiry",
        "domain": "Everyday physical lookups & basic unit queries",
    },
    PersonaType.PHYSICS_STUDENT: {
        "organization": "Academic Physics Student",
        "domain": "Conceptual homework, dimensional sanity, and unit derivations",
    },
    PersonaType.ACADEMIC_METROLOGIST: {
        "organization": "BIPM / National Metrology Institutes",
        "domain": "BIPM 9th Edition SI brochure, VIM3 formal metrology & LaTeX",
    },
    PersonaType.NIST_METROLOGIST: {
        "organization": "National Institute of Standards and Technology (NIST)",
        "domain": "NIST SP 330/811, Kibble balance, optical clocks, primary standards",
    },
    PersonaType.NRC_METROLOGIST: {
        "organization": "National Research Council Canada (NRC)",
        "domain": "Quantum Hall, Josephson voltage, mass dissemination, electrical standards",
    },
    PersonaType.ISC_SCIENCE_POLICY: {
        "organization": "International Science Council (ISC)",
        "domain": "Cross-disciplinary data interoperability, FAIR digital units",
    },
    PersonaType.IUPAP_PHYSICIST: {
        "organization": "International Union of Pure and Applied Physics (IUPAP)",
        "domain": "SUNAMCO symbols and units, base dimensions, SI realizations",
    },
    PersonaType.IUPAC_CHEMIST: {
        "organization": "International Union of Pure and Applied Chemistry (IUPAC)",
        "domain": "IUPAC Green Book, molar amounts, amount concentrations, gas constant",
    },
    PersonaType.IAU_ASTRONOMER: {
        "organization": "International Astronomical Union (IAU)",
        "domain": "Parsecs, astronomical units, solar masses (M_sun), Jansky flux",
    },
    PersonaType.IUCR_CRYSTALLOGRAPHER: {
        "organization": "International Union of Crystallography (IUCr)",
        "domain": "Unit cell dimensions (Å, pm), reciprocal lattices, electron density",
    },
    PersonaType.IMU_MATHEMATICIAN: {
        "organization": "International Mathematical Union (IMU)",
        "domain": "Dimensional vector spaces, Lie algebra symmetries, Buckingham Pi theorem",
    },
    PersonaType.URSI_RADIO_SCIENTIST: {
        "organization": "Union Radio Scientifique Internationale (URSI)",
        "domain": "RF telemetry, antenna gain (dBi), noise temperature (K), Poynting vectors",
    },
    PersonaType.IUTAM_MECHANICS_ENGINEER: {
        "organization": "International Union of Theoretical and Applied Mechanics (IUTAM)",
        "domain": "Stress tensors (Pa), dynamic/kinematic viscosity, Reynolds numbers",
    },
    PersonaType.AEROSPACE_PROPULSION_ENGINEER: {
        "organization": "Aerospace & Propulsion Engineering",
        "domain": "Specific impulse (I_sp), Mach numbers, knots, dynamic pressure",
    },
    PersonaType.PARTICLE_PHYSICIST: {
        "organization": "High-Energy & Particle Physics",
        "domain": "Electronvolts (eV/GeV), barn (b) cross-sections, Planck natural units",
    },
    PersonaType.IUGG_GEODESIST_GEOPHYSICIST: {
        "organization": "International Union of Geodesy and Geophysics (IUGG)",
        "domain": "Gravity anomalies (mGal), geoid heights, seismic moments, geomagnetic field (nT)",
    },
    PersonaType.IGU_GEOGRAPHER: {
        "organization": "International Geographical Union (IGU)",
        "domain": "Spatial scale, map projections, land area units (ha, km²), geospatial scaling",
    },
    PersonaType.ISPRS_PHOTOGRAMMETRIST: {
        "organization": "International Society for Photogrammetry and Remote Sensing (ISPRS)",
        "domain": "Ground sampling distance (cm/px), spectral radiance, LiDAR point density",
    },
    PersonaType.ISDE_DIGITAL_EARTH: {
        "organization": "International Society for Digital Earth (ISDE)",
        "domain": "Discrete Global Grid Systems (DGGS), planetary telemetry harmonization",
    },
    PersonaType.IUSS_SOIL_SCIENTIST: {
        "organization": "International Union of Soil Science (IUSS)",
        "domain": "Soil bulk density (g/cm³), cation exchange capacity, hydraulic conductivity",
    },
    PersonaType.ENERGY_ENVIRONMENTAL_SCIENTIST: {
        "organization": "Energy Systems & Climate Science",
        "domain": "Carbon intensity (gCO2e/kWh), trace mixing ratios (ppmv), solar irradiance (W/m²)",
    },
    PersonaType.IUBS_BIOLOGIST: {
        "organization": "International Union of Biological Sciences (IUBS)",
        "domain": "Metabolic scaling rates (W/kg), biomass density, organismal rates",
    },
    PersonaType.IUIS_IMMUNOLOGIST: {
        "organization": "International Union of Immunological Societies (IUIS)",
        "domain": "International Units (IU/mL) of biological activity, antibody titers, cytokines",
    },
    PersonaType.IUPHAR_PHARMACOLOGIST: {
        "organization": "International Union of Basic and Clinical Pharmacology (IUPHAR)",
        "domain": "PK/PD parameters, clearance (mL/min/kg), half-life, receptor binding affinity",
    },
    PersonaType.IUPS_PHYSIOLOGIST: {
        "organization": "International Union of Physiological Sciences (IUPS)",
        "domain": "Cardiac output (L/min), GFR, membrane potential (mV), blood pressure",
    },
    PersonaType.IUTOX_TOXICOLOGIST: {
        "organization": "International Union of Toxicology (IUTOX)",
        "domain": "LD50 (mg/kg), LC50 (mg/m³), acceptable daily intake (ADI), threshold limits",
    },
    PersonaType.IUNS_NUTRITIONIST: {
        "organization": "International Union of Nutritional Sciences (IUNS)",
        "domain": "Dietary reference intakes, kcal vs kJ, retinol equivalents, glycemic index",
    },
    PersonaType.IUFOST_FOOD_SCIENTIST: {
        "organization": "International Union of Food Science and Technology (IUFoST)",
        "domain": "Water activity (a_w), pasteurization lethality (F0), thermal death time (D-val)",
    },
    PersonaType.IOMP_MEDICAL_PHYSICIST: {
        "organization": "IUPESM / IOMP (Medical Physics)",
        "domain": "Absorbed radiation dose (Gray, Gy), biological dose (Sievert, Sv), Kerma product",
    },
    PersonaType.IFMBE_BIOMEDICAL_ENGINEER: {
        "organization": "IUPESM / IFMBE (Biomedical Engineering)",
        "domain": "Biosensor impedance (Ω·cm²), IEEE 11073 medical device metrics, UCUM codes",
    },
    PersonaType.IUPSYS_PSYCHOLOGIST: {
        "organization": "International Union of Psychological Science (IUPsyS)",
        "domain": "Reaction time latency (ms), psychophysical thresholds (JND, dB), z-scores",
    },
    PersonaType.ISA_SOCIOLOGIST: {
        "organization": "International Sociological Association (ISA)",
        "domain": "Socio-economic indexes, Gini inequality coefficients, demographic rates",
    },
    PersonaType.IUSSP_DEMOGRAPHER: {
        "organization": "International Union for the Scientific Study of Population (IUSSP)",
        "domain": "Total fertility rates, infant mortality rates, life expectancy, migration",
    },
    PersonaType.WAU_ANTHROPOLOGIST: {
        "organization": "World Anthropological Union (WAU)",
        "domain": "Cranial morphology (mm), radiocarbon dating (years BP), ethno-metrology systems",
    },
    PersonaType.FOUR_S_SCIENCE_STUDIES: {
        "organization": "Society for Social Studies of Science (4S)",
        "domain": "Sociotechnical infrastructure of SI redefinition, metrological standardization",
    },
    PersonaType.IUHPST_HISTORIAN_PHILOSOPHER: {
        "organization": "International Union for History and Philosophy of Science & Tech (IUHPST)",
        "domain": "Epistemology of measurement, operationalism vs realism, 1875 Metre Convention",
    },
    PersonaType.FIRMWARE_IOT_ENGINEER: {
        "organization": "Embedded Systems & IoT Engineering",
        "domain": "Sensor registers, ADC scaling, fixed-point integer conversions, UCUM formatting",
    },
    PersonaType.DATA_SCIENTIST_ANALYST: {
        "organization": "Data Science & Machine Learning",
        "domain": "Feature unit normalization, telemetry validation, Pandas/NumPy unit metadata",
    },
    PersonaType.ISO_COMPLIANCE_AUDITOR: {
        "organization": "ISO / Standards Calibration Auditor",
        "domain": "ISO 17025 calibration certificates, GUM combined uncertainty budgets, traceability",
    },
}


class PersonaReporter:
    """Generates detailed reports and analytics on generated dataset personas."""

    @staticmethod
    def analyze_records(records: List[AugmentedRecord]) -> Dict[str, Any]:
        """Calculates comprehensive distribution metrics across personas and archetypes."""
        total_samples = len(records)
        persona_counts = defaultdict(int)
        archetype_counts = defaultdict(int)
        persona_archetype_matrix = defaultdict(lambda: defaultdict(int))
        persona_samples = defaultdict(list)
        generators = defaultdict(int)

        for rec in records:
            p_key = rec.persona.value if hasattr(rec.persona, "value") else str(rec.persona)
            arch_key = rec.archetype.value if hasattr(rec.archetype, "value") else str(rec.archetype)

            persona_counts[p_key] += 1
            archetype_counts[arch_key] += 1
            persona_archetype_matrix[p_key][arch_key] += 1
            generators[rec.llm_generator] += 1

            if len(persona_samples[p_key]) < 2:
                persona_samples[p_key].append({
                    "query": rec.user_query,
                    "archetype": arch_key,
                    "entity_uri": rec.entity_uri,
                    "answer_preview": (rec.ground_truth_answer[:120] + "...") if len(rec.ground_truth_answer) > 120 else rec.ground_truth_answer,
                })

        # Build persona breakdown list
        personas_summary = []
        for p_key, count in sorted(persona_counts.items(), key=lambda x: x[1], reverse=True):
            enum_val = None
            try:
                enum_val = PersonaType(p_key)
            except ValueError:
                pass
            meta = PERSONA_METADATA.get(enum_val, {
                "organization": p_key.replace("_", " ").title(),
                "domain": "Domain-specific metrology",
            })

            personas_summary.append({
                "persona_key": p_key,
                "organization": meta["organization"],
                "domain": meta["domain"],
                "sample_count": count,
                "percentage": (count / max(1, total_samples)) * 100,
                "archetype_breakdown": dict(persona_archetype_matrix[p_key]),
                "sample_previews": persona_samples[p_key],
            })

        return {
            "total_samples": total_samples,
            "unique_personas_count": len(persona_counts),
            "archetype_distribution": dict(archetype_counts),
            "generators": dict(generators),
            "personas": personas_summary,
        }

    @classmethod
    def save_markdown_report(
        cls, report_data: Dict[str, Any], output_path: str = "./dataset/persona_report.md"
    ) -> Path:
        """Saves a comprehensive markdown report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# CODATA DRUM-ML Persona & Domain Distribution Report",
            "",
            f"**Total Samples:** {report_data['total_samples']:,}  ",
            f"**Active Personas:** {report_data['unique_personas_count']}  ",
            f"**Generators:** {', '.join(report_data['generators'].keys()) or 'Deterministic Ground-Truth'}  ",
            "",
            "---",
            "",
            "## 📊 Persona & Scientific Union Breakdown",
            "",
            "| Persona Key | Scientific Union / Organization | Samples | % Total | Domain Coverage |",
            "|---|---|---:|---:|---|",
        ]

        for p in report_data["personas"]:
            lines.append(
                f"| `{p['persona_key']}` | **{p['organization']}** | {p['sample_count']:,} | {p['percentage']:.1f}% | {p['domain']} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 🎯 Pedagogical Archetype Distribution",
            "",
            "| Archetype | Count | % Share |",
            "|---|---:|---:|",
        ])

        total = max(1, report_data["total_samples"])
        for arch, count in report_data["archetype_distribution"].items():
            lines.append(f"| `{arch}` | {count:,} | {(count/total)*100:.1f}% |")

        lines.extend([
            "",
            "---",
            "",
            "## 🔍 Representative Sample Queries by Scientific Union",
            "",
        ])

        for p in report_data["personas"]:
            lines.append(f"### `{p['persona_key']}` ({p['organization']})")
            lines.append(f"- **Domain Focus:** {p['domain']}")
            for idx, s in enumerate(p["sample_previews"], 1):
                lines.append(f"  {idx}. *\"{s['query']}\"* (Archetype: `{s['archetype']}`)")
            lines.append("")

        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return out

    @classmethod
    def save_json_report(
        cls, report_data: Dict[str, Any], output_path: str = "./dataset/persona_report.json"
    ) -> Path:
        """Saves machine-readable JSON analytics."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        return out

    @classmethod
    def print_rich_table(cls, report_data: Dict[str, Any], console: Optional[Console] = None) -> None:
        """Renders an interactive Rich summary table to the terminal."""
        con = console or Console()
        table = Table(
            title=f"DRUM-ML Persona Report (Total Samples: {report_data['total_samples']:,})",
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("Persona Key", style="cyan", no_wrap=True)
        table.add_column("Scientific Union / Organization", style="bold white")
        table.add_column("Samples", justify="right", style="green")
        table.add_column("Share", justify="right", style="magenta")
        table.add_column("Domain Focus", style="dim")

        for p in report_data["personas"]:
            table.add_row(
                p["persona_key"],
                p["organization"],
                f"{p['sample_count']:,}",
                f"{p['percentage']:.1f}%",
                p["domain"],
            )

        con.print(table)
