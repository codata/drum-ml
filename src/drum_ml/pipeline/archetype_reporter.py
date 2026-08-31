"""Pedagogical Metrology Archetype Reporting and Analytics."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from drum_ml.models.scaffolds import ArchetypeType, AugmentedRecord

ARCHETYPE_METADATA = {
    ArchetypeType.DIRECT_IDENTIFICATION: {
        "title": "Direct Identification & Symbol Mapping",
        "description": "Forward and reverse mapping between unit names, official SI/QUDT symbols, quantity kinds, and defining constants.",
        "eval_objective": "Zero-hallucination factual recall of official metrological symbols and quantity associations.",
    },
    ArchetypeType.DIMENSIONAL_DECOMPOSITION: {
        "title": "Dimensional Decomposition & Base SI",
        "description": "Rigorous derivation expanding derived units into the 7 base SI dimensions [L, M, T, I, Theta, N, J].",
        "eval_objective": "Mathematical reasoning and formal SI base unit realization.",
    },
    ArchetypeType.CONVERSION_SCALING: {
        "title": "Conversion Chains, Scaling & Temperature Offsets",
        "description": "Exact multiplier scaling, affine temperature offsets (°C/°F/K), and multi-step conversion chains.",
        "eval_objective": "Numerical precision, exact arbitrary arithmetic, and offset formula execution.",
    },
    ArchetypeType.DIMENSIONAL_ERROR_DETECTION: {
        "title": "Dimensional Sanity & Error Detection",
        "description": "Detecting inhomogeneous equations, incompatible quantity additions, and case-sensitive unit traps (mN vs MN).",
        "eval_objective": "Adversarial robustness and physics sanity checking.",
    },
    ArchetypeType.SEMANTIC_TOOL_USE: {
        "title": "Metrological Tool Use & Serialization",
        "description": "Generating SPARQL 1.1 queries, QUDT/UCUM triples, and executable Pint/SymPy Python unit code.",
        "eval_objective": "Autonomous tool calling and semantic web code generation.",
    },
    ArchetypeType.METROLOGICAL_UNCERTAINTY: {
        "title": "Metrological Uncertainty & GUM Propagation",
        "description": "Calculating combined standard uncertainty u_c(y), expanded uncertainty budgets (U = k * u_c), and exact SI constants (u = 0).",
        "eval_objective": "ISO/IEC Guide 98-3 (GUM) compliance and experimental budget calculation.",
    },
}


class ArchetypeReporter:
    """Generates detailed reports, token metrics, and pedagogical evaluations across Archetypes."""

    @staticmethod
    def _approx_tokens(text: str) -> int:
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return max(1, int(len(tokens) * 1.1))

    @staticmethod
    def _calc_stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"min": 0, "mean": 0.0, "median": 0, "p95": 0, "max": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "min": sorted_vals[0],
            "mean": round(sum(sorted_vals) / n, 1),
            "median": sorted_vals[n // 2],
            "p95": sorted_vals[min(n - 1, int(n * 0.95))],
            "max": sorted_vals[-1],
        }

    @classmethod
    def analyze_records(cls, records: list[AugmentedRecord]) -> dict[str, Any]:
        """Calculates token geometry, math complexity, and persona spread per Archetype."""
        total_samples = len(records)
        archetype_groups = defaultdict(list)

        for rec in records:
            arch_key = (
                rec.archetype.value if hasattr(rec.archetype, "value") else str(rec.archetype)
            )
            archetype_groups[arch_key].append(rec)

        archetypes_summary = []
        for arch_key, group in sorted(
            archetype_groups.items(), key=lambda x: len(x[1]), reverse=True
        ):
            enum_val = None
            try:
                enum_val = ArchetypeType(arch_key)
            except ValueError:
                pass
            meta = ARCHETYPE_METADATA.get(
                enum_val,
                {
                    "title": arch_key.replace("_", " ").title(),
                    "description": "Pedagogical task archetype",
                    "eval_objective": "Metrological instruction evaluation",
                },
            )

            prompt_toks = []
            response_toks = []
            total_toks = []
            latex_inline = 0
            latex_display = 0
            personas = Counter()

            for rec in group:
                p_val = rec.persona.value if hasattr(rec.persona, "value") else str(rec.persona)
                personas[p_val] += 1

                p_t = cls._approx_tokens(rec.user_query)
                r_t = cls._approx_tokens(rec.ground_truth_answer)
                prompt_toks.append(p_t)
                response_toks.append(r_t)
                total_toks.append(p_t + r_t)

                if "$" in rec.user_query or "$" in rec.ground_truth_answer:
                    latex_inline += 1
                if "$$" in rec.ground_truth_answer:
                    latex_display += 1

            n = len(group)
            sample_previews = []
            for rec in group[:2]:
                p_val = rec.persona.value if hasattr(rec.persona, "value") else str(rec.persona)
                sample_previews.append(
                    {
                        "persona": p_val,
                        "query": rec.user_query,
                        "answer": (rec.ground_truth_answer[:160] + "...")
                        if len(rec.ground_truth_answer) > 160
                        else rec.ground_truth_answer,
                    }
                )

            archetypes_summary.append(
                {
                    "archetype_key": arch_key,
                    "title": meta["title"],
                    "description": meta["description"],
                    "eval_objective": meta["eval_objective"],
                    "sample_count": n,
                    "percentage": (n / max(1, total_samples)) * 100,
                    "unique_personas": len(personas),
                    "prompt_tokens": cls._calc_stats(prompt_toks),
                    "response_tokens": cls._calc_stats(response_toks),
                    "total_tokens": cls._calc_stats(total_toks),
                    "math_inline_pct": round((latex_inline / n) * 100, 1),
                    "math_display_pct": round((latex_display / n) * 100, 1),
                    "sample_previews": sample_previews,
                }
            )

        return {
            "total_samples": total_samples,
            "unique_archetypes": len(archetype_groups),
            "archetypes": archetypes_summary,
        }

    @classmethod
    def save_markdown_report(
        cls, report_data: dict[str, Any], output_path: str = "./dataset/archetype_report.md"
    ) -> Path:
        """Saves a publication-ready Markdown Archetype Report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# 🎯 CODATA DRUM-ML Pedagogical Archetype Report",
            "",
            f"**Total Samples Analyzed:** {report_data['total_samples']:,}  ",
            f"**Active Pedagogical Archetypes:** {report_data['unique_archetypes']} / 6  ",
            "",
            "---",
            "",
            "## 📊 Archetype Distribution & Training Token Geometry",
            "",
            "| Archetype | Samples | % Share | Avg Prompt | Avg Response | Avg Total | Math % | Primary Evaluation Focus |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]

        for a in report_data["archetypes"]:
            lines.append(
                f"| `{a['archetype_key']}` | **{a['sample_count']:,}** | {a['percentage']:.1f}% | "
                f"{a['prompt_tokens']['mean']} tok | {a['response_tokens']['mean']} tok | "
                f"**{a['total_tokens']['mean']} tok** | {a['math_inline_pct']}% | {a['title']} |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 🔬 In-Depth Archetype Profiles & Showcases",
                "",
            ]
        )

        for a in report_data["archetypes"]:
            lines.extend(
                [
                    f"### Archetype: {a['title']} (`{a['archetype_key']}`)",
                    f"- **Pedagogical Purpose:** {a['description']}",
                    f"- **Evaluation Objective:** {a['eval_objective']}",
                    f"- **Volume:** {a['sample_count']:,} samples ({a['percentage']:.1f}% of corpus across {a['unique_personas']} personas)",
                    f"- **Token Profile:** Prompt: {a['prompt_tokens']['mean']} tok (p95: {a['prompt_tokens']['p95']}) | Response: {a['response_tokens']['mean']} tok (p95: {a['response_tokens']['p95']})",
                    "",
                    "#### Showcase Samples:",
                ]
            )
            for idx, s in enumerate(a["sample_previews"], 1):
                lines.extend(
                    [
                        f"**Sample {idx} (Persona: `{s['persona']}`):**",
                        f'> **User Query:** *"{s["query"]}"*  ',
                        f"> **Assistant Answer:** {s['answer']}",
                        "",
                    ]
                )
            lines.append("---")
            lines.append("")

        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return out

    @classmethod
    def save_json_report(
        cls, report_data: dict[str, Any], output_path: str = "./dataset/archetype_report.json"
    ) -> Path:
        """Saves structured archetype JSON analytics."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        return out

    @classmethod
    def print_rich_table(cls, report_data: dict[str, Any], console: Console | None = None) -> None:
        """Renders an interactive Rich Archetype table to the terminal."""
        con = console or Console()
        table = Table(
            title=f"🎯 DRUM-ML Archetype Report (Total Samples: {report_data['total_samples']:,})",
            header_style="bold cyan",
        )
        table.add_column("Archetype Key", style="cyan", no_wrap=True)
        table.add_column("Pedagogical Focus", style="bold white")
        table.add_column("Samples", justify="right", style="green")
        table.add_column("Share", justify="right", style="magenta")
        table.add_column("Avg Prompt", justify="right", style="yellow")
        table.add_column("Avg Response", justify="right", style="bold green")
        table.add_column("Math %", justify="right", style="cyan")

        for a in report_data["archetypes"]:
            table.add_row(
                a["archetype_key"],
                a["title"],
                f"{a['sample_count']:,}",
                f"{a['percentage']:.1f}%",
                f"{a['prompt_tokens']['mean']} tok",
                f"{a['response_tokens']['mean']} tok",
                f"{a['math_inline_pct']}%",
            )

        con.print(table)
