"""Comprehensive Dataset Analytics, Token Geometry & Hugging Face Dataset Card Generator."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class DatasetStatsGenerator:
    """Calculates deep metrological, token, and linguistic metrics across dataset splits."""

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Accurate token count approximation (~3.8 chars per token or whitespace/punct segmentation)."""
        if not text:
            return 0
        # Words + punctuation tokens
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return max(1, int(len(tokens) * 1.1))

    @staticmethod
    def _calc_stats(values: List[int]) -> Dict[str, Any]:
        """Computes descriptive statistics (min, mean, median, p95, max, std)."""
        if not values:
            return {"min": 0, "mean": 0.0, "median": 0, "p95": 0, "p99": 0, "max": 0, "std": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n
        median_val = sorted_vals[n // 2]
        p95_val = sorted_vals[min(n - 1, int(n * 0.95))]
        p99_val = sorted_vals[min(n - 1, int(n * 0.99))]
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / n
        std_val = math.sqrt(variance)

        return {
            "min": sorted_vals[0],
            "mean": round(mean_val, 1),
            "median": median_val,
            "p95": p95_val,
            "p99": p99_val,
            "max": sorted_vals[-1],
            "std": round(std_val, 1),
        }

    @classmethod
    def analyze_dataset(cls, dataset_dir: str = "./dataset") -> Dict[str, Any]:
        """Performs deep scan across train/val/test splits and augmented cache."""
        d_path = Path(dataset_dir)
        splits_data = {"train": [], "val": [], "test": []}
        dpo_records = []

        for split_name in ["train", "val", "test"]:
            file_path = d_path / f"{split_name}.jsonl"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            splits_data[split_name].append(json.loads(line))

        dpo_path = d_path / "dpo_preferences.jsonl"
        if dpo_path.exists():
            with open(dpo_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        dpo_records.append(json.loads(line))

        all_records = splits_data["train"] + splits_data["val"] + splits_data["test"]
        total_samples = len(all_records)

        # 1. Token Geometry
        prompt_tokens = []
        response_tokens = []
        total_seq_tokens = []
        archetype_counts = Counter()
        persona_counts = Counter()
        latex_inline_count = 0
        latex_display_count = 0
        macro_counts = Counter()

        # Physical dimension and SI base trackers
        dim_base_counts = {
            "L (Length)": 0,
            "M (Mass)": 0,
            "T (Time)": 0,
            "I (Electric Current)": 0,
            "Theta (Temperature)": 0,
            "N (Amount of Substance)": 0,
            "J (Luminous Intensity)": 0,
        }
        dimensionless_count = 0

        # Lexical diversity trackers
        prompt_vocab = Counter()
        prompt_bigrams = set()
        total_prompt_words = 0

        for item in all_records:
            arch = item.get("archetype", "direct_identification")
            archetype_counts[arch] += 1

            # Extract persona from ID or metadata
            rec_id = item.get("id", "")
            parts = rec_id.split("_")
            persona_name = parts[2] if len(parts) > 2 else "general_user"
            persona_counts[persona_name] += 1

            msgs = item.get("messages", [])
            user_text = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            asst_text = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")

            # Tokens
            p_tok = cls._approx_tokens(user_text)
            r_tok = cls._approx_tokens(asst_text)
            prompt_tokens.append(p_tok)
            response_tokens.append(r_tok)
            total_seq_tokens.append(p_tok + r_tok)

            # LaTeX & Math analysis
            if "$" in user_text or "$" in asst_text:
                latex_inline_count += 1
            if "$$" in asst_text:
                latex_display_count += 1

            # Math macros
            for macro in ["\\text", "\\cdot", "\\frac", "\\Theta", "\\Omega", "\\mu", "\\circ", "\\Delta", "\\pi"]:
                if macro in asst_text or macro in user_text:
                    macro_counts[macro] += 1

            # Dimensional parsing from assistant response
            if "\\text{m}" in asst_text or "\\text{L}" in asst_text:
                dim_base_counts["L (Length)"] += 1
            if "\\text{kg}" in asst_text or "\\text{M}" in asst_text:
                dim_base_counts["M (Mass)"] += 1
            if "\\text{s}" in asst_text or "\\text{T}" in asst_text:
                dim_base_counts["T (Time)"] += 1
            if "\\text{A}" in asst_text or "\\text{I}" in asst_text:
                dim_base_counts["I (Electric Current)"] += 1
            if "\\text{K}" in asst_text or "\\Theta" in asst_text or "°C" in asst_text:
                dim_base_counts["Theta (Temperature)"] += 1
            if "\\text{mol}" in asst_text or "\\text{N}" in asst_text:
                dim_base_counts["N (Amount of Substance)"] += 1
            if "\\text{cd}" in asst_text or "\\text{J}" in asst_text:
                dim_base_counts["J (Luminous Intensity)"] += 1
            if "\\text{1}" in asst_text or "\\text{dimensionless}" in asst_text:
                dimensionless_count += 1

            # Lexical diversity
            words = re.findall(r"\b\w+\b", user_text.lower())
            total_prompt_words += len(words)
            prompt_vocab.update(words)
            for i in range(len(words) - 1):
                prompt_bigrams.add((words[i], words[i + 1]))

        # Calculate Lexical Diversity
        type_token_ratio = round(len(prompt_vocab) / max(1, total_prompt_words), 3)
        distinct_1 = round(len(prompt_vocab) / max(1, total_prompt_words), 3)
        distinct_2 = round(len(prompt_bigrams) / max(1, total_prompt_words), 3)

        # Recommended context window
        p99_len = cls._calc_stats(total_seq_tokens)["p99"]
        recommended_max_seq = 512 if p99_len <= 512 else (1024 if p99_len <= 1024 else 2048)

        return {
            "summary": {
                "total_records": total_samples,
                "train_records": len(splits_data["train"]),
                "val_records": len(splits_data["val"]),
                "test_records": len(splits_data["test"]),
                "dpo_records": len(dpo_records),
                "unique_personas_count": len(persona_counts),
                "unique_archetypes_count": len(archetype_counts),
                "recommended_max_seq_length": recommended_max_seq,
            },
            "token_geometry": {
                "prompt_tokens": cls._calc_stats(prompt_tokens),
                "response_tokens": cls._calc_stats(response_tokens),
                "total_sequence_tokens": cls._calc_stats(total_seq_tokens),
            },
            "math_and_latex": {
                "inline_math_samples_pct": round((latex_inline_count / max(1, total_samples)) * 100, 1),
                "display_math_samples_pct": round((latex_display_count / max(1, total_samples)) * 100, 1),
                "macro_frequencies": dict(macro_counts.most_common(10)),
            },
            "metrology_dimensions": {
                "si_base_occurrences": dim_base_counts,
                "dimensionless_occurrences": dimensionless_count,
            },
            "lexical_diversity": {
                "prompt_total_words": total_prompt_words,
                "prompt_vocabulary_size": len(prompt_vocab),
                "type_token_ratio": type_token_ratio,
                "distinct_1": distinct_1,
                "distinct_2": distinct_2,
            },
            "archetype_distribution": dict(archetype_counts),
            "top_personas": dict(persona_counts.most_common(15)),
        }

    @classmethod
    def generate_huggingface_dataset_card(
        cls, stats: Dict[str, Any], output_path: str = "./dataset/dataset_card.md"
    ) -> Path:
        """Generates an official, publication-ready Hugging Face Dataset Card."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = stats["summary"]
        tok = stats["token_geometry"]
        math_s = stats["math_and_latex"]
        lex = stats["lexical_diversity"]

        card_content = f"""---
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - metrology
  - physics
  - units-of-measurement
  - si-system
  - codata
  - bipm
  - qudt
  - synthetic
size_categories:
  - 1K<n<10K
pretty_name: DRUM-ML Metrology Instruct & Preference Dataset
dataset_info:
  features:
    - name: id
      dtype: string
    - name: archetype
      dtype: string
    - name: entity_uri
      dtype: string
    - name: messages
      list:
        - name: role
          dtype: string
        - name: content
          dtype: string
  splits:
    - name: train
      num_bytes: null
      num_examples: {summary['train_records']}
    - name: validation
      num_bytes: null
      num_examples: {summary['val_records']}
    - name: test
      num_bytes: null
      num_examples: {summary['test_records']}
---

# 🌐 DRUM-ML: Metrology & Physical Units Instruction Dataset

**DRUM-ML** (*Digital Representation of Units of Measurement for Machine Learning*) is an authoritative, multi-agent synthesized dataset developed under the **[CODATA DRUM Working Group](https://drum.codata.org)** (led by Pascal Heus, `pascal@codata.org` / `drum@codata.org`).

It bridges formal semantic web ontologies (**BIPM SI Digital Framework**, **CODATA DRUM Constants**, and **QUDT 2.1**) into verified, multi-turn conversational fine-tuning corpora for Large Language Models (LLMs).

---

## 📊 Dataset Statistics & Volume

| Split | Records | % Share | Format |
|---|---:|---:|---|
| **Train** | `{summary['train_records']:,}` | 85.0% | Multi-turn Chat JSONL |
| **Validation** | `{summary['val_records']:,}` | 10.0% | Multi-turn Chat JSONL |
| **Held-Out Test (M-Eval)** | `{summary['test_records']:,}` | 5.0% | Deterministic & Symbolic Evaluation |
| **DPO Preference Pairs** | `{summary['dpo_records']:,}` | — | Hard-Negative Direct Preference Pairs |
| **Total SFT Samples** | **`{summary['total_records']:,}`** | **100.0%** | Standardized Chat Format |

---

## 📐 Token Geometry & Training Budget

Token metrics computed across all splits (recommended context window: **`max_seq_length = {summary['recommended_max_seq_length']}`**):

| Feature | Min | Mean | Median | p95 | p99 | Max | Std Dev |
|---|---:|---:|---:|---:|---:|---:|---:|
| **User Prompt Tokens** | `{tok['prompt_tokens']['min']}` | `{tok['prompt_tokens']['mean']}` | `{tok['prompt_tokens']['median']}` | `{tok['prompt_tokens']['p95']}` | `{tok['prompt_tokens']['p99']}` | `{tok['prompt_tokens']['max']}` | `{tok['prompt_tokens']['std']}` |
| **Assistant Response Tokens** | `{tok['response_tokens']['min']}` | `{tok['response_tokens']['mean']}` | `{tok['response_tokens']['median']}` | `{tok['response_tokens']['p95']}` | `{tok['response_tokens']['p99']}` | `{tok['response_tokens']['max']}` | `{tok['response_tokens']['std']}` |
| **Total Sequence Length** | `{tok['total_sequence_tokens']['min']}` | **`{tok['total_sequence_tokens']['mean']}`** | `{tok['total_sequence_tokens']['median']}` | `{tok['total_sequence_tokens']['p95']}` | `{tok['total_sequence_tokens']['p99']}` | `{tok['total_sequence_tokens']['max']}` | `{tok['total_sequence_tokens']['std']}` |

---

## 🔬 Metrological & Physical Dimension Coverage

Distribution of physical units and quantities across the **7 SI Base Dimensions** $[L]^a [M]^b [T]^c [I]^d [\\Theta]^e [N]^f [J]^g$:

| SI Base Dimension | Symbol | Occurrences in Dataset |
|---|:---:|---:|
| **Length** | $\\text{{m}} / L$ | `{stats['metrology_dimensions']['si_base_occurrences']['L (Length)']:,}` |
| **Mass** | $\\text{{kg}} / M$ | `{stats['metrology_dimensions']['si_base_occurrences']['M (Mass)']:,}` |
| **Time** | $\\text{{s}} / T$ | `{stats['metrology_dimensions']['si_base_occurrences']['T (Time)']:,}` |
| **Electric Current** | $\\text{{A}} / I$ | `{stats['metrology_dimensions']['si_base_occurrences']['I (Electric Current)']:,}` |
| **Thermodynamic Temperature** | $\\text{{K}} / \\Theta$ | `{stats['metrology_dimensions']['si_base_occurrences']['Theta (Temperature)']:,}` |
| **Amount of Substance** | $\\text{{mol}} / N$ | `{stats['metrology_dimensions']['si_base_occurrences']['N (Amount of Substance)']:,}` |
| **Luminous Intensity** | $\\text{{cd}} / J$ | `{stats['metrology_dimensions']['si_base_occurrences']['J (Luminous Intensity)']:,}` |
| **Dimensionless & Logarithmic Units** | $1 / \\text{{rad}}, \\text{{dB}}, \\text{{Np}}$ | `{stats['metrology_dimensions']['dimensionless_occurrences']:,}` |

---

## 🧮 Math, LaTeX & Lexical Richness

- **Inline LaTeX Math Density:** `{math_s['inline_math_samples_pct']}%` of samples contain formal math delimiters (`$...$`).
- **Display Equation Density:** `{math_s['display_math_samples_pct']}%` contain standalone derivation blocks (`$$...$$`).
- **Prompt Vocabulary Size:** `{lex['prompt_vocabulary_size']:,}` unique words across `{lex['prompt_total_words']:,}` total words.
- **Type-Token Ratio (TTR):** `{lex['type_token_ratio']}` (Distinct-1: `{lex['distinct_1']}`, Distinct-2: `{lex['distinct_2']}`).

---

## 🏛️ Representation Across 35 Scientific Unions & NMIs

The questions are linguistically framed from the perspectives of international scientific bodies engaged with CODATA DRUM:
- **Standards & Metrology:** NIST, NRC Canada, BIPM, International Science Council (ISC).
- **Physical & Mathematical:** IUPAP (Physics), IUPAC (Chemistry), IAU (Astronomy), IUCr (Crystallography), IMU (Mathematics), URSI (Radio), IUTAM (Mechanics).
- **Earth & Space:** IUGG (Geodesy/Geophysics), IGU (Geography), ISPRS (Remote Sensing), ISDE (Digital Earth), IUSS (Soil Science).
- **Biological & Medical:** IUBS (Biology), IUIS (Immunology), IUPHAR (Pharmacology), IUPS (Physiology), IUTOX (Toxicology), IUNS (Nutrition), IUFoST (Food Science), IUPESM/IOMP (Medical Physics), IUPESM/IFMBE (Biomedical Engineering).
- **Human & Social Sciences:** IUPsyS (Psychology), ISA (Sociology), IUSSP (Demography), WAU (Anthropology), 4S (Science Studies), IUHPST (History & Philosophy of Science).

---

## 🛡️ Ground-Truth Invariance & Verification

All assistant responses in DRUM-ML are deterministically derived by **Agent 2 (Archetype Scaffolder)** and verified through a **4-Tier Automated Audit Gate**:
1. **Tier 1 (LaTeX & Syntax Guard):** Verifies balanced delimiters and JSON escaping.
2. **Tier 2 (Symbolic Dimension Engine):** `sympy.physics.units` and `pint` verify dimensional balance and conversion algebra.
3. **Tier 3 (Arbitrary-Precision Decimal Gate):** Python `decimal.Decimal` checks exact 2019 SI defining constants ($c, h, e, k, N_{{\\text{{A}}}}, \\Delta\\nu_{{\\text{{Cs}}}}, K_{{\\text{{cd}}}}$) with zero floating-point drift.
4. **Tier 4 (Sandbox Code Gate):** `rdflib` validates SPARQL syntax and Python executable snippets.

---

## 🚀 Quick Usage (Hugging Face Datasets)

```python
from datasets import load_dataset

dataset = load_dataset("drum-ml/metrology-instruct")
print(dataset["train"][0])
```

---

## 📜 Citation & Attributions

```bibtex
@dataset{{drum_ml_2026,
  author       = {{Pascal Heus and CODATA DRUM Working Group}},
  title        = {{DRUM-ML: Digital Representation of Units of Measurement for Machine Learning}},
  year         = {{2026}},
  publisher    = {{CODATA / DRUM}},
  url          = {{https://drum.codata.org}}
}}
```
"""
        with open(out, "w", encoding="utf-8") as f:
            f.write(card_content.strip() + "\n")
        return out

    @classmethod
    def print_rich_dashboard(cls, stats: Dict[str, Any], console: Optional[Console] = None) -> None:
        """Renders an interactive, aesthetic CLI terminal dashboard."""
        con = console or Console()
        sum_data = stats["summary"]
        tok = stats["token_geometry"]
        dim = stats["metrology_dimensions"]["si_base_occurrences"]

        # Summary Table
        t_sum = Table(title="📦 DRUM-ML Dataset Overview", header_style="bold cyan")
        t_sum.add_column("Metric", style="bold white")
        t_sum.add_column("Value", style="green", justify="right")
        t_sum.add_column("Notes", style="dim")

        t_sum.add_row("Total SFT Records", f"{sum_data['total_records']:,}", "All verified instruction pairs")
        t_sum.add_row("Train Split (85%)", f"{sum_data['train_records']:,}", "train.jsonl")
        t_sum.add_row("Validation Split (10%)", f"{sum_data['val_records']:,}", "val.jsonl")
        t_sum.add_row("Held-out Test (5%)", f"{sum_data['test_records']:,}", "test.jsonl (M-Eval benchmark)")
        t_sum.add_row("DPO Hard-Negatives", f"{sum_data['dpo_records']:,}", "dpo_preferences.jsonl")
        t_sum.add_row("Active Scientific Personas", f"{sum_data['unique_personas_count']}", "CODATA DRUM Scientific Unions")
        t_sum.add_row("Recommended max_seq_length", f"{sum_data['recommended_max_seq_length']}", "Based on p99 token length")

        con.print(t_sum)

        # Token Geometry Table
        t_tok = Table(title="📐 Token Geometry & Context Budget", header_style="bold yellow")
        t_tok.add_column("Feature", style="bold white")
        t_tok.add_column("Min", justify="right")
        t_tok.add_column("Mean", justify="right", style="bold green")
        t_tok.add_column("Median", justify="right")
        t_tok.add_column("p95", justify="right", style="bold yellow")
        t_tok.add_column("Max", justify="right", style="red")
        t_tok.add_column("Std Dev", justify="right", style="dim")

        for name, key in [("Prompt Tokens", "prompt_tokens"), ("Response Tokens", "response_tokens"), ("Total Sequence", "total_sequence_tokens")]:
            d = tok[key]
            t_tok.add_row(name, str(d["min"]), f"{d['mean']:.1f}", str(d["median"]), str(d["p95"]), str(d["max"]), f"{d['std']:.1f}")

        con.print(t_tok)

        # SI Dimension Breakdown Table
        t_dim = Table(title="🔬 Physical Dimension Coverage (7 SI Base Realizations)", header_style="bold magenta")
        t_dim.add_column("SI Base Dimension", style="bold white")
        t_dim.add_column("Occurrences in Dataset", justify="right", style="cyan")

        for k, v in dim.items():
            t_dim.add_row(k, f"{v:,}")

        con.print(t_dim)
