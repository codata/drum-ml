#!/usr/bin/env python3
"""DRUM-ML: Constant-Only Augmentation using Local Ollama / MLX / OpenAI.

Generates diverse persona-conditioned question variations for the 353 CODATA & BIPM
fundamental physical constants and merges them into the train/val/test splits.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from drum_ml.models.export import DPOPreferenceRecord
from drum_ml.models.scaffolds import AugmentedRecord, PersonaType, ScaffoldRecord
from drum_ml.pipeline.augmenter import MetrologyAugmenter
from drum_ml.pipeline.exporter import MetrologyExporter


def parse_json_mapping(text: str) -> dict[str, str]:
    """Robustly extracts persona-to-question JSON mappings from LLM text."""
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    elif "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    if "```json" in cleaned:
        m = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
    elif "```" in cleaned:
        m = re.search(r"```\s*(.*?)\s*```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()

    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return {str(k).lower().replace(" ", "_"): str(v).strip() for k, v in d.items()}
        except Exception:
            pass
    return {}


def augment_constant_batch(
    scaffold: ScaffoldRecord,
    personas: list[PersonaType],
    augmenter: MetrologyAugmenter,
) -> list[AugmentedRecord]:
    """Generates persona-conditioned questions for all personas in a single LLM request."""
    cache_key = hashlib.sha256(
        f"batch_{scaffold.id}_{len(personas)}_{augmenter.model}".encode()
    ).hexdigest()

    cached = augmenter._get_cache(cache_key)
    if cached:
        augmenter.stats["cache_hits"] += 1
        try:
            mapping = json.loads(cached)
        except Exception:
            mapping = {}
    else:
        persona_bullets = "\n".join(
            [f"- {p.value}: {p.value.replace('_', ' ').title()}" for p in personas]
        )
        system_prompt = (
            "You are a metrology dataset generator for the CODATA DRUM initiative. "
            "Generate precise domain questions about physical constants from multiple scientific perspectives."
        )
        user_prompt = (
            f'Physical Constant: "{scaffold.canonical_query}"\n\n'
            f"Generate 1 realistic question for EACH of these scientific personas:\n{persona_bullets}\n\n"
            f'Output ONLY a valid JSON object mapping persona keys to questions:\n{{"persona_key": "question", ...}}'
        )
        llm_out = augmenter._call_llm_api(system_prompt, user_prompt)
        if llm_out:
            augmenter.stats["live_llm"] += 1
            mapping = parse_json_mapping(llm_out)
        else:
            augmenter.stats["offline_fallbacks"] += 1
            mapping = {}
        augmenter._set_cache(cache_key, scaffold.canonical_query, json.dumps(mapping))

    records = []
    for idx, p in enumerate(personas):
        q = mapping.get(p.value.lower()) or mapping.get(p.value.replace("_", " "))
        if not q or len(q.strip()) < 10 or "We need to" in q or "JSON" in q:
            q = (
                scaffold.canonical_query
                if p == PersonaType.GENERAL_USER
                else f"[{p.value.replace('_', ' ').title()}] {scaffold.canonical_query}"
            )
        records.append(
            AugmentedRecord(
                id=f"{scaffold.id}_{p.value}_{idx}",
                scaffold_id=scaffold.id,
                archetype=scaffold.archetype,
                persona=p,
                user_query=str(q).strip(),
                ground_truth_answer=scaffold.ground_truth_answer,
                entity_uri=scaffold.entity_uri,
                quantity_kind_uri=scaffold.quantity_kind_uri,
                temperature=augmenter.temperature,
                llm_generator=augmenter.model,
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Augment fundamental physical constants using local Ollama, MLX, or OpenAI-compatible server."
    )
    parser.add_argument(
        "--model",
        default="gemma4:e2b-mlx",
        help="Model identifier / Ollama tag (e.g., gemma4:e2b-mlx, nemotron-3-nano:4b, gemma2:2b)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai", "mlx", "google"],
        help="Inference provider (ollama, openai, mlx, google)",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:11434",
        help="API base URL (default: http://localhost:11434 for Ollama, http://localhost:8080/v1 for MLX server)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent generation workers (default: 8)",
    )
    parser.add_argument(
        "--personas",
        default="all",
        help="Personas to generate: 'all', 'core' (10 representative unions), or comma-separated list",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of constants to augment (e.g. 10 for quick testing)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=True,
        help="Batch multiple personas per LLM call (reduces HTTP calls from 13,767 to 353, ~30x faster)",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=2,
        help="Number of question variations per persona in unbatched mode (default: 2)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--scaffolds-path",
        default="./data/scaffolds.json",
        help="Path to canonical scaffolds JSON (default: ./data/scaffolds.json)",
    )
    parser.add_argument(
        "--output-augmented",
        default="./data/augmented_constants.json",
        help="Output path for augmented constant records",
    )
    parser.add_argument(
        "--merge-splits",
        action="store_true",
        default=True,
        help="Merge generated constant records into dataset/train.jsonl, val.jsonl, and test.jsonl",
    )
    args = parser.parse_args()

    provider_name = "openai" if args.provider == "mlx" else args.provider
    console = Console()

    print("=" * 70)
    print(" DRUM-ML: CONSTANT-ONLY AUGMENTATION")
    print(f" Provider:  {args.provider.upper()}")
    print(f" Model:     {args.model}")
    print(f" Endpoint:  {args.api_base}")
    print(f" Workers:   {args.concurrency}")
    print("=" * 70)

    # 1. Load only constant scaffolds
    scaffolds_file = Path(args.scaffolds_path)
    if not scaffolds_file.exists():
        raise FileNotFoundError(f"Scaffolds file not found at {scaffolds_file}")

    with open(scaffolds_file, encoding="utf-8") as f:
        all_scaffolds = [ScaffoldRecord(**s) for s in json.load(f)]

    const_scaffolds = [s for s in all_scaffolds if "_const_arch1" in s.id]
    print(f"✓ Filtered {len(const_scaffolds)} fundamental physical constant scaffolds.")

    if args.limit:
        const_scaffolds = const_scaffolds[: args.limit]
        print(f"✓ Limited to first {len(const_scaffolds)} constants (--limit {args.limit})")

    if not const_scaffolds:
        print("No constant scaffolds found in data/scaffolds.json!")
        return

    # 2. Check Connection & initialize Augmenter
    if provider_name == "ollama":
        print(f"\nChecking connection to Ollama at {args.api_base}...")
        import urllib.request

        try:
            req = urllib.request.Request(f"{args.api_base.rstrip('/')}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                available_models = [m.get("name") for m in data.get("models", [])]
                print(
                    f"✓ Connected to Ollama! Available models: {', '.join(available_models[:5])}..."
                )
        except Exception as e:
            print(f"⚠️ Warning: Could not reach Ollama at {args.api_base} ({e}).")

    augmenter = MetrologyAugmenter(
        provider=provider_name,
        model=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        concurrency_limit=args.concurrency,
        cache_db_path="./data/cache/llm_cache.sqlite",
    )

    # Parse personas
    if args.personas == "all":
        personas = list(PersonaType)
    elif args.personas == "core":
        core_names = [
            "general_user",
            "physics_student",
            "academic_metrologist",
            "nist_metrologist",
            "iupap_physicist",
            "iupac_chemist",
            "iau_astronomer",
            "firmware_iot_engineer",
            "data_scientist_analyst",
            "iso_compliance_auditor",
        ]
        personas = [PersonaType(p) for p in core_names if p in PersonaType._value2member_map_]
    else:
        personas = [
            PersonaType(p.strip())
            for p in args.personas.split(",")
            if p.strip() in PersonaType._value2member_map_
        ]

    total_tasks = len(const_scaffolds) if args.batch else (len(const_scaffolds) * len(personas))
    mode_desc = (
        f"Batched by Constant ({len(personas)} personas per call)"
        if args.batch
        else "Single Persona per Call"
    )
    console.print(
        f"\n[bold green]Augmenting {len(const_scaffolds)} constants across {len(personas)} personas "
        f"({total_tasks:,} total LLM requests, Mode: {mode_desc})...[/bold green]"
    )

    try:
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=35),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Generating variations...", total=total_tasks)

            def on_step(n=1):
                progress.update(task_id, advance=n)

            t0 = time.perf_counter()
            augmented_records: list[AugmentedRecord] = []

            if args.batch:
                max_workers = min(args.concurrency, len(const_scaffolds))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(augment_constant_batch, scaffold, personas, augmenter)
                        for scaffold in const_scaffolds
                    ]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            augmented_records.extend(f.result())
                        except Exception:
                            pass
                        on_step(1)
            else:
                augmented_records = augmenter.augment_all(
                    scaffolds=const_scaffolds,
                    personas=personas,
                    variations_per_archetype=args.variations,
                    progress_callback=on_step,
                )

            duration = time.perf_counter() - t0

        console.print(
            f"\n[bold green]✓ Generated {len(augmented_records):,} constant samples in {duration:.1f}s "
            f"({len(augmented_records) / max(0.1, duration):.1f} samples/sec)[/bold green]"
        )
        console.print(f"  └─ Live LLM calls: [cyan]{augmenter.stats['live_llm']:,}[/cyan]")
        console.print(f"  └─ SQLite Cache hits: [yellow]{augmenter.stats['cache_hits']:,}[/yellow]")
        console.print(
            f"  └─ Offline fallbacks: [red]{augmenter.stats['offline_fallbacks']:,}[/red]"
        )

        # 3. Save augmented constants
        out_path = Path(args.output_augmented)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in augmented_records], f, indent=2)
        print(f"✓ Saved augmented records to {out_path}")

        # 4. Merge with existing dataset splits if requested
        if args.merge_splits and len(augmented_records) > 0:
            print("\nMerging newly augmented constants into master dataset splits...")
            augmented_main_path = Path("./data/augmented.json")
            if augmented_main_path.exists():
                with open(augmented_main_path, encoding="utf-8") as f:
                    existing_augmented = [AugmentedRecord(**r) for r in json.load(f)]

                # Filter out previous constant records and append new ones
                units_augmented = [
                    r for r in existing_augmented if "_const_arch1" not in r.scaffold_id
                ]
                merged_augmented = units_augmented + augmented_records

                with open(augmented_main_path, "w", encoding="utf-8") as f:
                    json.dump([r.model_dump() for r in merged_augmented], f, indent=2)
                print(
                    f"✓ Updated data/augmented.json (Units: {len(units_augmented)}, Constants: {len(augmented_records)})"
                )

                # Re-export dataset splits
                dpo_pairs = []
                dpo_file = Path("./dataset/dpo_preferences.jsonl")
                if dpo_file.exists():
                    with open(dpo_file, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                dpo_pairs.append(DPOPreferenceRecord.model_validate_json(line))

                exporter = MetrologyExporter(output_dir="./dataset")
                paths = exporter.export_all(merged_augmented, dpo_pairs)
                print("✓ Master dataset splits re-exported successfully:")
                for name, path in paths.items():
                    print(f"  └─ {name}: {path}")

        print("\n✓ Constant-only augmentation complete!")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠️ Augmentation interrupted by user (<Ctrl+C>).[/bold yellow]")
        console.print(
            "[dim]Completed LLM responses were cached in SQLite, but master dataset splits were left untouched.[/dim]\n"
        )


if __name__ == "__main__":
    main()
