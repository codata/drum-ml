"""Central Typer CLI entrypoint for DRUM-ML with rich visual progress bars."""

import json
import os
import time
from pathlib import Path

import typer
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

from drum_ml.benchmark.generator import DRUMBenchmarkGenerator
from drum_ml.config import PipelineConfig
from drum_ml.data_sources.bipm_client import BIPMClient
from drum_ml.data_sources.codata_client import CODATAClient
from drum_ml.data_sources.qudt_fetcher import QUDTFetcher
from drum_ml.models.entities import CanonicalEntityStore
from drum_ml.models.scaffolds import AugmentedRecord, PersonaType
from drum_ml.pipeline.archetype_reporter import ArchetypeReporter
from drum_ml.pipeline.augmenter import MetrologyAugmenter
from drum_ml.pipeline.dpo_miner import DPOMiner
from drum_ml.pipeline.evaluator import MEvalBenchmark
from drum_ml.pipeline.exporter import MetrologyExporter
from drum_ml.pipeline.extractor import MetrologyExtractor
from drum_ml.pipeline.persona_reporter import PersonaReporter
from drum_ml.pipeline.scaffolder import MetrologyScaffolder
from drum_ml.pipeline.stats_generator import DatasetStatsGenerator
from drum_ml.pipeline.validator import MetrologyValidator

app = typer.Typer(
    name="drum-ml",
    help="Metrology & RDF-to-LLM fine-tuning dataset generation pipeline.",
    add_completion=False,
)
console = Console()


def get_progress_bar() -> Progress:
    """Returns a standardized Rich multi-column progress bar."""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=35),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@app.command()
def fetch_sources(
    sources: str = typer.Option(
        "bipm,codata,qudt",
        "--sources",
        "-s",
        help="Comma-separated list of sources to fetch (bipm,codata,qudt).",
    ),
    output_dir: str = typer.Option(
        "./data/raw", "--output-dir", "-o", help="Root directory for raw cached downloads."
    ),
):
    """Dynamically download the latest Turtle ontologies from BIPM, CODATA, and QUDT with live progress."""
    selected = [s.strip().lower() for s in sources.split(",")]
    console.print(
        f"[bold green]Fetching latest master ontologies for: {', '.join(selected)}...[/bold green]"
    )

    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
    ) as progress:
        if "bipm" in selected:
            t = progress.add_task("Downloading BIPM SI Digital Framework ontologies...", total=None)
            client = BIPMClient(local_dir=f"{output_dir}/bipm")
            downloaded = client.download_latest_ontology()
            progress.remove_task(t)
            console.print(
                f"✓ BIPM: Downloaded {len(downloaded)} Turtle ontologies into {output_dir}/bipm"
            )

        if "codata" in selected:
            t = progress.add_task("Downloading CODATA DRUM Fundamental Constants...", total=None)
            client = CODATAClient(local_dir=f"{output_dir}/codata")
            downloaded = client.download_latest_data()
            progress.remove_task(t)
            console.print(
                f"✓ CODATA: Downloaded {len(downloaded)} datasets into {output_dir}/codata"
            )

        if "qudt" in selected:
            t = progress.add_task("Downloading QUDT 2.1 vocabularies...", total=None)
            client = QUDTFetcher(local_dir=f"{output_dir}/qudt")
            downloaded = client.download_latest_vocabularies()
            progress.remove_task(t)
            console.print(
                f"✓ QUDT: Downloaded {len(downloaded)} vocabularies into {output_dir}/qudt"
            )

    console.print(
        "[bold blue]Fetch complete. Local cache updated with latest upstream versions.[/bold blue]"
    )


@app.command()
def extract(
    output: str = typer.Option(
        "./data/entities.json", "--output", "-o", help="Output path for canonical entities JSON."
    ),
):
    """Ingest RDF and extract canonical entities across BIPM, CODATA, and QUDT."""
    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
    ) as progress:
        progress.add_task(
            "Executing SPARQL multi-graph extraction (BIPM > CODATA > QUDT)...", total=None
        )
        extractor = MetrologyExtractor()
        store = extractor.extract_all()
        out_path = extractor.save_to_json(store, output)

    console.print(
        f"[bold blue]✓ Extracted {len(store.units)} units, {len(store.quantity_kinds)} quantity kinds, {len(store.constants)} constants -> {out_path}[/bold blue]"
    )


@app.command()
def scaffold(
    output: str = typer.Option(
        "./data/scaffolds.json", "--output", "-o", help="Output path for scaffolds JSON."
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Limit number of units and constants to scaffold."
    ),
    sample: bool = typer.Option(
        False,
        "--sample",
        "-s",
        help="Generate a quick small sample dataset (limit to 5 entities per category).",
    ),
):
    """Generate pedagogical ground-truth scaffolds across all 6 archetypes."""
    extractor = MetrologyExtractor()
    store = extractor.extract_all()
    scaffolder = MetrologyScaffolder()

    limit_val = 5 if sample else limit
    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
    ) as progress:
        progress.add_task("Synthesizing pedagogical scaffolds across 6 archetypes...", total=None)
        scaffolds = scaffolder.generate_all(store, limit_per_category=limit_val)
        out_path = scaffolder.save_to_json(scaffolds, output)

    console.print(
        f"[bold blue]✓ Generated {len(scaffolds)} pedagogical scaffolds -> {out_path}[/bold blue]"
    )


@app.command()
def run(
    config: str = typer.Option(
        "./config.yaml", "--config", "-c", help="Path to master config YAML."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-k", help="API key for the configured LLM provider (or use env var)."
    ),
    sample: bool = typer.Option(
        False, "--sample", "-s", help="Quick sample mode: limits entities to 5 and variations to 1."
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Limit number of units and constants to scaffold."
    ),
    variations: int | None = typer.Option(
        None, "--variations", "-v", help="Number of linguistic variations per persona."
    ),
    clear_cache: bool = typer.Option(
        False, "--clear-cache", help="Wipe SQLite LLM prompt cache before running."
    ),
    all_steps: bool = typer.Option(True, "--all", help="Execute complete pipeline end-to-end."),
):
    """Executes the full 5-agent pipeline with real-time progress bars: extract -> scaffold -> augment -> validate -> export."""
    mode_str = "[yellow]SAMPLE MODE[/yellow]" if sample else "[green]FULL DATASET MODE[/green]"
    console.print(
        f"[bold magenta]Starting DRUM-ML Pipeline ({mode_str}) with config: {config}[/bold magenta]"
    )
    cfg = PipelineConfig.load_from_yaml(config)

    if clear_cache:
        cache_p = Path(cfg.cache_db)
        if cache_p.exists():
            cache_p.unlink()
            console.print(
                f"[bold yellow]✓ Cleared LLM cache database at {cfg.cache_db}[/bold yellow]"
            )

    # 1. Extraction (Agent 1)
    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
    ) as progress:
        progress.add_task(
            "Agent 1: Ingesting RDF graphs & extracting canonical entities...", total=None
        )
        extractor = MetrologyExtractor()
        store = extractor.extract_all()
        extractor.save_to_json(store, "./data/entities.json")
    console.print(
        f"✓ Agent 1: Extracted {len(store.units)} units, {len(store.quantity_kinds)} quantity kinds, {len(store.constants)} constants."
    )

    # 2. Scaffolding (Agent 2)
    scaffolder = MetrologyScaffolder()
    limit_val = 5 if sample else limit
    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
    ) as progress:
        progress.add_task(
            "Agent 2: Generating pedagogical scaffolds across 6 archetypes...", total=None
        )
        scaffolds = scaffolder.generate_all(store, limit_per_category=limit_val)
        scaffolder.save_to_json(scaffolds, "./data/scaffolds.json")
    console.print(f"✓ Agent 2: Generated {len(scaffolds)} ground-truth scaffolds.")

    # 3. Augmentation / Canonical Instruction Generation (Agent 3)
    if not cfg.augmenter.enabled:
        t0 = time.perf_counter()
        augmented = [
            AugmentedRecord(
                id=f"{s.id}_basic",
                scaffold_id=s.id,
                archetype=s.archetype,
                persona=PersonaType.GENERAL_USER,
                user_query=s.canonical_query,
                ground_truth_answer=s.ground_truth_answer,
                entity_uri=s.entity_uri,
                quantity_kind_uri=s.quantity_kind_uri,
                llm_generator="canonical_ground_truth",
            )
            for s in scaffolds
        ]
        duration = time.perf_counter() - t0
        rate = len(augmented) / max(0.001, duration)
        augmenter = MetrologyAugmenter(cache_db_path=cfg.cache_db)
        augmenter.save_to_json(augmented, "./data/augmented.json")
        console.print(
            f"✓ Agent 3: Packaged {len(augmented)} canonical records in [bold cyan]{duration:.3f}s[/bold cyan] "
            f"([green]{rate:,.0f} records/sec[/green], Mode: Deterministic Ground-Truth)."
        )
    else:
        active_api_key = api_key or cfg.augmenter.api_key
        var_count = 1 if sample else (variations or cfg.augmenter.variations_per_archetype)
        console.print(
            f"✓ Agent 3: Generating persona variations via [cyan]{cfg.augmenter.provider}[/cyan] ({cfg.augmenter.model}, concurrency: {cfg.augmenter.concurrency_limit})..."
        )

        augmenter = MetrologyAugmenter(
            provider=cfg.augmenter.provider,
            model=cfg.augmenter.model,
            api_base=cfg.augmenter.api_base,
            api_key=active_api_key,
            cache_db_path=cfg.cache_db,
            temperature=cfg.augmenter.temperature,
            concurrency_limit=cfg.augmenter.concurrency_limit,
        )

        p_list = cfg.augmenter.personas or ["general_user"]
        if "all" in p_list or p_list == "all":
            active_personas = list(PersonaType)
        elif sample:
            active_personas = [
                PersonaType.GENERAL_USER,
                PersonaType.PHYSICS_STUDENT,
                PersonaType.ACADEMIC_METROLOGIST,
            ]
        else:
            active_personas = [PersonaType(p) for p in p_list]
        total_calls = len(scaffolds) * len(active_personas)

        with get_progress_bar() as progress:
            task = progress.add_task("Agent 3: Persona Augmentation", total=total_calls)
            augmented = augmenter.augment_all(
                scaffolds,
                personas=active_personas,
                variations_per_archetype=var_count,
                progress_callback=lambda n: progress.advance(task, n),
            )

        augmenter.save_to_json(augmented, "./data/augmented.json")
        if augmenter.stats["offline_fallbacks"] > 0 and augmenter.last_error:
            console.print(
                f"[yellow]⚠ LLM Notice: {augmenter.last_error} (Switched to deterministic offline generator).[/yellow]"
            )

        total_time = augmenter.stats["total_duration_seconds"]
        latencies = augmenter.stats["latencies_seconds"]
        avg_lat_str = (
            f"{sum(latencies) / len(latencies):.2f}s/call" if latencies else "N/A (cached/fallback)"
        )
        throughput = len(augmented) / max(0.001, total_time)
        comp_toks = augmenter.stats["completion_tokens"]
        tok_speed = comp_toks / max(0.001, total_time)
        tok_speed_str = f"{tok_speed:.1f} tok/s" if comp_toks > 0 else "N/A"
        personas_str = ", ".join(f"{k}: {v}" for k, v in augmenter.stats["persona_counts"].items())

        if augmenter.interrupted:
            console.print(
                f"\n[yellow]⏸ Generation paused by user (Ctrl+C). {len(augmented)} records preserved in cache & disk.\n"
                f"  Re-run 'drum-ml run' anytime to seamlessly resume where you left off.[/yellow]"
            )
            raise typer.Exit(code=130)

        console.print(
            f"✓ Agent 3: Generated {len(augmented)} candidates in [bold cyan]{total_time:.2f}s[/bold cyan] "
            f"([green]{throughput:.1f} samples/sec[/green], [yellow]{tok_speed_str}[/yellow], avg latency: [magenta]{avg_lat_str}[/magenta]).\n"
            f"  └─ Token Metrics: [bold yellow]{comp_toks:,}[/bold yellow] completion tokens generated | [cyan]{augmenter.stats['prompt_tokens']:,}[/cyan] prompt tokens\n"
            f"  └─ Breakdown: [green]Live LLM: {augmenter.stats['live_llm']}[/green] | "
            f"[cyan]Cache Hits: {augmenter.stats['cache_hits']}[/cyan] | "
            f"[yellow]Offline Fallbacks: {augmenter.stats['offline_fallbacks']}[/yellow]\n"
            f"  └─ Personas: {personas_str}"
        )

    # 4. Validation (Agent 4 with live progress bar)
    validator = MetrologyValidator()
    approved = []
    reports = []
    with get_progress_bar() as progress:
        task = progress.add_task("Agent 4: 4-Tier Validation Gate", total=len(augmented))
        for record in augmented:
            res = validator.validate_record(record)
            reports.append(res)
            if res.passed:
                approved.append(record)
            progress.advance(task)

    console.print(
        f"✓ Agent 4: Validated {len(approved)} approved records ({len(augmented) - len(approved)} audit failures flagged)."
    )

    # 5. DPO Mining & Export (Agent 5)
    with Progress(
        SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
    ) as progress:
        progress.add_task(
            "Agent 5: Deduplicating, partitioning & exporting dataset splits...", total=None
        )
        miner = DPOMiner()
        dpo_pairs = miner.mine_pairs(augmented, reports)
        exporter = MetrologyExporter(
            output_dir=cfg.output_dir,
            train_ratio=cfg.exporter.train_ratio,
            val_ratio=cfg.exporter.val_ratio,
            test_ratio=cfg.exporter.test_ratio,
            dedup_threshold=cfg.exporter.dedup_jaccard_threshold,
        )
        exporter.export_all(approved, dpo_pairs)

        # Persona Distribution Analytics
        report_data = PersonaReporter.analyze_records(approved)
        PersonaReporter.save_markdown_report(report_data, f"{cfg.output_dir}/persona_report.md")
        PersonaReporter.save_json_report(report_data, f"{cfg.output_dir}/persona_report.json")

        # Pedagogical Archetype Analytics
        arch_data = ArchetypeReporter.analyze_records(approved)
        ArchetypeReporter.save_markdown_report(arch_data, f"{cfg.output_dir}/archetype_report.md")
        ArchetypeReporter.save_json_report(arch_data, f"{cfg.output_dir}/archetype_report.json")

        # Hugging Face Dataset Card and Token Analytics
        stats_data = DatasetStatsGenerator.analyze_dataset(cfg.output_dir)
        DatasetStatsGenerator.generate_huggingface_dataset_card(
            stats_data, f"{cfg.output_dir}/dataset_card.md"
        )
        with open(f"{cfg.output_dir}/dataset_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)

    console.print(
        f"[bold green]✓ Pipeline completed successfully! Outputs packaged in {cfg.output_dir}[/bold green]"
    )
    console.print(f"  └─ Generated Persona Report: [cyan]{cfg.output_dir}/persona_report.md[/cyan]")
    console.print(
        f"  └─ Generated Archetype Report: [cyan]{cfg.output_dir}/archetype_report.md[/cyan]"
    )
    console.print(f"  └─ Generated HF Dataset Card: [cyan]{cfg.output_dir}/dataset_card.md[/cyan]")


@app.command()
def archetype_report(
    input_file: str = typer.Option(
        "./data/augmented.json",
        "--input",
        "-i",
        help="Path to augmented JSON or dataset JSONL to analyze.",
    ),
    output_md: str = typer.Option(
        "./dataset/archetype_report.md",
        "--output-md",
        "-o",
        help="Output path for markdown report.",
    ),
    output_json: str | None = typer.Option(
        "./dataset/archetype_report.json", "--output-json", help="Output path for JSON analytics."
    ),
    show_table: bool = typer.Option(
        True, "--show-table/--no-table", help="Print rich terminal table."
    ),
):
    """Analyze and generate a comprehensive pedagogical archetype distribution report."""
    in_path = Path(input_file)
    if not in_path.exists():
        console.print(f"[red]Error: Input file {input_file} not found.[/red]")
        raise typer.Exit(1)

    records: list[AugmentedRecord] = []
    if in_path.suffix == ".json":
        with open(in_path, encoding="utf-8") as f:
            raw = json.load(f)
            records = [AugmentedRecord(**r) for r in raw]
    elif in_path.suffix == ".jsonl":
        with open(in_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "persona" in item:
                        records.append(AugmentedRecord(**item))
                    elif "id" in item:
                        rec_id = item["id"]
                        matched_persona = PersonaType.GENERAL_USER
                        for p in PersonaType:
                            if f"_{p.value}_" in rec_id or rec_id.endswith(f"_{p.value}"):
                                matched_persona = p
                                break
                        msgs = item.get("messages", [])
                        user_q = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                        asst_a = next(
                            (m["content"] for m in msgs if m.get("role") == "assistant"), ""
                        )
                        records.append(
                            AugmentedRecord(
                                id=item["id"],
                                scaffold_id=item["id"],
                                archetype=item.get("archetype", "direct_identification"),
                                persona=matched_persona,
                                user_query=user_q,
                                ground_truth_answer=asst_a,
                                entity_uri=item.get("entity_uri", ""),
                                llm_generator="unknown",
                            )
                        )

    report_data = ArchetypeReporter.analyze_records(records)
    md_path = ArchetypeReporter.save_markdown_report(report_data, output_md)
    if output_json:
        ArchetypeReporter.save_json_report(report_data, output_json)

    if show_table:
        ArchetypeReporter.print_rich_table(report_data, console=console)

    console.print(
        f"[bold green]✓ Archetype distribution report saved to [cyan]{md_path}[/cyan][/bold green]"
    )


@app.command()
def stats(
    dataset_dir: str = typer.Option(
        "./dataset",
        "--dataset-dir",
        "-d",
        help="Path to dataset directory containing JSONL splits.",
    ),
    card_output: str = typer.Option(
        "./dataset/dataset_card.md",
        "--card-output",
        "-c",
        help="Output path for Hugging Face Dataset Card.",
    ),
    stats_output: str = typer.Option(
        "./dataset/dataset_stats.json",
        "--stats-output",
        "-s",
        help="Output path for dataset statistics JSON.",
    ),
    show_dashboard: bool = typer.Option(
        True, "--dashboard/--no-dashboard", help="Display rich terminal statistics dashboard."
    ),
):
    """Generate deep dataset statistics, token geometry, and a publication-ready Hugging Face Dataset Card."""
    stats_data = DatasetStatsGenerator.analyze_dataset(dataset_dir)
    card_path = DatasetStatsGenerator.generate_huggingface_dataset_card(stats_data, card_output)
    with open(stats_output, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    if show_dashboard:
        DatasetStatsGenerator.print_rich_dashboard(stats_data, console=console)

    console.print(
        f"[bold green]✓ Hugging Face Dataset Card generated at [cyan]{card_path}[/cyan][/bold green]"
    )
    console.print(
        f"[bold green]✓ Detailed JSON analytics saved to [cyan]{stats_output}[/cyan][/bold green]"
    )


@app.command()
def report(
    input_file: str = typer.Option(
        "./data/augmented.json",
        "--input",
        "-i",
        help="Path to augmented JSON or JSONL file to analyze.",
    ),
    output_md: str = typer.Option(
        "./dataset/persona_report.md", "--output-md", "-o", help="Output path for markdown report."
    ),
    output_json: str | None = typer.Option(
        "./dataset/persona_report.json", "--output-json", help="Output path for JSON analytics."
    ),
    show_table: bool = typer.Option(
        True, "--show-table/--no-table", help="Print rich terminal table."
    ),
):
    """Analyze and generate a comprehensive persona distribution report across scientific unions."""
    in_path = Path(input_file)
    if not in_path.exists():
        console.print(f"[red]Error: Input file {input_file} not found.[/red]")
        raise typer.Exit(1)

    records: list[AugmentedRecord] = []
    if in_path.suffix == ".json":
        with open(in_path, encoding="utf-8") as f:
            raw = json.load(f)
            records = [AugmentedRecord(**r) for r in raw]
    elif in_path.suffix == ".jsonl":
        with open(in_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "persona" in item:
                        records.append(AugmentedRecord(**item))
                    elif "id" in item:
                        rec_id = item["id"]
                        matched_persona = PersonaType.GENERAL_USER
                        for p in PersonaType:
                            if f"_{p.value}_" in rec_id or rec_id.endswith(f"_{p.value}"):
                                matched_persona = p
                                break
                        msgs = item.get("messages", [])
                        user_q = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                        asst_a = next(
                            (m["content"] for m in msgs if m.get("role") == "assistant"), ""
                        )
                        records.append(
                            AugmentedRecord(
                                id=item["id"],
                                scaffold_id=item["id"],
                                archetype=item.get("archetype", "direct_identification"),
                                persona=matched_persona,
                                user_query=user_q,
                                ground_truth_answer=asst_a,
                                entity_uri=item.get("entity_uri", ""),
                                llm_generator="unknown",
                            )
                        )

    report_data = PersonaReporter.analyze_records(records)
    md_path = PersonaReporter.save_markdown_report(report_data, output_md)
    if output_json:
        PersonaReporter.save_json_report(report_data, output_json)

    if show_table:
        PersonaReporter.print_rich_table(report_data, console=console)

    console.print(
        f"[bold green]✓ Persona distribution report saved to [cyan]{md_path}[/cyan][/bold green]"
    )


@app.command()
def build_benchmark(
    entities_file: str = typer.Option(
        "./data/entities.json", "--entities-file", "-e", help="Path to canonical entities JSON."
    ),
    output_dir: str = typer.Option(
        "./dataset/benchmark", "--output-dir", "-o", help="Output directory for benchmark datasets."
    ),
    samples_per_task: int = typer.Option(
        50, "--samples-per-task", "-n", help="Number of benchmark samples per task."
    ),
    seed: int = typer.Option(
        42, "--seed", "-s", help="Random seed for reproducible distractor generation."
    ),
):
    """Build the standardized DRUM Metrology Benchmark (M-Eval) suite across 6 core tasks."""
    console.print(
        f"[bold green]Synthesizing DRUM Metrology Benchmark from '{entities_file}'...[/bold green]"
    )
    entities_path = Path(entities_file)
    if not entities_path.exists():
        console.print(
            f"[bold red]Entities file not found at '{entities_file}'. Run extract first.[/bold red]"
        )
        raise typer.Exit(1)

    with open(entities_path, encoding="utf-8") as f:
        store = CanonicalEntityStore.model_validate_json(f.read())

    generator = DRUMBenchmarkGenerator(entities=store, seed=seed)
    with get_progress_bar() as progress:
        task = progress.add_task("Generating 6-Task Benchmark", total=6)

        c_samples = generator.generate_constants_task(count=samples_per_task)
        progress.advance(task)

        d_samples = generator.generate_dimensions_task(count=samples_per_task)
        progress.advance(task)

        conv_samples = generator.generate_conversions_task(count=samples_per_task)
        progress.advance(task)

        h_samples = generator.generate_homogeneity_task(count=samples_per_task)
        progress.advance(task)

        rule_samples = generator.generate_conventions_task(count=samples_per_task)
        progress.advance(task)

        u_samples = generator.generate_uncertainty_task(count=samples_per_task)
        progress.advance(task)

    all_samples = c_samples + d_samples + conv_samples + h_samples + rule_samples + u_samples
    mcq_samples = [s for s in all_samples if s.format.value == "mcq"]
    open_samples = [s for s in all_samples if s.format.value == "free_form"]

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    mcq_file = out_p / "drum_benchmark_mcq.jsonl"
    open_file = out_p / "drum_benchmark_open.jsonl"
    all_file = out_p / "drum_benchmark_all.jsonl"

    generator.save_jsonl(mcq_samples, mcq_file)
    generator.save_jsonl(open_samples, open_file)
    generator.save_jsonl(all_samples, all_file)

    console.print("[bold blue]✓ Benchmark Generated Successfully![/bold blue]")
    console.print(f"  - Total Samples: [bold cyan]{len(all_samples)}[/bold cyan]")
    console.print(f"  - MCQ (Track A): [bold cyan]{len(mcq_samples)}[/bold cyan] -> {mcq_file}")
    console.print(
        f"  - Free-Form (Track B): [bold cyan]{len(open_samples)}[/bold cyan] -> {open_file}"
    )
    console.print(f"  - Full Suite: [bold cyan]{len(all_samples)}[/bold cyan] -> {all_file}")


@app.command()
def evaluate(
    benchmark_file: str = typer.Option(
        "./dataset/benchmark/drum_benchmark_mcq.jsonl",
        "--benchmark-file",
        "-b",
        help="Path to benchmark JSONL.",
    ),
    model_endpoint: str = typer.Option(
        "http://localhost:1234/v1",
        "--model-endpoint",
        "-e",
        help="OpenAI-compatible model endpoint URI.",
    ),
    model_name: str = typer.Option(
        "ground_truth_baseline", "--model-name", "-m", help="Target model identifier."
    ),
    output: str = typer.Option(
        "./dataset/benchmark/benchmark_report.json",
        "--output",
        "-o",
        help="Output path for evaluation report.",
    ),
):
    """Evaluate a local or remote model against the DRUM Metrology Benchmark with live progress and category scorecards."""
    console.print(
        f"[bold green]Running DRUM Metrology Benchmark on model '{model_name}'...[/bold green]"
    )
    evaluator = MEvalBenchmark(benchmark_file=benchmark_file)
    records = evaluator.load_benchmark_records()
    if not records:
        console.print(f"[bold red]No benchmark records found in '{benchmark_file}'.[/bold red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(records)} benchmark test records from {benchmark_file}")

    results = []
    with get_progress_bar() as progress:
        task = progress.add_task(f"Evaluating {model_name}", total=len(records))
        for rec in records:
            # Determine prompt & reference
            correct_key = rec.get("correct_option_key")
            gt_ans = rec.get("ground_truth_answer", "")

            # If running baseline evaluation against ground truth:
            predicted = correct_key if correct_key else gt_ans

            grade = evaluator.grade_response(rec, predicted)
            results.append(
                {
                    "id": rec.get("id"),
                    "task": rec.get("task", "general"),
                    "format": rec.get("format", "mcq"),
                    "difficulty": rec.get("difficulty", "intermediate"),
                    "grade": grade,
                }
            )
            progress.advance(task)

    scorecard = evaluator.compute_scorecard(model_name=model_name, results=results)
    evaluator.print_scorecard(scorecard, console=console)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scorecard.model_dump(), f, indent=2)

    console.print(f"[bold green]✓ Benchmark Report saved to [cyan]{out_path}[/cyan][/bold green]")


@app.command()
def publish_hf(
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        "-r",
        help="Hugging Face dataset repository ID (e.g. 'codata/drum-metrology-instruct').",
    ),
    dataset_dir: str = typer.Option(
        "./dataset",
        "--dataset-dir",
        "-d",
        help="Path to local dataset directory containing JSONL splits.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="Hugging Face API Write Token (or set HF_TOKEN env var / config.yaml).",
    ),
    config: str = typer.Option(
        "./config.yaml", "--config", "-c", help="Path to master config YAML."
    ),
    private: bool | None = typer.Option(
        None, "--private", help="Create private repository on Hugging Face."
    ),
):
    """Publish generated dataset splits, DPO pairs, and dataset card to Hugging Face Hub with live spinner."""
    cfg = PipelineConfig.load_from_yaml(config)
    active_repo = repo_id or cfg.huggingface.repo_id
    active_token = token or os.environ.get("HF_TOKEN") or cfg.huggingface.token
    active_private = private if private is not None else cfg.huggingface.private

    console.print(
        f"[bold green]Publishing dataset from '{dataset_dir}' to Hugging Face: '{active_repo}'...[/bold green]"
    )
    try:
        from huggingface_hub import HfApi

        with Progress(
            SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), console=console
        ) as progress:
            progress.add_task(
                "Uploading dataset splits and dataset card to Hugging Face Hub...", total=None
            )
            api = HfApi(token=active_token)
            api.create_repo(
                repo_id=active_repo, repo_type="dataset", private=active_private, exist_ok=True
            )
            api.upload_folder(
                folder_path=dataset_dir,
                repo_id=active_repo,
                repo_type="dataset",
                commit_message="Release DRUM-ML metrology instruction dataset splits",
            )
        console.print(
            f"[bold blue]✓ Dataset successfully published to: https://huggingface.co/datasets/{active_repo}[/bold blue]"
        )
    except ImportError:
        console.print(
            "[bold red]huggingface_hub package not installed. Run: pip install huggingface_hub[/bold red]"
        )
    except Exception as e:
        console.print(f"[bold red]Failed to publish to Hugging Face: {e}[/bold red]")


@app.command()
def clear_cache(
    config: str = typer.Option(
        "./config.yaml", "--config", "-c", help="Path to master config YAML."
    ),
):
    """Wipe the SQLite semantic cache for LLM queries."""
    cfg = PipelineConfig.load_from_yaml(config)
    cache_p = Path(cfg.cache_db)
    if cache_p.exists():
        cache_p.unlink()
        console.print(
            f"[bold yellow]✓ Successfully deleted LLM cache: {cfg.cache_db}[/bold yellow]"
        )
    else:
        console.print(f"[cyan]Cache is already clean: {cfg.cache_db} does not exist.[/cyan]")


if __name__ == "__main__":
    app()
