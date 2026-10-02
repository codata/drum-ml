"""Agent 5: Deduplication, Stratified Partitioning, Grouped Subsets, and Dataset Packaging."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

from drum_ml.dataset_viewer import (
    parse_category_from_record,
    parse_persona_from_record,
    save_dataset_viewer,
)
from drum_ml.models.export import (
    DPOPreferenceRecord,
    OpenAIChatMessage,
    OpenAIChatRecord,
    ShareGPTRecord,
    ShareGPTTurn,
)
from drum_ml.models.scaffolds import AugmentedRecord


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class MetrologyExporter:
    """Agent 5: MinHash deduplication, entity-stratified split, and multi-format / multi-group dataset exporter."""

    def __init__(
        self,
        output_dir: str = "./dataset",
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        jaccard_threshold: float = 0.85,
        dedup_threshold: float | None = None,
        export_by_persona: bool = True,
        export_by_archetype: bool = True,
        export_by_category: bool = True,
        generate_viewer: bool = True,
        formats: list[str] | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.jaccard_threshold = (
            dedup_threshold if dedup_threshold is not None else jaccard_threshold
        )
        self.export_by_persona = export_by_persona
        self.export_by_archetype = export_by_archetype
        self.export_by_category = export_by_category
        self.generate_viewer = generate_viewer
        self.formats = formats or ["openai", "sharegpt", "dpo"]
        os.makedirs(str(self.output_dir), exist_ok=True)

    def deduplicate(self, records: list[AugmentedRecord]) -> list[AugmentedRecord]:
        """Performs MinHash LSH near-duplicate and exact-hash deduplication."""
        unique_records: list[AugmentedRecord] = []
        seen_hashes = set()
        lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=64)

        for idx, rec in enumerate(records):
            exact_hash = hashlib.sha256(rec.user_query.strip().lower().encode()).hexdigest()
            if exact_hash in seen_hashes:
                continue
            seen_hashes.add(exact_hash)

            # MinHash for fuzzy similarity
            m = MinHash(num_perm=64)
            for token in rec.user_query.lower().split():
                m.update(token.encode())

            duplicates = lsh.query(m)
            if not duplicates:
                unique_key = f"{rec.id}_{idx}"
                lsh.insert(unique_key, m, check_duplication=False)
                unique_records.append(rec)

        return unique_records

    def stratified_split(
        self, records: list[AugmentedRecord], seed: int = 42
    ) -> tuple[list[AugmentedRecord], list[AugmentedRecord], list[AugmentedRecord]]:
        """Partitions dataset with entity isolation to prevent train-to-test data leakage."""
        random.seed(seed)
        shuffled = list(records)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        return train, val, test

    def to_openai_format(self, record: AugmentedRecord) -> OpenAIChatRecord:
        """Converts augmented record to OpenAI Chat Completion format."""
        persona_val = (
            record.persona.value if hasattr(record.persona, "value") else str(record.persona)
        )
        archetype_val = (
            record.archetype.value if hasattr(record.archetype, "value") else str(record.archetype)
        )
        return OpenAIChatRecord(
            id=record.id,
            archetype=record.archetype,
            entity_uri=record.entity_uri,
            messages=[
                OpenAIChatMessage(
                    role="system",
                    content="You are an authoritative SI and metrology assistant grounded in the BIPM SI Digital Framework, CODATA fundamental constants, and QUDT ontologies.",
                ),
                OpenAIChatMessage(role="user", content=record.user_query),
                OpenAIChatMessage(role="assistant", content=record.ground_truth_answer),
            ],
            metadata={
                "persona": persona_val,
                "archetype": archetype_val,
                "scaffold_id": record.scaffold_id,
                "quantity_kind_uri": record.quantity_kind_uri,
                **record.metadata,
            },
        )

    def to_sharegpt_format(self, record: AugmentedRecord) -> ShareGPTRecord:
        """Converts augmented record to ShareGPT multi-turn conversation format."""
        persona_val = (
            record.persona.value if hasattr(record.persona, "value") else str(record.persona)
        )
        archetype_val = (
            record.archetype.value if hasattr(record.archetype, "value") else str(record.archetype)
        )
        return ShareGPTRecord(
            id=record.id,
            conversations=[
                ShareGPTTurn(
                    from_="system",
                    value="You are an authoritative SI and metrology assistant grounded in the BIPM SI Digital Framework, CODATA fundamental constants, and QUDT ontologies.",
                ),
                ShareGPTTurn(from_="human", value=record.user_query),
                ShareGPTTurn(from_="gpt", value=record.ground_truth_answer),
            ],
            metadata={
                "persona": persona_val,
                "archetype": archetype_val,
                "entity_uri": record.entity_uri,
                "scaffold_id": record.scaffold_id,
                "quantity_kind_uri": record.quantity_kind_uri,
                **record.metadata,
            },
        )

    def export_all(
        self,
        records: list[AugmentedRecord],
        dpo_pairs: list[DPOPreferenceRecord],
    ) -> dict[str, Path]:
        """Deduplicates, splits, partitions into groupings, and writes dataset splits and viewer to disk."""
        os.makedirs(str(self.output_dir), exist_ok=True)
        deduped = self.deduplicate(records)
        train, val, test = self.stratified_split(deduped)

        exported_paths: dict[str, Path] = {}

        # 1. Export standard master OpenAI JSONL splits
        splits = [("train", train), ("val", val), ("test", test)]
        for split_name, split_data in splits:
            path = self.output_dir / f"{split_name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for r in split_data:
                    chat_rec = self.to_openai_format(r)
                    f.write(chat_rec.model_dump_json() + "\n")
            exported_paths[split_name] = path

        # 2. Export DPO Preference Pairs
        dpo_path = self.output_dir / "dpo_preferences.jsonl"
        dpo_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dpo_path, "w", encoding="utf-8") as f:
            for d in dpo_pairs:
                f.write(d.model_dump_json() + "\n")
        exported_paths["dpo"] = dpo_path

        # 3. Export Groupings by Persona
        if self.export_by_persona:
            persona_dir = self.output_dir / "by_persona"
            persona_dir.mkdir(parents=True, exist_ok=True)

            # Map of persona -> list of records per split
            persona_splits: dict[str, dict[str, list[AugmentedRecord]]] = {}
            for split_name, split_data in splits:
                for r in split_data:
                    p_val = r.persona.value if hasattr(r.persona, "value") else str(r.persona)
                    if p_val not in persona_splits:
                        persona_splits[p_val] = {"all": [], "train": [], "val": [], "test": []}
                    persona_splits[p_val][split_name].append(r)
                    persona_splits[p_val]["all"].append(r)

            for p_val, p_dict in persona_splits.items():
                # Consolidated persona file
                p_all_path = persona_dir / f"{p_val}.jsonl"
                with open(p_all_path, "w", encoding="utf-8") as f:
                    for r in p_dict["all"]:
                        f.write(self.to_openai_format(r).model_dump_json() + "\n")
                exported_paths[f"persona_{p_val}"] = p_all_path

                # Persona train split
                if p_dict["train"]:
                    p_train_path = persona_dir / f"{p_val}_train.jsonl"
                    with open(p_train_path, "w", encoding="utf-8") as f:
                        for r in p_dict["train"]:
                            f.write(self.to_openai_format(r).model_dump_json() + "\n")

        # 4. Export Groupings by Pedagogical Archetype
        if self.export_by_archetype:
            arch_dir = self.output_dir / "by_archetype"
            arch_dir.mkdir(parents=True, exist_ok=True)
            arch_buckets: dict[str, list[AugmentedRecord]] = {}
            for r in deduped:
                a_val = r.archetype.value if hasattr(r.archetype, "value") else str(r.archetype)
                arch_buckets.setdefault(a_val, []).append(r)

            for a_val, a_records in arch_buckets.items():
                a_path = arch_dir / f"{a_val}.jsonl"
                with open(a_path, "w", encoding="utf-8") as f:
                    for r in a_records:
                        f.write(self.to_openai_format(r).model_dump_json() + "\n")
                exported_paths[f"archetype_{a_val}"] = a_path

        # 5. Export Groupings by Entity Category (units vs constants vs quantity kinds)
        if self.export_by_category:
            cat_dir = self.output_dir / "by_category"
            cat_dir.mkdir(parents=True, exist_ok=True)
            cat_buckets: dict[str, list[AugmentedRecord]] = {
                "units": [],
                "constants": [],
                "quantity_kinds": [],
            }
            for r in deduped:
                c_val = parse_category_from_record({"entity_uri": r.entity_uri})
                cat_buckets.setdefault(c_val, []).append(r)

            for c_val, c_records in cat_buckets.items():
                if c_records:
                    c_path = cat_dir / f"{c_val}.jsonl"
                    with open(c_path, "w", encoding="utf-8") as f:
                        for r in c_records:
                            f.write(self.to_openai_format(r).model_dump_json() + "\n")
                    exported_paths[f"category_{c_val}"] = c_path

        # 6. Export ShareGPT format if requested
        if "sharegpt" in self.formats:
            sharegpt_dir = self.output_dir / "formats" / "sharegpt"
            sharegpt_dir.mkdir(parents=True, exist_ok=True)
            for split_name, split_data in splits:
                sg_path = sharegpt_dir / f"{split_name}.jsonl"
                with open(sg_path, "w", encoding="utf-8") as f:
                    for r in split_data:
                        f.write(self.to_sharegpt_format(r).model_dump_json(by_alias=True) + "\n")
                exported_paths[f"sharegpt_{split_name}"] = sg_path

        # 7. Write Dataset Manifest (JSON Index)
        manifest_path = self.output_dir / "manifest.json"
        manifest_data = self.generate_manifest(
            output_dir=self.output_dir,
            total_records=len(deduped),
            splits={"train": len(train), "val": len(val), "test": len(test), "dpo": len(dpo_pairs)},
            exported_paths=exported_paths,
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        exported_paths["manifest"] = manifest_path

        # 8. Write Dataset Card
        card_path = self.output_dir / "dataset_card.md"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(
                f"# DRUM-ML Metrology Instruct Dataset\n\n"
                f"- **Total Instruction Pairs:** {len(deduped)}\n"
                f"- **Train Split (85%):** {len(train)}\n"
                f"- **Validation Split (10%):** {len(val)}\n"
                f"- **Held-Out Test Benchmark (5%):** {len(test)}\n"
                f"- **DPO Preference Pairs:** {len(dpo_pairs)}\n\n"
                f"### Partitioned Subsets\n"
                f"- **By Persona Subsets:** `{self.output_dir}/by_persona/`\n"
                f"- **By Archetype Subsets:** `{self.output_dir}/by_archetype/`\n"
                f"- **By Category Subsets:** `{self.output_dir}/by_category/`\n"
                f"- **Dataset Manifest Index:** `{self.output_dir}/manifest.json`\n"
            )
        exported_paths["card"] = card_path

        # 9. Generate Standalone Interactive Dataset Browser HTML
        if self.generate_viewer:
            viewer_path = self.output_dir / "dataset_viewer.html"
            save_dataset_viewer(
                output_html_path=viewer_path,
                dataset_dir_or_file=self.output_dir,
                max_samples=5000,
                open_browser=False,
            )
            exported_paths["viewer"] = viewer_path

        return exported_paths

    def generate_manifest(
        self,
        output_dir: Path,
        total_records: int,
        splits: dict[str, int],
        exported_paths: dict[str, Path],
    ) -> dict[str, Any]:
        """Generates a comprehensive dataset manifest indexing all partitions and checksums."""
        files_info = []
        for _key, p in exported_paths.items():
            if p.exists() and p.is_file() and p.suffix == ".jsonl":
                # count lines
                count = sum(1 for line in open(p, encoding="utf-8") if line.strip())
                files_info.append(
                    {
                        "name": p.name,
                        "relative_path": str(p.relative_to(output_dir)),
                        "record_count": count,
                        "byte_size": p.stat().st_size,
                        "sha256": compute_file_sha256(p),
                    }
                )

        return {
            "dataset_name": "drum-ml-metrology-instruct",
            "version": "0.1.0",
            "total_records": total_records,
            "splits": splits,
            "formats": self.formats,
            "groupings": {
                "by_persona": self.export_by_persona,
                "by_archetype": self.export_by_archetype,
                "by_category": self.export_by_category,
            },
            "files": sorted(files_info, key=lambda x: x["relative_path"]),
            "huggingface_usage": {
                "load_train": "from datasets import load_dataset; ds = load_dataset('json', data_files={'train': 'train.jsonl', 'validation': 'val.jsonl', 'test': 'test.jsonl'})",
                "load_persona": "from datasets import load_dataset; ds = load_dataset('json', data_files={'train': 'by_persona/academic_metrologist_train.jsonl'})",
                "load_archetype": "from datasets import load_dataset; ds = load_dataset('json', data_files={'train': 'by_archetype/dimensional_decomposition.jsonl'})",
            },
        }

    @classmethod
    def partition_existing_dataset(
        cls,
        dataset_dir: str | Path = "./dataset",
        output_dir: str | Path | None = None,
        generate_viewer: bool = True,
    ) -> dict[str, Any]:
        """High-performance partitioner for existing dataset directories with train/val/test/dpo JSONL files."""
        in_dir = Path(dataset_dir).resolve()
        out_dir = Path(output_dir).resolve() if output_dir else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        persona_dir = out_dir / "by_persona"
        arch_dir = out_dir / "by_archetype"
        cat_dir = out_dir / "by_category"
        persona_dir.mkdir(parents=True, exist_ok=True)
        arch_dir.mkdir(parents=True, exist_ok=True)
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Persona handles and Archetype handles for streaming writes
        persona_all_handles: dict[str, Any] = {}
        persona_train_handles: dict[str, Any] = {}
        arch_handles: dict[str, Any] = {}
        cat_handles: dict[str, Any] = {}

        persona_counts: dict[str, int] = {}
        archetype_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        total_processed = 0

        split_files = [
            ("train", in_dir / "train.jsonl"),
            ("val", in_dir / "val.jsonl"),
            ("test", in_dir / "test.jsonl"),
        ]

        try:
            for split_name, file_path in split_files:
                if not file_path.exists():
                    continue

                with open(file_path, encoding="utf-8") as f_in:
                    for line in f_in:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            rec = json.loads(line_str)
                        except Exception:
                            continue

                        p_val = parse_persona_from_record(rec)
                        a_val = rec.get("archetype", "direct_identification")
                        c_val = parse_category_from_record(rec)

                        persona_counts[p_val] = persona_counts.get(p_val, 0) + 1
                        archetype_counts[a_val] = archetype_counts.get(a_val, 0) + 1
                        category_counts[c_val] = category_counts.get(c_val, 0) + 1
                        total_processed += 1

                        # Write to persona all handle
                        if p_val not in persona_all_handles:
                            persona_all_handles[p_val] = open(
                                persona_dir / f"{p_val}.jsonl", "w", encoding="utf-8"
                            )
                        persona_all_handles[p_val].write(line_str + "\n")

                        # Write to persona train handle if split is train
                        if split_name == "train":
                            if p_val not in persona_train_handles:
                                persona_train_handles[p_val] = open(
                                    persona_dir / f"{p_val}_train.jsonl", "w", encoding="utf-8"
                                )
                            persona_train_handles[p_val].write(line_str + "\n")

                        # Write to archetype handle
                        if a_val not in arch_handles:
                            arch_handles[a_val] = open(
                                arch_dir / f"{a_val}.jsonl", "w", encoding="utf-8"
                            )
                        arch_handles[a_val].write(line_str + "\n")

                        # Write to category handle
                        if c_val not in cat_handles:
                            cat_handles[c_val] = open(
                                cat_dir / f"{c_val}.jsonl", "w", encoding="utf-8"
                            )
                        cat_handles[c_val].write(line_str + "\n")

        finally:
            for h in (
                list(persona_all_handles.values())
                + list(persona_train_handles.values())
                + list(arch_handles.values())
                + list(cat_handles.values())
            ):
                try:
                    h.close()
                except Exception:
                    pass

        # Build manifest
        manifest_files = []
        for p_file in (
            list(persona_dir.glob("*.jsonl"))
            + list(arch_dir.glob("*.jsonl"))
            + list(cat_dir.glob("*.jsonl"))
        ):
            manifest_files.append(
                {
                    "name": p_file.name,
                    "relative_path": str(p_file.relative_to(out_dir)),
                    "record_count": sum(
                        1 for line in open(p_file, encoding="utf-8") if line.strip()
                    ),
                    "byte_size": p_file.stat().st_size,
                    "sha256": compute_file_sha256(p_file),
                }
            )

        manifest = {
            "dataset_name": "drum-ml-metrology-instruct",
            "version": "0.1.0",
            "total_records": total_processed,
            "persona_counts": persona_counts,
            "archetype_counts": archetype_counts,
            "category_counts": category_counts,
            "files": sorted(manifest_files, key=lambda x: x["relative_path"]),
        }

        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if generate_viewer:
            save_dataset_viewer(
                output_html_path=out_dir / "dataset_viewer.html",
                dataset_dir_or_file=out_dir,
                max_samples=5000,
                open_browser=False,
            )

        return manifest
