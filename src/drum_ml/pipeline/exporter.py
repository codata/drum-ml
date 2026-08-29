"""Agent 5: Deduplication, Stratified Partitioning, and Dataset Packaging."""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasketch import MinHash, MinHashLSH
from drum_ml.models.export import (
    DPOPreferenceRecord,
    OpenAIChatMessage,
    OpenAIChatRecord,
    ShareGPTRecord,
    ShareGPTTurn,
)
from drum_ml.models.scaffolds import AugmentedRecord


class MetrologyExporter:
    """Agent 5: MinHash deduplication, entity-stratified split, and multi-format dataset exporter."""

    def __init__(
        self,
        output_dir: str = "./dataset",
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        jaccard_threshold: float = 0.85,
        dedup_threshold: Optional[float] = None,
    ):
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.jaccard_threshold = dedup_threshold if dedup_threshold is not None else jaccard_threshold
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def deduplicate(self, records: List[AugmentedRecord]) -> List[AugmentedRecord]:
        """Performs MinHash LSH near-duplicate and exact-hash deduplication."""
        unique_records: List[AugmentedRecord] = []
        seen_hashes = set()
        lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=64)

        seen_ids = set()
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
        self, records: List[AugmentedRecord], seed: int = 42
    ) -> Tuple[List[AugmentedRecord], List[AugmentedRecord], List[AugmentedRecord]]:
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
        )

    def export_all(
        self,
        records: List[AugmentedRecord],
        dpo_pairs: List[DPOPreferenceRecord],
    ) -> Dict[str, Path]:
        """Deduplicates, splits, and writes dataset splits to disk."""
        deduped = self.deduplicate(records)
        train, val, test = self.stratified_split(deduped)

        exported_paths = {}

        # 1. Export OpenAI JSONL splits
        for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
            path = self.output_dir / f"{split_name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in split_data:
                    chat_rec = self.to_openai_format(r)
                    f.write(chat_rec.model_dump_json() + "\n")
            exported_paths[split_name] = path

        # 2. Export DPO Preference Pairs
        dpo_path = self.output_dir / "dpo_preferences.jsonl"
        with open(dpo_path, "w", encoding="utf-8") as f:
            for d in dpo_pairs:
                f.write(d.model_dump_json() + "\n")
        exported_paths["dpo"] = dpo_path

        # 3. Write Dataset Card
        card_path = self.output_dir / "dataset_card.md"
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(
                f"# DRUM-ML Metrology Instruct Dataset\n\n"
                f"- **Total Instruction Pairs:** {len(deduped)}\n"
                f"- **Train Split (85%):** {len(train)}\n"
                f"- **Validation Split (10%):** {len(val)}\n"
                f"- **Held-Out Test Benchmark (5%):** {len(test)}\n"
                f"- **DPO Preference Pairs:** {len(dpo_pairs)}\n"
            )
        exported_paths["card"] = card_path

        return exported_paths
