"""Unit tests for DatasetStatsGenerator."""

import json
from drum_ml.pipeline.stats_generator import DatasetStatsGenerator


def test_dataset_stats_generator(tmp_path):
    train_data = [
        {
            "id": "rec_1_iupac_chemist_0",
            "archetype": "direct_identification",
            "messages": [
                {"role": "user", "content": "What is the molar volume of an ideal gas?"},
                {"role": "assistant", "content": "The molar volume is $22.71\\text{ L/mol}$."},
            ],
        },
        {
            "id": "rec_2_iau_astronomer_0",
            "archetype": "conversion_scaling",
            "messages": [
                {"role": "user", "content": "How many meters in a parsec?"},
                {"role": "assistant", "content": "$$1\\text{ pc} = 3.085677581\\times 10^{16}\\text{ m}$$"},
            ],
        },
    ]

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    stats = DatasetStatsGenerator.analyze_dataset(str(dataset_dir))
    assert stats["summary"]["total_records"] == 2
    assert stats["summary"]["train_records"] == 2
    assert stats["token_geometry"]["prompt_tokens"]["min"] > 0
    assert stats["math_and_latex"]["inline_math_samples_pct"] > 0

    card_out = dataset_dir / "dataset_card.md"
    DatasetStatsGenerator.generate_huggingface_dataset_card(stats, str(card_out))
    assert card_out.exists()
    card_text = card_out.read_text()
    assert "CODATA DRUM Working Group" in card_text
    assert "Token Geometry" in card_text
