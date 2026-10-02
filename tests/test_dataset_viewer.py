"""Unit tests for the interactive dataset browser & viewer."""

from pathlib import Path

from drum_ml.dataset_viewer import (
    generate_dataset_html,
    load_dataset_records,
    normalize_record_for_viewer,
    save_dataset_viewer,
)


def test_normalize_record_openai_format():
    sample = {
        "id": "N_arch1_physics_student_0",
        "archetype": "direct_identification",
        "entity_uri": "http://qudt.org/vocab/unit/N",
        "messages": [
            {"role": "system", "content": "SI Assistant"},
            {"role": "user", "content": "What is a newton?"},
            {"role": "assistant", "content": "A newton is the SI coherent derived unit of force."},
        ],
        "metadata": {"persona": "physics_student"},
    }
    norm = normalize_record_for_viewer(sample, default_split="train")
    assert norm["id"] == "N_arch1_physics_student_0"
    assert norm["persona"] == "physics_student"
    assert norm["archetype"] == "direct_identification"
    assert norm["user_query"] == "What is a newton?"
    assert "force" in norm["ground_truth_answer"]
    assert norm["split"] == "train"
    assert not norm["is_dpo"]


def test_normalize_record_dpo_format():
    sample = {
        "id": "dpo_rec_1",
        "archetype": "error_detection",
        "entity_uri": "http://qudt.org/vocab/unit/Joule",
        "prompt": "Is adding Joules and Newtons valid?",
        "chosen": "No, dimensional homogeneity forbids adding energy [L2 M T-2] to force [L M T-2].",
        "rejected": "Yes, both are SI units so they can be added.",
        "rejection_reason": "Dimensional homogeneity violation",
    }
    norm = normalize_record_for_viewer(sample)
    assert norm["split"] == "dpo"
    assert norm["is_dpo"]
    assert norm["user_query"] == sample["prompt"]
    assert norm["ground_truth_answer"] == sample["chosen"]
    assert norm["rejected"] == sample["rejected"]
    assert norm["rejection_reason"] == sample["rejection_reason"]


def test_generate_and_save_dataset_html(tmp_path: Path):
    # Create sample JSONL file
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        '{"id":"rec_1_academic_metrologist_0","archetype":"direct_identification","entity_uri":"http://qudt.org/vocab/unit/M","messages":[{"role":"system","content":"SI"},{"role":"user","content":"Define meter."},{"role":"assistant","content":"Meter is base unit."}],"metadata":{"persona":"academic_metrologist"}}\n',
        encoding="utf-8",
    )

    records, stats = load_dataset_records(tmp_path, max_samples=100)
    assert len(records) == 1
    assert stats["total_records"] == 1
    assert stats["splits"]["train"] == 1

    html = generate_dataset_html(records, stats)
    assert "<!DOCTYPE html>" in html
    assert "DRUM-ML" in html
    assert "academic_metrologist" in html

    out_html = tmp_path / "dataset_viewer.html"
    saved_path = save_dataset_viewer(out_html, dataset_dir_or_file=tmp_path, max_samples=100)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 5000
