"""Unit tests for MetrologyExporter and dataset partitioning."""

from pathlib import Path

from drum_ml.models.export import DPOPreferenceRecord
from drum_ml.models.scaffolds import ArchetypeType, AugmentedRecord, PersonaType
from drum_ml.pipeline.exporter import MetrologyExporter


def test_exporter_partitioning_and_manifest(tmp_path: Path):
    output_dir = tmp_path / "dataset_out"

    rec1 = AugmentedRecord(
        id="unit_m_academic_metrologist_0",
        scaffold_id="unit_m",
        archetype=ArchetypeType.DIRECT_IDENTIFICATION,
        persona=PersonaType.ACADEMIC_METROLOGIST,
        user_query="Define meter in SI 2019.",
        ground_truth_answer="The meter is defined by taking the fixed numerical value of the speed of light in vacuum c.",
        entity_uri="http://qudt.org/vocab/unit/M",
        llm_generator="ground_truth",
    )
    rec2 = AugmentedRecord(
        id="unit_s_physics_student_0",
        scaffold_id="unit_s",
        archetype=ArchetypeType.DIMENSIONAL_DECOMPOSITION,
        persona=PersonaType.PHYSICS_STUDENT,
        user_query="What is the base SI unit of time?",
        ground_truth_answer="The second (symbol: s) is the SI base unit of time.",
        entity_uri="http://qudt.org/vocab/unit/SEC",
        llm_generator="ground_truth",
    )
    dpo_pair = DPOPreferenceRecord(
        id="dpo_1",
        archetype=ArchetypeType.DIMENSIONAL_ERROR_DETECTION,
        entity_uri="http://qudt.org/vocab/unit/Joule",
        prompt="Can you sum 5 J and 3 N?",
        chosen="No, dimensional homogeneity forbids summing quantities with different dimensions.",
        rejected="Yes, 5 J + 3 N = 8.",
        rejection_reason="Dimensional homogeneity violation",
    )

    exporter = MetrologyExporter(
        output_dir=str(output_dir),
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        export_by_persona=True,
        export_by_archetype=True,
        export_by_category=True,
        generate_viewer=True,
    )

    paths = exporter.export_all([rec1, rec2], [dpo_pair])
    assert "train" in paths
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "dpo_preferences.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "dataset_card.md").exists()
    assert (output_dir / "dataset_viewer.html").exists()

    # Check persona subsets
    assert (output_dir / "by_persona" / "academic_metrologist.jsonl").exists()
    assert (output_dir / "by_persona" / "physics_student.jsonl").exists()

    # Check archetype subsets
    assert (output_dir / "by_archetype" / "direct_identification.jsonl").exists()
    assert (output_dir / "by_archetype" / "dimensional_decomposition.jsonl").exists()

    # Check category subsets
    assert (output_dir / "by_category" / "units.jsonl").exists()


def test_partition_existing_dataset(tmp_path: Path):
    in_dir = tmp_path / "in_data"
    in_dir.mkdir()
    train_file = in_dir / "train.jsonl"
    train_file.write_text(
        '{"id":"m_iupap_physicist_0","archetype":"direct_identification","entity_uri":"http://qudt.org/vocab/unit/M","messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}],"metadata":{"persona":"iupap_physicist"}}\n',
        encoding="utf-8",
    )

    manifest = MetrologyExporter.partition_existing_dataset(
        dataset_dir=in_dir,
        output_dir=in_dir,
        generate_viewer=True,
    )
    assert manifest["total_records"] == 1
    assert "iupap_physicist" in manifest["persona_counts"]
    assert (in_dir / "by_persona" / "iupap_physicist.jsonl").exists()
    assert (in_dir / "by_persona" / "iupap_physicist_train.jsonl").exists()
    assert (in_dir / "dataset_viewer.html").exists()
