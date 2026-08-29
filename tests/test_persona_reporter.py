"""Unit tests for PersonaReporter."""

from drum_ml.models.scaffolds import ArchetypeType, AugmentedRecord, PersonaType
from drum_ml.pipeline.persona_reporter import PersonaReporter


def test_persona_reporter_analysis(tmp_path):
    records = [
        AugmentedRecord(
            id="rec_1",
            scaffold_id="scaff_1",
            archetype=ArchetypeType.DIRECT_IDENTIFICATION,
            persona=PersonaType.IUPAC_CHEMIST,
            user_query="What is the standard molar concentration?",
            ground_truth_answer="Answer 1",
            entity_uri="http://qudt.org/vocab/unit/MOL",
            llm_generator="ollama/nemotron-3-nano:4b",
        ),
        AugmentedRecord(
            id="rec_2",
            scaffold_id="scaff_2",
            archetype=ArchetypeType.DIMENSIONAL_DECOMPOSITION,
            persona=PersonaType.IAU_ASTRONOMER,
            user_query="What is a parsec in base SI units?",
            ground_truth_answer="Answer 2",
            entity_uri="http://qudt.org/vocab/unit/PARSEC",
            llm_generator="ollama/nemotron-3-nano:4b",
        ),
    ]

    report = PersonaReporter.analyze_records(records)
    assert report["total_samples"] == 2
    assert report["unique_personas_count"] == 2
    assert report["archetype_distribution"]["direct_identification"] == 1
    assert report["archetype_distribution"]["dimensional_decomposition"] == 1

    md_out = tmp_path / "persona_report.md"
    json_out = tmp_path / "persona_report.json"

    PersonaReporter.save_markdown_report(report, str(md_out))
    PersonaReporter.save_json_report(report, str(json_out))

    assert md_out.exists()
    assert json_out.exists()
    assert "International Union of Pure and Applied Chemistry" in md_out.read_text()
    assert "International Astronomical Union" in md_out.read_text()
