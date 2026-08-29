"""Unit tests for ArchetypeReporter."""

from drum_ml.models.scaffolds import ArchetypeType, AugmentedRecord, PersonaType
from drum_ml.pipeline.archetype_reporter import ArchetypeReporter


def test_archetype_reporter_analysis(tmp_path):
    records = [
        AugmentedRecord(
            id="rec_1",
            scaffold_id="scaff_1",
            archetype=ArchetypeType.DIRECT_IDENTIFICATION,
            persona=PersonaType.IUPAC_CHEMIST,
            user_query="What is a mole?",
            ground_truth_answer="The mole is $6.022\\times 10^{23}\\text{ entities}$.",
            entity_uri="http://qudt.org/vocab/unit/MOL",
            llm_generator="ollama/nemotron-3-nano:4b",
        ),
        AugmentedRecord(
            id="rec_2",
            scaffold_id="scaff_2",
            archetype=ArchetypeType.DIMENSIONAL_DECOMPOSITION,
            persona=PersonaType.IAU_ASTRONOMER,
            user_query="What is the base SI breakdown of a parsec?",
            ground_truth_answer="$$1\\text{ pc} = 3.085677581\\times 10^{16}\\text{ m}$$",
            entity_uri="http://qudt.org/vocab/unit/PARSEC",
            llm_generator="ollama/nemotron-3-nano:4b",
        ),
    ]

    report = ArchetypeReporter.analyze_records(records)
    assert report["total_samples"] == 2
    assert report["unique_archetypes"] == 2

    md_out = tmp_path / "archetype_report.md"
    json_out = tmp_path / "archetype_report.json"

    ArchetypeReporter.save_markdown_report(report, str(md_out))
    ArchetypeReporter.save_json_report(report, str(json_out))

    assert md_out.exists()
    assert json_out.exists()
    assert "Direct Identification & Symbol Mapping" in md_out.read_text()
    assert "Dimensional Decomposition & Base SI" in md_out.read_text()
