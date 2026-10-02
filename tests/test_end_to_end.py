"""End-to-end and pipeline tests across all 5 agents using dynamic RDF Turtle ingestion."""

from drum_ml.pipeline.augmenter import MetrologyAugmenter
from drum_ml.pipeline.dpo_miner import DPOMiner
from drum_ml.pipeline.exporter import MetrologyExporter
from drum_ml.pipeline.extractor import MetrologyExtractor
from drum_ml.pipeline.scaffolder import MetrologyScaffolder
from drum_ml.pipeline.validator import MetrologyValidator


def test_full_pipeline_flow(tmp_path):
    # Setup test Turtle files for dynamic SPARQL extraction
    bipm_dir = tmp_path / "bipm"
    bipm_dir.mkdir(parents=True)
    (bipm_dir / "si-core.ttl").write_text(
        """
        @prefix si: <https://si-digital-framework.org/SI/ontology/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <https://si-digital-framework.org/SI/units/second> a si:BaseUnit ;
            rdfs:label "second" ;
            si:hasSymbol "s" ;
            si:hasDimension "T1" .
        <https://si-digital-framework.org/SI/constants/speed_of_light> a si:DefiningConstant ;
            rdfs:label "Speed of light" ;
            si:hasSymbol "c" ;
            si:hasNumericalValue "299792458" ;
            si:hasUnit "m/s" .
        """,
        encoding="utf-8",
    )

    qudt_dir = tmp_path / "qudt"
    qudt_dir.mkdir(parents=True)
    (qudt_dir / "VOCAB_QUDT-UNITS-ALL.ttl").write_text(
        """
        @prefix qudt: <http://qudt.org/schema/qudt/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        <http://qudt.org/vocab/quantitykind/Force> a qudt:QuantityKind ;
            rdfs:label "Force" ;
            qudt:symbol "F" ;
            qudt:hasDimensionVector <http://qudt.org/vocab/dimensionvector/L1M1T-2> .
        <http://qudt.org/vocab/unit/N> a qudt:Unit ;
            rdfs:label "Newton" ;
            qudt:symbol "N" ;
            qudt:conversionMultiplier 1.0 ;
            qudt:conversionOffset 0.0 ;
            qudt:isSI true ;
            qudt:isCoherent true ;
            qudt:hasDimensionVector <http://qudt.org/vocab/dimensionvector/L1M1T-2> ;
            qudt:hasQuantityKind <http://qudt.org/vocab/quantitykind/Force> .
        """,
        encoding="utf-8",
    )

    codata_dir = tmp_path / "codata"
    codata_dir.mkdir(parents=True)
    (codata_dir / "codata-constants.ttl").write_text(
        """
        @prefix qudt: <http://qudt.org/schema/qudt/> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix drum: <https://codata.org/constants/ontology/> .
        <https://codata.org/constants/2022/G> a qudt:PhysicalConstant ;
            rdfs:label "Newtonian constant of gravitation" ;
            qudt:symbol "G" ;
            qudt:numericValue "6.67430(15)e-11" ;
            qudt:standardUncertainty "0.00015e-11" ;
            qudt:unitSymbol "m^3/(kg*s^2)" ;
            drum:evaluationYear 2022 .
        """,
        encoding="utf-8",
    )

    # 1. Extraction (Agent 1)
    extractor = MetrologyExtractor(
        bipm_dir=str(bipm_dir),
        codata_dir=str(codata_dir),
        qudt_dir=str(qudt_dir),
    )
    store = extractor.extract_all()
    assert len(store.units) > 0
    assert len(store.quantity_kinds) > 0
    assert len(store.constants) > 0

    # 2. Scaffolding (Agent 2)
    scaffolder = MetrologyScaffolder()
    scaffolds = scaffolder.generate_all(store)
    assert len(scaffolds) > 0

    # 3. Augmentation (Agent 3)
    augmenter = MetrologyAugmenter(provider="offline", cache_db_path=str(tmp_path / "cache.sqlite"))
    augmented = augmenter.augment_all(scaffolds[:5], variations_per_archetype=1)
    assert len(augmented) > 0

    # 4. Validation (Agent 4)
    validator = MetrologyValidator()
    approved, reports = validator.validate_all(augmented)
    assert len(approved) > 0

    # 5. DPO Mining & Export (Agent 5)
    miner = DPOMiner()
    dpo_pairs = miner.mine_pairs(augmented, reports)
    exporter = MetrologyExporter(output_dir=str(tmp_path / "dataset"))
    exporter.export_all(approved, dpo_pairs)

    assert (tmp_path / "dataset" / "train.jsonl").exists()
    assert (tmp_path / "dataset" / "val.jsonl").exists()
    assert (tmp_path / "dataset" / "test.jsonl").exists()
    assert (tmp_path / "dataset" / "dataset_card.md").exists()
