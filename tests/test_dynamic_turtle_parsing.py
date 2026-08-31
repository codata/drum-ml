"""Tests for 100% dynamic SPARQL extraction from RDF Turtle graphs with zero hardcoding."""

from drum_ml.data_sources.bipm_client import BIPMClient
from drum_ml.data_sources.codata_client import CODATAClient
from drum_ml.data_sources.qudt_fetcher import QUDTFetcher


def test_dynamic_bipm_turtle_parsing(tmp_path):
    # Create sample BIPM SI Turtle file dynamically
    ttl_content = """
    @prefix si: <https://si-digital-framework.org/SI/ontology/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <https://si-digital-framework.org/SI/units/second> a si:BaseUnit ;
        rdfs:label "second" ;
        si:hasSymbol "s" ;
        si:hasDimension "T1" .

    <https://si-digital-framework.org/SI/constants/speed_of_light> a si:DefiningConstant ;
        rdfs:label "Speed of light in vacuum" ;
        si:hasSymbol "c" ;
        si:hasNumericalValue "299792458" ;
        si:hasUnit "m/s" .
    """
    ttl_file = tmp_path / "si-core.ttl"
    ttl_file.write_text(ttl_content, encoding="utf-8")

    client = BIPMClient(local_dir=str(tmp_path))
    graph = client.load_local_graph()
    units, constants = client.extract_all(graph)

    assert len(units) == 1
    assert "https://si-digital-framework.org/SI/units/second" in units
    assert units["https://si-digital-framework.org/SI/units/second"].symbol == "s"
    assert units["https://si-digital-framework.org/SI/units/second"].dimension_vector.T == 1

    assert len(constants) == 1
    c_entry = next(iter(constants.values()))
    assert c_entry.symbol == "c"
    assert c_entry.numeric_value == "299792458"
    assert c_entry.category.value == "exact_si_defining"


def test_dynamic_qudt_turtle_parsing(tmp_path):
    ttl_content = """
    @prefix qudt: <http://qudt.org/schema/qudt/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <http://qudt.org/vocab/quantitykind/Torque> a qudt:QuantityKind ;
        rdfs:label "Torque" ;
        qudt:symbol "T" ;
        qudt:hasDimensionVector <http://qudt.org/vocab/dimensionvector/A0E0L2I0M1H0T-2D0> ;
        qudt:applicableUnit <http://qudt.org/vocab/unit/N-M> .

    <http://qudt.org/vocab/unit/N-M> a qudt:Unit ;
        rdfs:label "Newton Metre" ;
        qudt:symbol "N-m" ;
        qudt:conversionMultiplier 1.0 ;
        qudt:conversionOffset 0.0 ;
        qudt:isSI true ;
        qudt:isCoherent true ;
        qudt:hasDimensionVector <http://qudt.org/vocab/dimensionvector/A0E0L2I0M1H0T-2D0> ;
        qudt:hasQuantityKind <http://qudt.org/vocab/quantitykind/Torque> .
    """
    ttl_file = tmp_path / "VOCAB_QUDT-UNITS-ALL.ttl"
    ttl_file.write_text(ttl_content, encoding="utf-8")

    fetcher = QUDTFetcher(local_dir=str(tmp_path))
    graph = fetcher.load_local_graph()
    units, kinds = fetcher.extract_all(graph)

    assert len(units) == 1
    u = units["http://qudt.org/vocab/unit/N-M"]
    assert u.symbol == "N-m"
    assert u.dimension_vector.L == 2
    assert u.dimension_vector.M == 1
    assert u.dimension_vector.T == -2

    assert len(kinds) == 1
    k = kinds["http://qudt.org/vocab/quantitykind/Torque"]
    assert k.label == "Torque"
    assert k.dimension_vector.L == 2
    assert "http://qudt.org/vocab/unit/N-M" in k.applicable_units


def test_dynamic_codata_turtle_parsing(tmp_path):
    ttl_content = """
    @prefix qudt: <http://qudt.org/schema/qudt/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix drum: <https://codata.org/constants/ontology/> .

    <https://codata.org/constants/2022/G> a qudt:PhysicalConstant ;
        rdfs:label "Newtonian constant of gravitation" ;
        qudt:symbol "G" ;
        qudt:numericValue "6.67430(15)e-11" ;
        qudt:standardUncertainty "0.00015e-11" ;
        qudt:relativeUncertainty "2.2e-5" ;
        qudt:unitSymbol "m^3/(kg*s^2)" ;
        drum:evaluationYear 2022 .
    """
    ttl_file = tmp_path / "codata-constants.ttl"
    ttl_file.write_text(ttl_content, encoding="utf-8")

    client = CODATAClient(local_dir=str(tmp_path))
    graph = client.load_local_graph()
    constants = client.extract_from_graph(graph, year=2022)

    assert len(constants) == 1
    g_const = constants["https://codata.org/constants/2022/G"]
    assert g_const.symbol == "G"
    assert g_const.numeric_value == "6.67430e-11"
    assert g_const.standard_uncertainty == "0.00015e-11"
    assert g_const.relative_uncertainty == "2.2e-5"
    assert g_const.defining_year == 2022
