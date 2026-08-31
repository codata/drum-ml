"""QUDT 2.1 Dynamic Vocabulary Downloader and SPARQL Ingestion Client."""

from pathlib import Path
from typing import ClassVar

import httpx
import rdflib

from drum_ml.models.entities import (
    ConversionRelation,
    QuantityKindEntity,
    UnitEntity,
)
from drum_ml.symbolic.dimensions import parse_qudt_dimension_string


class QUDTFetcher:
    """Client for downloading latest QUDT releases and querying units/quantity kinds via SPARQL."""

    QUDT_VOCAB_URLS: ClassVar[dict[str, str]] = {
        "VOCAB_QUDT-UNITS-ALL.ttl": "https://qudt.org/2.1/vocab/unit",
        "VOCAB_QUDT-QUANTITY-KINDS-ALL.ttl": "https://qudt.org/2.1/vocab/quantitykind",
        "VOCAB_QUDT-DIMENSION-VECTORS-ALL.ttl": "https://qudt.org/2.1/vocab/dimensionvector",
        "VOCAB_QUDT-PREFIXES-ALL.ttl": "https://qudt.org/2.1/vocab/prefix",
    }

    def __init__(self, local_dir: str = "./data/raw/qudt", release_tag: str = "v2.1.37"):
        self.local_dir = Path(local_dir)
        self.release_tag = release_tag
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def download_latest_vocabularies(self, tag: str | None = None) -> list[Path]:
        """Downloads official QUDT TTL vocabularies from official endpoints."""
        downloaded = []
        headers = {"Accept": "text/turtle, application/x-turtle;q=0.9, text/plain;q=0.5"}
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            for fname, url in self.QUDT_VOCAB_URLS.items():
                out_path = self.local_dir / fname
                try:
                    resp = client.get(url, headers=headers)
                    if resp.status_code == 200:
                        with open(out_path, "wb") as f:
                            f.write(resp.content)
                        downloaded.append(out_path)
                except Exception:
                    pass
        return downloaded

    def load_local_graph(self) -> rdflib.Graph:
        """Loads all available local QUDT Turtle files into an rdflib Graph."""
        g = rdflib.Graph()
        for ttl_path in self.local_dir.glob("*.ttl"):
            try:
                g.parse(str(ttl_path), format="turtle")
            except Exception:
                pass
        return g

    def extract_quantity_kinds_from_graph(
        self, graph: rdflib.Graph
    ) -> dict[str, QuantityKindEntity]:
        """Extracts QuantityKinds dynamically via SPARQL."""
        kinds: dict[str, QuantityKindEntity] = {}
        query = """
        PREFIX qudt: <http://qudt.org/schema/qudt/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?qk ?label ?symbol ?description ?dimVector ?unit
        WHERE {
          ?qk a qudt:QuantityKind ;
              rdfs:label ?label .
          OPTIONAL { ?qk qudt:symbol ?symbol }
          OPTIONAL { ?qk qudt:description ?description }
          OPTIONAL { ?qk qudt:hasDimensionVector ?dimVector }
          OPTIONAL { ?qk qudt:applicableUnit ?unit }
        }
        """
        for row in graph.query(query):
            uri = str(row.qk)
            dim_str = str(row.dimVector).split("/")[-1] if row.dimVector else ""
            dim_vec = parse_qudt_dimension_string(dim_str)

            if uri not in kinds:
                kinds[uri] = QuantityKindEntity(
                    uri=uri,
                    label=str(row.label),
                    symbol=str(row.symbol) if row.symbol else None,
                    description=str(row.description) if row.description else None,
                    dimension_vector=dim_vec,
                    applicable_units=[],
                )
            if row.unit:
                unit_uri = str(row.unit)
                if unit_uri not in kinds[uri].applicable_units:
                    kinds[uri].applicable_units.append(unit_uri)

        return kinds

    def extract_units_from_graph(self, graph: rdflib.Graph) -> dict[str, UnitEntity]:
        """Extracts Units dynamically via SPARQL from the QUDT RDF Graph."""
        units: dict[str, UnitEntity] = {}
        query = """
        PREFIX qudt: <http://qudt.org/schema/qudt/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?unit ?symbol ?label ?description ?multiplier ?offset ?isSI ?isCoherent ?ucumCode ?dimVector ?qk
        WHERE {
          ?unit a qudt:Unit ;
                rdfs:label ?label .
          OPTIONAL { ?unit qudt:symbol ?symbol }
          OPTIONAL { ?unit qudt:description ?description }
          OPTIONAL { ?unit qudt:conversionMultiplier ?multiplier }
          OPTIONAL { ?unit qudt:conversionOffset ?offset }
          OPTIONAL { ?unit qudt:isSI ?isSI }
          OPTIONAL { ?unit qudt:isCoherent ?isCoherent }
          OPTIONAL { ?unit qudt:ucumCode ?ucumCode }
          OPTIONAL { ?unit qudt:hasDimensionVector ?dimVector }
          OPTIONAL { ?unit qudt:hasQuantityKind ?qk }
        }
        """
        for row in graph.query(query):
            uri = str(row.unit)
            symbol = str(row.symbol) if row.symbol else str(row.label)
            dim_str = str(row.dimVector).split("/")[-1] if row.dimVector else ""
            dim_vec = parse_qudt_dimension_string(dim_str)

            mult = float(row.multiplier) if row.multiplier else 1.0
            offset = float(row.offset) if row.offset else 0.0
            is_si = bool(row.isSI) if row.isSI else False
            is_coh = bool(row.isCoherent) if row.isCoherent else False

            if uri not in units:
                units[uri] = UnitEntity(
                    uri=uri,
                    symbol=symbol,
                    label=str(row.label),
                    description=str(row.description) if row.description else None,
                    is_si_base=False,
                    is_si_derived=is_si,
                    is_coherent=is_coh,
                    dimension_vector=dim_vec,
                    ucum_code=str(row.ucumCode) if row.ucumCode else None,
                    conversion=ConversionRelation(multiplier=mult, offset=offset),
                    has_quantity_kinds=[],
                )
            if row.qk:
                qk_uri = str(row.qk)
                if qk_uri not in units[uri].has_quantity_kinds:
                    units[uri].has_quantity_kinds.append(qk_uri)

        return units

    def extract_all(
        self, graph: rdflib.Graph | None = None
    ) -> tuple[dict[str, UnitEntity], dict[str, QuantityKindEntity]]:
        """Loads local graph or provided graph and executes dynamic SPARQL extraction."""
        target_graph = graph if (graph is not None and len(graph) > 0) else self.load_local_graph()
        units = self.extract_units_from_graph(target_graph)
        kinds = self.extract_quantity_kinds_from_graph(target_graph)
        return units, kinds
