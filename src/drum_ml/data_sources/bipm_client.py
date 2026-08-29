"""BIPM SI Digital Framework Dynamic Downloader and SPARQL Parser."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httpx
import rdflib
from drum_ml.models.entities import (
    ConstantCategory,
    DimensionVector,
    PhysicalConstantEntity,
    UnitEntity,
)
from drum_ml.symbolic.dimensions import parse_qudt_dimension_string


class BIPMClient:
    """Client for dynamically downloading and parsing the official BIPM SI Digital Framework ontologies via SPARQL."""

    BASE_GITHUB_RAW = "https://raw.githubusercontent.com/TheBIPM/SI_Digital_Framework/main/knowledge_graphs/SI_Reference_Point"

    def __init__(
        self,
        local_dir: str = "./data/raw/bipm",
        base_api_url: str = "https://si-digital-framework.org/SI",
    ):
        self.local_dir = Path(local_dir)
        self.base_api_url = base_api_url
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def download_latest_ontology(self) -> List[Path]:
        """Downloads the latest official BIPM SI Digital Framework TTL files from GitHub."""
        files = ["si.ttl", "units.ttl", "constants.ttl", "prefixes.ttl"]
        downloaded = []
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            for fname in files:
                url = f"{self.BASE_GITHUB_RAW}/{fname}"
                out_path = self.local_dir / fname
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        with open(out_path, "wb") as f:
                            f.write(resp.content)
                        downloaded.append(out_path)
                except Exception:
                    pass
        return downloaded

    def load_local_graph(self) -> rdflib.Graph:
        """Loads all available local BIPM Turtle files into an rdflib Graph."""
        g = rdflib.Graph()
        for ttl_path in self.local_dir.glob("*.ttl"):
            try:
                g.parse(str(ttl_path), format="turtle")
            except Exception:
                pass
        return g

    def extract_base_units_from_graph(self, graph: rdflib.Graph) -> Dict[str, UnitEntity]:
        """Dynamically extracts SI base units from BIPM SI ontology using SPARQL."""
        units: Dict[str, UnitEntity] = {}
        query = """
        PREFIX si: <https://si-digital-framework.org/SI/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?u ?label ?symbol ?dim
        WHERE {
          ?u a ?type ;
             rdfs:label ?label .
          FILTER (?type IN (si:BaseUnit, si:Unit, <https://si-digital-framework.org/SI/ontology/BaseUnit>))
          OPTIONAL { ?u si:hasSymbol ?symbol }
          OPTIONAL { ?u si:hasDimension ?dim }
        }
        """
        for row in graph.query(query):
            uri = str(row.u)
            sym = str(row.symbol) if row.symbol else str(row.label)
            dim_str = str(row.dim).split("/")[-1] if row.dim else ""
            dim_vec = parse_qudt_dimension_string(dim_str)

            units[uri] = UnitEntity(
                uri=uri,
                symbol=sym,
                label=str(row.label),
                is_si_base=True,
                is_si_derived=False,
                is_coherent=True,
                dimension_vector=dim_vec,
                ucum_code=sym,
            )

        return units

    def extract_defining_constants_from_graph(self, graph: rdflib.Graph) -> Dict[str, PhysicalConstantEntity]:
        """Dynamically extracts SI defining constants from BIPM SI ontology using SPARQL."""
        constants: Dict[str, PhysicalConstantEntity] = {}
        query = """
        PREFIX si: <https://si-digital-framework.org/SI/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?c ?label ?symbol ?value ?unit ?dim
        WHERE {
          ?c a ?type ;
             rdfs:label ?label .
          FILTER (?type IN (si:DefiningConstant, <https://si-digital-framework.org/SI/ontology/DefiningConstant>))
          OPTIONAL { ?c si:hasSymbol ?symbol }
          OPTIONAL { ?c si:hasNumericalValue ?value }
          OPTIONAL { ?c si:hasUnit ?unit }
          OPTIONAL { ?c si:hasDimension ?dim }
        }
        """
        for row in graph.query(query):
            uri = str(row.c)
            sym = str(row.symbol) if row.symbol else str(row.label)
            val = str(row.value) if row.value else "0"
            u_sym = str(row.unit) if row.unit else "1"
            dim_str = str(row.dim).split("/")[-1] if row.dim else ""
            dim_vec = parse_qudt_dimension_string(dim_str)

            constants[uri] = PhysicalConstantEntity(
                uri=uri,
                name=str(row.label),
                symbol=sym,
                latex_symbol=sym,
                category=ConstantCategory.EXACT_SI_DEFINING,
                numeric_value=val,
                standard_uncertainty="0",
                relative_uncertainty="0",
                unit_symbol=u_sym,
                dimension_vector=dim_vec,
                defining_year=2019,
                description=f"BIPM SI Defining Constant: {row.label}",
            )

        return constants

    def extract_all(self, graph: Optional[rdflib.Graph] = None) -> Tuple[Dict[str, UnitEntity], Dict[str, PhysicalConstantEntity]]:
        """Extracts SI base units and defining constants dynamically from local or provided graph."""
        target_graph = graph if (graph is not None and len(graph) > 0) else self.load_local_graph()
        units = self.extract_base_units_from_graph(target_graph)
        constants = self.extract_defining_constants_from_graph(target_graph)
        return units, constants
