"""CODATA Fundamental Constants Dynamic Fetcher and Parser with QUDT Crosswalks."""

import json
from pathlib import Path

import httpx
import rdflib

from drum_ml.models.entities import (
    ConstantCategory,
    DimensionVector,
    PhysicalConstantEntity,
)
from drum_ml.symbolic.precision import parse_codata_value_uncertainty


class CODATAClient:
    """Client for dynamically fetching and parsing CODATA fundamental constants (1969-2022)
    from RDF Turtle, JSON releases, or the CODATA DRUM API with QUDT crosswalks.
    """

    BASE_GITHUB_RAW = "https://raw.githubusercontent.com/codata/drum-constants/main"

    def __init__(
        self,
        local_dir: str = "./data/raw/codata",
        api_url: str = "https://api.codata.org/drum/constants",
        default_evaluation_year: int = 2022,
    ):
        self.local_dir = Path(local_dir)
        self.api_url = api_url
        self.default_evaluation_year = default_evaluation_year
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def download_latest_data(self) -> list[Path]:
        """Downloads latest CODATA constants Turtle files and JSON history from GitHub."""
        files = [
            ("dist/rdf/codata_constants.ttl", "codata-constants.ttl"),
            ("nist/2022/allascii_2022.json", "codata-history.json"),
        ]
        downloaded = []
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            for rel_source, target_name in files:
                url = f"{self.BASE_GITHUB_RAW}/{rel_source}"
                out_path = self.local_dir / target_name
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
        """Loads available local CODATA Turtle files into an rdflib Graph."""
        g = rdflib.Graph()
        for ttl_path in self.local_dir.glob("*.ttl"):
            try:
                g.parse(str(ttl_path), format="turtle")
            except Exception:
                pass
        return g

    def fetch_from_api(self, year: int | None = None) -> list[dict]:
        """Fetches constants dynamically from the CODATA DRUM REST API."""
        eval_year = year or self.default_evaluation_year
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.api_url}?year={eval_year}")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return []

    def extract_from_json(
        self, json_path: Path, year: int | None = None
    ) -> dict[str, PhysicalConstantEntity]:
        """Parses constants dynamically from codata-history.json or versioned JSON files."""
        constants: dict[str, PhysicalConstantEntity] = {}
        if not json_path.exists():
            return constants

        target_year = str(year or self.default_evaluation_year)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle list of constants or history dictionary
        items = data if isinstance(data, list) else data.get("constants", data.get(target_year, []))
        for item in items:
            name = item.get("name", item.get("label", "Unknown"))
            sym = item.get("symbol", "")
            lsym = item.get("latex_symbol", sym)
            raw_val = str(item.get("value", item.get("numeric_value", "0")))
            val, std_u = parse_codata_value_uncertainty(raw_val)
            rel_u = str(item.get("relative_uncertainty", item.get("rel_uncertainty", "")))
            u_sym = item.get("unit", item.get("unit_symbol", "1"))
            eval_year = int(item.get("year", item.get("defining_year", target_year)))
            is_exact = item.get("is_exact", False) or (std_u == "0" or not std_u)

            uri = item.get("uri", f"https://codata.org/constants/{eval_year}/{sym or name}")
            constants[uri] = PhysicalConstantEntity(
                uri=uri,
                name=name,
                symbol=sym,
                latex_symbol=lsym,
                category=ConstantCategory.EXACT_SI_DEFINING
                if is_exact
                else ConstantCategory.CODATA_RECOMMENDED,
                numeric_value=val,
                standard_uncertainty=std_u if not is_exact else "0",
                relative_uncertainty=rel_u if not is_exact else "0",
                unit_symbol=u_sym,
                dimension_vector=DimensionVector(),
                defining_year=eval_year,
                description=item.get("description", f"{name} ({eval_year} CODATA evaluation)"),
            )

        return constants

    def extract_from_graph(
        self, graph: rdflib.Graph, year: int | None = None
    ) -> dict[str, PhysicalConstantEntity]:
        """Parses constants from an RDF graph using SPARQL."""
        constants: dict[str, PhysicalConstantEntity] = {}
        eval_year = year or self.default_evaluation_year

        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX qudt: <http://qudt.org/schema/qudt/>
        PREFIX drum: <https://codata.org/constants/ontology/>

        SELECT DISTINCT ?constant ?name ?symbol ?latexSymbol ?value ?stdUncertainty ?relUncertainty ?unitSymbol ?isExact ?year
        WHERE {
          ?constant a ?type ;
                    rdfs:label ?name .
          FILTER (?type IN (qudt:PhysicalConstant, drum:PhysicalConstant))
          OPTIONAL { ?constant qudt:symbol ?symbol }
          OPTIONAL { ?constant drum:symbol ?symbol }
          OPTIONAL { ?constant qudt:latexSymbol ?latexSymbol }
          OPTIONAL { ?constant qudt:numericValue ?value }
          OPTIONAL { ?constant drum:numericValue ?value }
          OPTIONAL { ?constant qudt:standardUncertainty ?stdUncertainty }
          OPTIONAL { ?constant qudt:relativeUncertainty ?relUncertainty }
          OPTIONAL { ?constant qudt:unitSymbol ?unitSymbol }
          OPTIONAL { ?constant qudt:isExact ?isExact }
          OPTIONAL { ?constant drum:evaluationYear ?year }
        }
        """
        try:
            qres = graph.query(query)
            for row in qres:
                c_uri = str(row.constant)
                name = str(row.name)
                sym = str(row.symbol) if row.symbol else ""
                lsym = str(row.latexSymbol) if row.latexSymbol else sym
                raw_val = str(row.value) if row.value else "0"
                val, std_u = parse_codata_value_uncertainty(raw_val)
                if row.stdUncertainty:
                    std_u = str(row.stdUncertainty)
                rel_u = str(row.relUncertainty) if row.relUncertainty else ""
                u_sym = str(row.unitSymbol) if row.unitSymbol else "1"
                is_exact = bool(row.isExact) if row.isExact else (std_u == "0")
                row_year = int(row.year) if row.year else eval_year

                constants[c_uri] = PhysicalConstantEntity(
                    uri=c_uri,
                    name=name,
                    symbol=sym,
                    latex_symbol=lsym,
                    category=ConstantCategory.EXACT_SI_DEFINING
                    if is_exact
                    else ConstantCategory.CODATA_RECOMMENDED,
                    numeric_value=val,
                    standard_uncertainty=std_u if not is_exact else "0",
                    relative_uncertainty=rel_u if not is_exact else "0",
                    unit_symbol=u_sym,
                    dimension_vector=DimensionVector(),
                    defining_year=row_year,
                    description=f"{name} ({row_year} CODATA evaluation)",
                )
        except Exception:
            pass

        return constants

    def extract_canonical_constants(
        self,
        graph: rdflib.Graph | None = None,
        year: int | None = None,
    ) -> dict[str, PhysicalConstantEntity]:
        """Extracts canonical physical constants dynamically, prioritizing exact SI 2019 defining constants
        and CODATA recommended values for the requested evaluation year (default: 2022).
        """
        eval_year = year or self.default_evaluation_year
        constants: dict[str, PhysicalConstantEntity] = {}

        # 1. First check if RDF graph is provided or available locally
        if graph and len(graph) > 0:
            constants.update(self.extract_from_graph(graph, eval_year))

        local_graph = self.load_local_graph()
        if len(local_graph) > 0:
            constants.update(self.extract_from_graph(local_graph, eval_year))

        # 2. Check local JSON files (e.g. codata-history.json or codata-constants.json)
        for json_file in self.local_dir.glob("*.json"):
            constants.update(self.extract_from_json(json_file, eval_year))

        # 3. If empty, attempt to fetch from the CODATA DRUM REST API
        if not constants:
            api_data = self.fetch_from_api(eval_year)
            if api_data:
                # Save API data locally for caching
                cached_json = self.local_dir / f"codata_{eval_year}.json"
                with open(cached_json, "w", encoding="utf-8") as f:
                    json.dump(api_data, f, indent=2)
                constants.update(self.extract_from_json(cached_json, eval_year))

        return constants
