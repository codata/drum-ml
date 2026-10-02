"""Agent 1: SPARQL Ingestion and Multi-Graph Canonical Extractor."""

from pathlib import Path

from drum_ml.data_sources.bipm_client import BIPMClient
from drum_ml.data_sources.codata_client import CODATAClient
from drum_ml.data_sources.qudt_fetcher import QUDTFetcher
from drum_ml.models.entities import CanonicalEntityStore


class MetrologyExtractor:
    """Agent 1: Ingests and harmonizes BIPM SI, CODATA DRUM, and QUDT 2.1 entities."""

    def __init__(
        self,
        bipm_dir: str = "./data/raw/bipm",
        codata_dir: str = "./data/raw/codata",
        qudt_dir: str = "./data/raw/qudt",
        auto_download_if_empty: bool = True,
    ):
        self.bipm_client = BIPMClient(local_dir=bipm_dir)
        self.codata_client = CODATAClient(local_dir=codata_dir)
        self.qudt_fetcher = QUDTFetcher(local_dir=qudt_dir)
        self.auto_download = auto_download_if_empty

    def _ensure_sources_present(self) -> None:
        """Auto-downloads master ontologies if local cache directory is empty."""
        if not any(self.bipm_client.local_dir.glob("*.ttl")):
            self.bipm_client.download_latest_ontology()

        if not any(self.codata_client.local_dir.glob("*.ttl")) and not any(
            self.codata_client.local_dir.glob("*.json")
        ):
            self.codata_client.download_latest_data()

        if not any(self.qudt_fetcher.local_dir.glob("*.ttl")):
            self.qudt_fetcher.download_latest_vocabularies()

    def extract_all(self) -> CanonicalEntityStore:
        """Executes multi-graph extraction with strict precedence hierarchy:
        Tier 1 (BIPM SI) > Tier 2 (CODATA DRUM Constants) > Tier 3 (QUDT 2.1).
        """
        if self.auto_download:
            self._ensure_sources_present()

        store = CanonicalEntityStore()

        # 1. Extract QUDT baseline (Tier 3)
        qudt_units, qudt_kinds = self.qudt_fetcher.extract_all()
        store.units.update(qudt_units)
        store.quantity_kinds.update(qudt_kinds)

        # 2. Extract and overwrite with BIPM SI Base units & 2019 Defining Constants (Tier 1 Precedence)
        bipm_units, bipm_constants = self.bipm_client.extract_all()
        store.units.update(bipm_units)
        store.constants.update(bipm_constants)

        # 3. Extract CODATA Fundamental Constants with QUDT crosswalks (Tier 2 Precedence)
        codata_constants = self.codata_client.extract_canonical_constants()
        store.constants.update(codata_constants)

        # Filter out placeholder/dummy entities (e.g. QUDT UNKNOWN unit placeholder)
        store.units = {
            uri: u
            for uri, u in store.units.items()
            if not (uri.endswith("/UNKNOWN") or (u.symbol == "Unknown" and u.label == "Unknown"))
        }
        store.quantity_kinds = {
            uri: qk
            for uri, qk in store.quantity_kinds.items()
            if not (uri.endswith("/Unknown") or qk.label == "Unknown")
        }

        return store

    def save_to_json(
        self, store: CanonicalEntityStore, output_path: str = "./data/entities.json"
    ) -> Path:
        """Serializes canonical entity store to JSON with UTF-8 encoding."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(store.model_dump_json(indent=2))
        return out
