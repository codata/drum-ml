Multi-Agent Pipeline Architecture
=================================

The DRUM-ML pipeline is organized into five specialized, autonomous agents executing in a sequential directed acyclic graph (DAG):

.. code-block:: text

   [Master Ontologies: BIPM / CODATA / QUDT]
                     │
                     ▼
   [Agent 1: Ingestion & Extraction (Agent-SPARQL)]
                     │
                     ▼ (entities.json)
   [Agent 2: Pedagogical Archetype Generator (Agent-Archetype)]
                     │
                     ▼ (scaffolds.json)
   [Agent 3: Linguistic Diversity & Personas (Agent-Augmenter)]
                     │
                     ▼ (augmented.json)
   [Agent 4: Metrology Validation Gate (Agent-Validator)]
          │                                  │
          ▼ [Pass]                           ▼ [Fail: Plausible Errors]
   (validated.json)                   [DPO Negative Miner]
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
   [Agent 5: Stratified Split & Packaging (Agent-Exporter)]
                           │
                           ▼
   [./dataset/]
       ├── train.jsonl, val.jsonl, test.jsonl, dpo_preferences.jsonl
       ├── by_persona/          (35+ scientific union & domain subsets)
       ├── by_archetype/        (6 pedagogical metrology archetypes)
       ├── by_category/         (units vs constants vs quantity kinds)
       ├── manifest.json        (SHA-256 checksums, token geometry, counts)
       └── dataset_viewer.html  (Standalone Interactive HTML Dataset Browser)

Agent Details
-------------

1. Agent 1: SPARQL Ingestion & Canonical Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Module:** ``drum_ml.pipeline.extractor`` & ``drum_ml.data_sources``
- **Function:** Downloads or reads local Turtle ontologies from BIPM, CODATA DRUM, and QUDT 2.1.
- **Output:** Serializes canonical entities into structured Pydantic models (``CanonicalEntityStore``).

2. Agent 2: Pedagogical Archetype Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Module:** ``drum_ml.pipeline.scaffolder``
- **Function:** Deterministically synthesizes ground-truth instruction-response scaffolds across 6 core metrological archetypes.

3. Agent 3: Linguistic Diversity & Persona Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Module:** ``drum_ml.pipeline.augmenter``
- **Function:** Leverages cloud frontier LLMs (Anthropic Claude 3.7 Sonnet, OpenAI GPT-4.5/GPT-4o, Google Gemini via LiteLLM) or local models (Nemotron, Qwen, Ollama, LM Studio) with SQLite semantic caching to generate diverse persona-conditioned user queries across 35+ International Scientific Unions.

4. Agent 4: 4-Tier Automated Validation Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Module:** ``drum_ml.pipeline.validator`` & ``drum_ml.pipeline.dpo_miner``
- **Function:** Deterministically verifies LaTeX syntax, Pint/SymPy algebraic equivalence, Decimal precision, and SPARQL/Python code execution. Mines high-quality negative pairs for Direct Preference Optimization (DPO).

5. Agent 5: Deduplication, Grouped Partitioning & Packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Module:** ``drum_ml.pipeline.exporter`` & ``drum_ml.dataset_viewer``
- **Function:** 
  - Executes MinHash LSH fuzzy deduplication and exact SHA-256 hash pruning.
  - Applies entity-isolated stratified splitting (85% Train / 10% Val / 5% Test).
  - Partitions dataset into organized modular files (``by_persona/``, ``by_archetype/``, ``by_category/``).
  - Exports standard OpenAI Chat, ShareGPT, and DPO JSONL formats.
  - Generates ``manifest.json`` with file sizes, counts, and SHA-256 integrity checksums.
  - Compiles the standalone interactive HTML dataset browser (``dataset_viewer.html``).
