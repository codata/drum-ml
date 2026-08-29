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
   [./dataset/ -> train.jsonl, val.jsonl, test.jsonl, dpo_preferences.jsonl]

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
- **Function:** Leverages cloud frontier LLMs (Anthropic Claude 3.5 Sonnet, OpenAI GPT-4o via LiteLLM) or local models (MLX, LM Studio, Ollama) with SQLite semantic caching to generate diverse persona-conditioned user queries.

4. Agent 4: 4-Tier Automated Validation Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Module:** ``drum_ml.pipeline.validator`` & ``drum_ml.pipeline.dpo_miner``
- **Function:** Deterministically verifies LaTeX syntax, Pint/SymPy algebraic equivalence, Decimal precision, and SPARQL/Python code execution. Mines high-quality negative pairs for DPO.

5. Agent 5: Deduplication, Partitioning & Packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Module:** ``drum_ml.pipeline.exporter``
- **Function:** Executes MinHash LSH deduplication, entity-isolated stratified splitting (85% Train / 10% Val / 5% Test), and exports into standard OpenAI, ShareGPT, and DPO JSONL formats.
