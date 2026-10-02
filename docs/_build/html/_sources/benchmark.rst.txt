DRUM Metrology Benchmark (M-Eval) Suite
========================================

The **DRUM Metrology Benchmark** (also designated **M-Eval**) is an authoritative, standardized evaluation testbed designed to measure the knowledge, reasoning, and precision of Large Language Models (LLMs) across physical quantities, units, and fundamental constants.

Developed under the **CODATA DRUM Working Group**, the benchmark enforces strict alignment with the international ground truth hierarchy:

1. **Tier 1 (Ultimate Ground Truth):** BIPM SI Digital Framework (Official SI Brochure, 9th Edition, 2019 Revision).
2. **Tier 2 (Official Physical Constants):** CODATA Fundamental Physical Constants (2018/2022 NIST/CODATA Re-evaluations).
3. **Tier 3 (Broad Semantic Graph):** QUDT 2.1 (Quantities, Units, Dimensions, and Types) & UCUM codes.

1. The 6 Core Metrological Tasks
--------------------------------

The benchmark evaluates LLMs across six fundamental metrological sub-disciplines:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Benchmark Task
     - Scope & Concepts Tested
     - Ground Truth Authority & Validation
   * - **1. Fundamental Constants & SI 2019** (``constants``)
     - Exact defined SI constants (:math:`c, h, e, k, N_{\text{A}}, \Delta\nu_{\text{Cs}}, K_{\text{cd}}`) with :math:`u=0` vs. experimental CODATA constants (:math:`G, \alpha, m_e`), standard uncertainties, relative uncertainties (:math:`u_r`), and historical re-evaluations.
     - BIPM SI Core Ontology & CODATA DRUM Constants API (2018/2022).
   * - **2. Dimensional Decomposition & Base SI** (``dimensions``)
     - Derivation and decomposition of derived physical units into the 7 ISQ base dimensions :math:`[L, M, T, I, \Theta, N, J]` and coherent SI base units (e.g., :math:`\text{Tesla} = \text{kg}\cdot\text{s}^{-2}\cdot\text{A}^{-1}`).
     - BIPM SI Digital Framework & SymPy symbolic dimension vector arithmetic.
   * - **3. Unit Conversions & Affine Transformations** (``conversions``)
     - Multiplicative scaling, non-SI accepted units, compound rates (:math:`\text{kWh} \to \text{J}`), and affine temperature conversions (:math:`^\circ\text{C} \leftrightarrow \text{K}`, :math:`^\circ\text{F} \leftrightarrow \text{K}`) distinguishing absolute points from temperature intervals (:math:`\Delta T`).
     - QUDT 2.1 Conversion Multipliers & Pint unit algebra engine.
   * - **4. Error Detection & Dimensional Homogeneity** (``homogeneity``)
     - Detection of dimensional inhomogeneity (:math:`15\,\text{kg} + 4.2\,\text{m}`), illegal compound metric prefixes (:math:`\text{k}\mu\text{F}`), and semantic quantity-kind confusion (e.g. Torque :math:`\text{N}\cdot\text{m}` vs. Energy :math:`\text{J}`).
     - SymPy Physics Dimensional Homogeneity Engine & VIM3 rules.
   * - **5. SI Typography & Metrological Conventions** (``conventions``)
     - Compliance with BIPM 9th Edition style rules: non-breaking space between value and unit symbol (:math:`25.4\,\text{mm}`), lowercase unit names vs. capitalized symbols for person-named units (:math:`\text{newton} \leftrightarrow \text{N}`), and Roman vs. italic typography.
     - BIPM SI Brochure (9th Edition, 2019) & ISO 80000-1.
   * - **6. Metrological Uncertainty (GUM) & Sig-Figs** (``uncertainty``)
     - Combined standard uncertainty propagation (:math:`u_c`), expanded uncertainty (:math:`U = k \cdot u_c`) with coverage factor :math:`k=2`, relative standard uncertainty (:math:`u_r`), and strict significant figures rules.
     - JCGM 100:2008 (Guide to the Expression of Uncertainty in Measurement - GUM).

---

2. Dual-Format Testing Architecture
-----------------------------------

The benchmark provides two complementary evaluation tracks to support both zero-ambiguity deterministic evaluation and generative symbolic verification:

Track A: Deterministic 4-Option Multiple Choice (MCQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Format:** MMLU / Big-Bench style 4-choice questions (options ``A``, ``B``, ``C``, ``D``).
- **Metrological Distractors:** Distractors are deliberately engineered based on common physical pitfalls:
  - *Dimensional Sign Errors:* Flipping time or length exponent signs (:math:`T^{-2} \leftrightarrow T^{2}`).
  - *Affine vs. Linear Offsets:* Omitting or misapplying the :math:`+273.15\,\text{K}` offset on temperature intervals vs. absolute values.
  - *SI Redefinition Errors:* Presenting pre-2019 experimental uncertainties for the 7 exact defining constants.
  - *Prefix Stacking:* Presenting invalid compound metric prefixes (e.g., :math:`\mu\mu\text{F}` instead of :math:`\text{pF}`).
- **Scoring:** Exact matching of the predicted option letter (extracted via robust regex parsing).

Track B: Open-Ended Generative & Symbolic Derivations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Format:** Free-form questions asking models to derive base SI equations, calculate conversions, or compute uncertainty budgets.
- **Verification Engine:**
  - **LaTeX AST Normalization:** Cleans and normalizes LaTeX math expressions.
  - **Pint Algebraic Equivalence:** Converts candidate expressions into base SI dimensions to verify exact equivalence regardless of formatting.
  - **Arbitrary-Precision Decimal Engine:** Verifies exact constant values down to full defining precision without float truncation.

---

3. CLI Usage & Workflow
-----------------------

Synthesizing the Benchmark Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build the standardized benchmark splits directly from the ingested canonical entities:

.. code-block:: bash

   # Generate 50 samples per task (total 250+ balanced questions)
   drum-ml build-benchmark \
       --entities-file ./data/entities.json \
       --output-dir ./dataset/benchmark \
       --samples-per-task 50 \
       --seed 42

This produces three artifacts in ``./dataset/benchmark/``:
- ``drum_benchmark_mcq.jsonl``: Track A Multiple-Choice test suite.
- ``drum_benchmark_open.jsonl``: Track B Free-Form Symbolic test suite.
- ``drum_benchmark_all.jsonl``: Complete combined benchmark suite.

Running Benchmark Evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate a model against the benchmark:

.. code-block:: bash

   # Evaluate a model endpoint or baseline with live progress
   drum-ml evaluate \
       --benchmark-file ./dataset/benchmark/drum_benchmark_mcq.jsonl \
       --model-name "Qwen2.5-7B-Instruct" \
       --model-endpoint "http://localhost:1234/v1" \
       --output ./dataset/benchmark/benchmark_report.json

Interactive Benchmark Browser Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DRUM-ML provides a standalone, interactive HTML browser dashboard to visually explore, inspect, filter, and audit benchmark questions across all 6 tasks:

.. code-block:: bash

   # Launch interactive benchmark explorer in default web browser
   drum-ml view-benchmark --benchmark-file ./dataset/benchmark/drum_benchmark_all.jsonl

   # Or export HTML file to a custom path:
   drum-ml view-benchmark --benchmark-file ./dataset/benchmark/drum_benchmark_all.jsonl --output-html ./dataset/benchmark/benchmark_viewer.html

Terminal Output Scorecard
~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluation renders a formatted Rich table with stratified category scores:

.. code-block:: text

          DRUM Metrology Benchmark Scorecard: Qwen2.5-7B-Instruct        
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
   ┃ Category / Sub-Discipline              ┃ Samples ┃ Passed ┃ Accuracy ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
   │ 1. Fundamental Constants & SI 2019     │      50 │     48 │    96.0% │
   │ 2. Dimensional Decomposition & Base SI │      50 │     43 │    86.0% │
   │ 3. Unit Conversions & Affine Offsets   │      50 │     47 │    94.0% │
   │ 4. Error Detection & Homogeneity       │      50 │     39 │    78.0% │
   │ 5. SI Typography & Metrological Rules  │      50 │     41 │    82.0% │
   │ 6. Metrological Uncertainty (GUM)      │      50 │     33 │    66.0% │
   ├────────────────────────────────────────┼─────────┼────────┼──────────┤
   │ OVERALL DRUM BENCHMARK SCORE           │     300 │    251 │   83.67% │
   └────────────────────────────────────────┴─────────┴────────┴──────────┘

---

4. Programmatic Python API
--------------------------

Synthesizing Benchmark Samples in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from drum_ml.models.entities import CanonicalEntityStore
   from drum_ml.benchmark.generator import DRUMBenchmarkGenerator

   # Load canonical RDF entities
   with open("./data/entities.json", "r", encoding="utf-8") as f:
       store = CanonicalEntityStore.model_validate_json(f.read())

   # Initialize generator
   generator = DRUMBenchmarkGenerator(entities=store, seed=42)

   # Generate full 6-task benchmark
   samples = generator.generate_all(samples_per_task=50)
   print(f"Generated {len(samples)} benchmark samples.")

Grading and Computing Scorecards Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from drum_ml.pipeline.evaluator import MEvalBenchmark

   evaluator = MEvalBenchmark(benchmark_file="./dataset/benchmark/drum_benchmark_mcq.jsonl")
   records = evaluator.load_benchmark_records()

   results = []
   for record in records:
       # Replace with actual model prediction
       predicted = "The correct answer is (B)"
       grade = evaluator.grade_response(record, predicted)
       results.append({
           "id": record["id"],
           "task": record["task"],
           "format": record["format"],
           "difficulty": record.get("difficulty", "intermediate"),
           "grade": grade,
       })

   # Compute full scorecard
   scorecard = evaluator.compute_scorecard("my-model", results)
   print(f"Overall Score: {scorecard.overall_accuracy_pct:.2f}%")
   evaluator.print_scorecard(scorecard)

---

5. Data Schema Specifications
-----------------------------

Benchmark Sample Record Schema (JSONL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "id": "drum_bench_const_mcq_001",
     "task": "constants",
     "format": "mcq",
     "difficulty": "introductory",
     "question": "Under the 2019 SI redefinition, what is the exact defined numerical value of the Planck constant (h)?",
     "options": [
       {
         "key": "A",
         "text": "6.62607015e-34 J s (Exact, standard uncertainty u = 0)",
         "is_correct": true,
         "distractor_rationale": "Ground truth"
       },
       {
         "key": "B",
         "text": "6.62606957e-34 J s (Outdated pre-2019 value)",
         "is_correct": false,
         "distractor_rationale": "Outdated pre-2019 value"
       },
       {
         "key": "C",
         "text": "6.62607015e-34 J s (Recommended value with u = 4.5e-9)",
         "is_correct": false,
         "distractor_rationale": "Incorrect claim of non-zero uncertainty budget"
       },
       {
         "key": "D",
         "text": "6.62607015e-34 J s (Fixed prior to 1983)",
         "is_correct": false,
         "distractor_rationale": "Incorrect historical date"
       }
     ],
     "correct_option_key": "A",
     "ground_truth_answer": "6.62607015e-34 J s (Exact, standard uncertainty u = 0)",
     "entity_uri": "http://qudt.org/vocab/constant/PlanckConstant",
     "explanation": "In the 2019 revision of the SI, the Planck constant (h) was given the exact value 6.62607015e-34 J s by definition, fixing its standard uncertainty to exactly zero.",
      "metadata": {
        "constant_symbol": "h",
        "category": "exact_si_defining"
      }
    }

---

6. EleutherAI ``lm-evaluation-harness`` Integration
----------------------------------------------------

The DRUM benchmark provides full native support for `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_, allowing automated evaluation of open-weights models and frontier APIs using standardized log-likelihood multi-choice grading.

The task configurations are organized under ``tasks/drum_benchmark/``:

- ``drum_benchmark.yaml``: Master group task aggregating all 6 subtasks.
- ``drum_constants.yaml``: Task 1 (Constants & SI 2019).
- ``drum_dimensions.yaml``: Task 2 (Dimensional Decomposition).
- ``drum_conversions.yaml``: Task 3 (Unit Conversions).
- ``drum_homogeneity.yaml``: Task 4 (Homogeneity & Error Detection).
- ``drum_conventions.yaml``: Task 5 (SI Typography & Rules).
- ``drum_uncertainty.yaml``: Task 6 (Uncertainty & GUM).

Running with ``lm_eval`` CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Evaluate an open Hugging Face model across the full DRUM benchmark suite
   lm_eval --model hf \
           --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
           --include_path ./tasks \
           --tasks drum_benchmark \
           --batch_size auto \
           --output_path ./dataset/benchmark/qwen_eval_results

   # Evaluate a single subtask (e.g. Fundamental Physical Constants)
   lm_eval --model hf \
           --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct \
           --include_path ./tasks \
           --tasks drum_constants \
           --batch_size auto
