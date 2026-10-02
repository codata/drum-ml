Training & Evaluation Guide
===========================

Local Storage & Partitioned File Hierarchy
------------------------------------------

Generated datasets and raw semantic caches are saved in ``./dataset/`` (and ignored by Git via ``.gitignore``):

.. code-block:: text

   dataset/
   ├── train.jsonl                          # Master SFT training split (85%)
   ├── val.jsonl                            # Master validation split (10%)
   ├── test.jsonl                           # Master held-out test split (5%)
   ├── dpo_preferences.jsonl                # Direct Preference Optimization (DPO) pairs
   ├── manifest.json                        # Index with file counts, byte sizes, and SHA-256 hashes
   ├── dataset_viewer.html                  # Standalone Interactive HTML Dataset Browser
   │
   ├── by_persona/                          # 35+ Individual Scientific Union & Domain Subsets
   │   ├── academic_metrologist.jsonl       # Consolidated persona dataset
   │   ├── academic_metrologist_train.jsonl # Training split for Academic Metrologist
   │   ├── physics_student.jsonl
   │   ├── physics_student_train.jsonl
   │   ├── iupap_physicist_train.jsonl
   │   ├── iupac_chemist_train.jsonl
   │   └── ...
   │
   ├── by_archetype/                        # 6 Pedagogical Metrology Archetype Subsets
   │   ├── direct_identification.jsonl
   │   ├── dimensional_decomposition.jsonl
   │   ├── conversion_scaling.jsonl
   │   ├── error_detection.jsonl
   │   ├── semantic_tool_use.jsonl
   │   └── metrological_uncertainty.jsonl
   │
   ├── by_category/                         # Physical Entity Category Subsets
   │   ├── units.jsonl                      # All unit definitions & conversions
   │   └── constants.jsonl                  # Physical constants & uncertainties
   │
   └── formats/                             # Alternative formats (ShareGPT, Alpaca)
       └── sharegpt/
           ├── train.jsonl
           └── val.jsonl

Dataset Partitioning CLI
------------------------

DRUM-ML includes a high-performance standalone partitioner that breaks down large dataset files into per-persona, per-archetype, and per-category subsets without re-running LLM generation:

.. code-block:: bash

   # Partition existing dataset directory into grouped subsets
   drum-ml partition --dataset-dir ./dataset --viewer

Interactive HTML Dataset Browser
--------------------------------

DRUM-ML includes a standalone, zero-dependency interactive HTML dataset browser designed for reviewing, searching, and auditing fine-tuning and benchmark datasets:

.. code-block:: bash

   # Launch the Interactive Dataset Browser in the default web browser
   drum-ml view-dataset --dataset-dir ./dataset

   # Or specify sample limit and output file:
   drum-ml view-dataset --dataset-dir ./dataset --max-samples 5000 --output-html ./dataset/dataset_viewer.html

Key Browser Features:
- **Instant Search & Real-Time Highlighting:** Filter across prompts, assistant answers, ontology URIs, and equations.
- **Multi-Dimensional Filters:** Filter simultaneously by Split (``Train``, ``Val``, ``Test``, ``DPO``), Persona (35+ scientific unions), Archetype (6 archetypes), and Category.
- **Multi-Format View Switcher:** Toggle between Rendered Markdown + KaTeX LaTeX math, OpenAI Chat messages, ShareGPT conversation turns, and Raw JSON.
- **DPO Preference Arena:** Side-by-side comparison of Prompt, Chosen (Passed), and Rejected (Failed) responses with exact diagnostic reason badges.
- **Persona Matrix & Archetype Hub:** Visual overview of all scientific unions and pedagogical archetypes with 1-click filtering.
- **Local File Drag & Drop:** Drop any local ``.jsonl`` or ``.json`` file directly into the browser to explore it instantly.

Training Recipes
----------------

1. Local Fine-Tuning on Mac (Apple MLX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using ``mlx-lm`` for LoRA SFT and DPO alignment on Apple Silicon:

.. code-block:: bash

   # 1. Supervised Fine-Tuning (SFT) on master split
   python -m mlx_lm.lora \
       --model mlx-community/Qwen2.5-14B-Instruct-4bit \
       --data ./dataset/ \
       --train \
       --batch-size 4 \
       --iters 1000 \
       --adapter-path ./adapters/drum-sft-qwen14b

   # Or fine-tune on a specific persona subset (e.g. IUPAP Physicist):
   python -m mlx_lm.lora \
       --model mlx-community/Qwen2.5-14B-Instruct-4bit \
       --data ./dataset/by_persona/ \
       --train \
       --adapter-path ./adapters/drum-sft-iupap

   # 2. Direct Preference Optimization (DPO)
   python -m mlx_lm.dpo \
       --model ./adapters/drum-sft-qwen14b \
       --data ./dataset/ \
       --train \
       --adapter-path ./adapters/drum-dpo-qwen14b

2. GPU Cluster Training (Hugging Face TRL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using Hugging Face ``TRL`` (``SFTTrainer`` & ``DPOTrainer``):

.. code-block:: python

   from datasets import load_dataset
   from trl import SFTTrainer, DPOTrainer
   from transformers import TrainingArguments

   # Load specific persona subset or full master split
   train_ds = load_dataset("json", data_files={"train": "./dataset/by_persona/academic_metrologist_train.jsonl"})

   # SFT Fine-Tuning
   trainer = SFTTrainer(
       model="meta-llama/Llama-3.1-8B-Instruct",
       train_dataset=train_ds["train"],
       dataset_text_field="messages",
       max_seq_length=2048,
       args=TrainingArguments(output_dir="./drum-llama3-sft", per_device_train_batch_size=4, num_train_epochs=3),
   )
   trainer.train()

   # DPO Alignment
   dpo_ds = load_dataset("json", data_files="./dataset/dpo_preferences.jsonl")
   dpo_trainer = DPOTrainer(
       model="./drum-llama3-sft",
       train_dataset=dpo_ds["train"],
       args=TrainingArguments(output_dir="./drum-llama3-dpo", per_device_train_batch_size=2),
   )
   dpo_trainer.train()

3. Cloud Fine-Tuning (OpenAI API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openai api fine_tuning.jobs.create \
     -t ./dataset/train.jsonl \
     -v ./dataset/val.jsonl \
     -m gpt-4o-mini-2024-07-18

DRUM Metrology Benchmark (M-Eval) Evaluation
---------------------------------------------

DRUM-ML includes a dedicated, standardized benchmark suite evaluating LLMs across **6 Core Sub-Disciplines**:
1. **Fundamental Physical Constants & SI 2019** (``constants``)
2. **Dimensional Decomposition & Base SI** (``dimensions``)
3. **Unit Conversions & Affine Transformations** (``conversions``)
4. **Error Detection & Dimensional Homogeneity** (``homogeneity``)
5. **SI Typography & Metrological Conventions** (``conventions``)
6. **Metrological Uncertainty (GUM) & Sig-Figs** (``uncertainty``)

1. Synthesize Benchmark Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate 50 samples per task (total 300 balanced questions)
   drum-ml build-benchmark \
       --entities-file ./data/entities.json \
       --output-dir ./dataset/benchmark \
       --samples-per-task 50

2. Evaluate Models & Generate Scorecards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Evaluate baseline model
   drum-ml evaluate \
       --benchmark-file ./dataset/benchmark/drum_benchmark_mcq.jsonl \
       --model-endpoint http://localhost:1234/v1 \
       --model-name qwen2.5-14b-base \
       --output ./dataset/benchmark/base_eval_report.json

   # Evaluate fine-tuned model (measure metrological gain)
   drum-ml evaluate \
       --benchmark-file ./dataset/benchmark/drum_benchmark_mcq.jsonl \
       --model-endpoint http://localhost:1234/v1 \
       --model-name qwen2.5-14b-drum-finetuned \
       --output ./dataset/benchmark/finetuned_eval_report.json

3. Standardized ``lm-evaluation-harness`` Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run EleutherAI lm-eval against the complete DRUM benchmark
   lm_eval --model hf \
           --model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
           --include_path ./tasks \
           --tasks drum_benchmark \
           --batch_size auto \
           --output_path ./dataset/benchmark/qwen_eval_results

For full architectural details, data schemas, and distractor rationales, see :doc:`benchmark`.

Publishing to Hugging Face Hub
------------------------------

DRUM-ML includes built-in commands to publish generated dataset splits, grouped subsets, DPO preference pairs, and dataset cards directly to Hugging Face:

.. code-block:: bash

   # 1. Login to Hugging Face
   huggingface-cli login

   # 2. Publish dataset splits directly via the DRUM-ML CLI
   drum-ml publish-hf --repo-id codata/drum-metrology-instruct --dataset-dir ./dataset

   # Optional: Publish as a private repository
   drum-ml publish-hf --repo-id codata/drum-metrology-instruct --private

   # Alternatively, upload via official huggingface-cli
   huggingface-cli upload codata/drum-metrology-instruct ./dataset/ . --repo-type=dataset
