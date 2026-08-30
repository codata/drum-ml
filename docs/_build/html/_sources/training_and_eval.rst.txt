Training & Evaluation Guide
===========================

Local Storage & Git Exclusion
-----------------------------

Generated datasets and raw semantic caches are saved locally and strictly ignored by Git via ``.gitignore``:

- ``dataset/train.jsonl``: 85% SFT training split.
- ``dataset/val.jsonl``: 10% validation split.
- ``dataset/test.jsonl``: 5% held-out test split for M-Eval.
- ``dataset/dpo_preferences.jsonl``: Direct Preference Optimization pairs.
- ``data/raw/``: Local cache of downloaded BIPM, CODATA, and QUDT ontologies.
- ``data/cache/llm_cache.sqlite``: SQLite semantic cache for LLM queries.

Training Recipes
----------------

1. Local Fine-Tuning on Mac (Apple MLX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using ``mlx-lm`` for LoRA SFT and DPO alignment on Apple Silicon:

.. code-block:: bash

   # 1. Supervised Fine-Tuning (SFT)
   python -m mlx_lm.lora \
       --model mlx-community/Qwen2.5-14B-Instruct-4bit \
       --data ./dataset/ \
       --train \
       --batch-size 4 \
       --iters 1000 \
       --adapter-path ./adapters/drum-sft-qwen14b

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

   # Load local DRUM-ML splits
   train_ds = load_dataset("json", data_files={"train": "./dataset/train.jsonl", "validation": "./dataset/val.jsonl"})

   # SFT Fine-Tuning
   trainer = SFTTrainer(
       model="meta-llama/Llama-3.1-8B-Instruct",
       train_dataset=train_ds["train"],
       eval_dataset=train_ds["validation"],
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

   # Generate 50 samples per task (total 250+ balanced questions)
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

DRUM-ML includes built-in commands to publish generated dataset splits (``train.jsonl``, ``val.jsonl``, ``test.jsonl``), DPO preference pairs (``dpo_preferences.jsonl``), and dataset cards directly to Hugging Face:

.. code-block:: bash

   # 1. Login to Hugging Face
   huggingface-cli login

   # 2. Publish dataset splits directly via the DRUM-ML CLI
   drum-ml publish-hf --repo-id codata/drum-metrology-instruct --dataset-dir ./dataset

   # Optional: Publish as a private repository
   drum-ml publish-hf --repo-id codata/drum-metrology-instruct --private

   # Alternatively, upload via official huggingface-cli
   huggingface-cli upload codata/drum-metrology-instruct ./dataset/ . --repo-type=dataset

