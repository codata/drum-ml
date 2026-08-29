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

M-Eval Benchmark Evaluation
---------------------------

Evaluate base vs. fine-tuned models on the held-out benchmark:

.. code-block:: bash

   # Evaluate baseline model
   drum-ml evaluate \
       --benchmark-file ./dataset/test.jsonl \
       --model-endpoint http://localhost:1234/v1 \
       --model-name qwen2.5-14b-base \
       --output ./dataset/base_eval_report.json

   # Evaluate fine-tuned model
   drum-ml evaluate \
       --benchmark-file ./dataset/test.jsonl \
       --model-endpoint http://localhost:1234/v1 \
       --model-name qwen2.5-14b-drum-finetuned \
       --output ./dataset/finetuned_eval_report.json

Evaluation Metrics
~~~~~~~~~~~~~~~~~~
- **Exact Accuracy Score (EAS):** Percentage of exact constants and symbols reproduced without drift.
- **Dimensional Homogeneity Score (DHS):** Accuracy on detecting invalid equations and dimensional violations via SymPy/Pint.
- **Conversion Error Rate (CER):** Mean relative error on multi-step conversion chains.
- **Code Executability Score (CES):** Percentage of generated Python/SPARQL snippets executing without runtime errors.

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

