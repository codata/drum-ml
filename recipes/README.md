# 🍳 DRUM-ML Fine-Tuning Recipes

This directory contains standalone, reproducible training recipes to fine-tune open-weights LLMs on the **DRUM-ML** metrology dataset.

---

## 🎯 Recommended Models

| Model | Why | Recommended VRAM |
|---|---|---|
| **`Qwen/Qwen2.5-7B-Instruct`** *(Recommended)* | State-of-the-art in STEM, dimensional analysis, and LaTeX formula derivation. | 16–24 GB (Single GPU with QLoRA) |
| **`meta-llama/Llama-3.1-8B-Instruct`** | Robust general instruction following and ecosystem alignment. | 16–24 GB (Single GPU with QLoRA) |
| **`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`** | Exceptional chain-of-thought derivation for multi-step metrological uncertainty. | 24 GB (Single GPU with QLoRA) |

---

## 🚀 Quickstart: Option A — Unsloth (Fastest & Lowest Memory)

[Unsloth](https://github.com/unslothai/unsloth) delivers 2–5x faster training and 70% lower VRAM usage.

### 1. Install Dependencies
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### 2. Launch Fine-Tuning
```bash
python recipes/train_unsloth_qwen.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_file "dataset/train.jsonl" \
    --val_file "dataset/val.jsonl" \
    --output_dir "outputs/qwen25_7b_metrology" \
    --batch_size 4 \
    --grad_accum 4 \
    --epochs 1 \
    --save_merged
```

---

## 🚀 Option B — Standard Hugging Face TRL (Multi-GPU / Accelerate)

### 1. Install Dependencies
```bash
pip install -e ".[training]"
```

### 2. Launch Single or Multi-GPU Training
```bash
# Single GPU
python recipes/train_trl_sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_file "dataset/train.jsonl" \
    --val_file "dataset/val.jsonl" \
    --output_dir "outputs/drum_ml_sft"

# Multi-GPU with Accelerate
accelerate launch recipes/train_trl_sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "outputs/drum_ml_sft"
```

---

## 📤 Publishing Fine-Tuned Models to Hugging Face Hub

Once trained, upload your model adapter or merged weights to Hugging Face:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/qwen25_7b_metrology/merged_16bit",
    repo_id="codata-drum/qwen-2.5-7b-metrology",
    repo_type="model",
)
```
