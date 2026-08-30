"""DRUM-ML Fine-Tuning Recipe: Unsloth QLoRA / LoRA for Qwen 2.5 & LLaMA 3.1.

Fast, memory-efficient fine-tuning on consumer/cloud GPUs (Colab, RunPod, Vast.ai, RTX 3090/4090, A100).
Supports:
- Qwen/Qwen2.5-7B-Instruct (Recommended)
- meta-llama/Llama-3.1-8B-Instruct
- deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
"""

import argparse
import os
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM on DRUM-ML using Unsloth.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="dataset/train.jsonl",
        help="Path to training JSONL dataset",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="dataset/val.jsonl",
        help="Path to validation JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/drum_ml_model",
        help="Output directory for checkpoints and adapters",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum context length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Use 4-bit QLoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_merged", action="store_true", help="Merge adapter and save 16-bit model")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch
    except ImportError:
        raise ImportError(
            "Unsloth and training dependencies not found.\n"
            "Install via: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'\n"
            "             pip install --no-deps trl peft accelerate bitsandbytes"
        )

    print(f"🚀 Initializing base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"🔧 Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"📂 Loading DRUM-ML dataset: {args.train_file}")
    data_files = {"train": args.train_file}
    if os.path.exists(args.val_file):
        data_files["validation"] = args.val_file

    dataset = load_dataset("json", data_files=data_files)

    def format_chat(batch):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    print("📝 Applying tokenizer chat template...")
    formatted_dataset = dataset.map(format_chat, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        eval_strategy="steps" if "validation" in formatted_dataset else "no",
        eval_steps=500 if "validation" in formatted_dataset else None,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset.get("validation"),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("🔥 Starting training...")
    trainer.train()

    print(f"💾 Saving LoRA adapter checkpoint to {args.output_dir}/adapter...")
    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "adapter"))

    if args.save_merged:
        merged_path = os.path.join(args.output_dir, "merged_16bit")
        print(f"🔄 Merging LoRA weights and saving 16-bit model to {merged_path}...")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print("✅ Training complete!")


if __name__ == "__main__":
    main()
