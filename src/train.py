"""Fine-tune GPT-2 for legal summarization using prepared JSONL splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

try:
    from .config import load_config
    from .utils import load_jsonl, set_seed, summarization_prompt
except ImportError:
    from config import load_config
    from utils import load_jsonl, set_seed, summarization_prompt


def build_dataset(path: Path, tokenizer, max_length: int) -> Dataset:
    rows = load_jsonl(path)
    texts = [summarization_prompt(row["document"]) + row["summary"] for row in rows]
    return Dataset.from_dict({"text": texts}).map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True, remove_columns=["text"],
    )


def train(data_dir: Path, output_dir: Path, config_path: Path) -> None:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    train_data = build_dataset(data_dir / "train.jsonl", tokenizer, int(config["max_length"]))
    validation_data = build_dataset(data_dir / "validation.jsonl", tokenizer, int(config["max_length"]))
    args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["train_batch_size"]), per_device_eval_batch_size=int(config["eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]), learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), warmup_ratio=float(config["warmup_ratio"]),
        evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", fp16=torch.cuda.is_available(), report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
                      tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/legal-gpt2"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.config)