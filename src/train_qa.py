from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from .utils import load_jsonl, qa_prompt, set_seed
    from .config import load_config
except ImportError:
    from utils import load_jsonl, qa_prompt, set_seed
    from config import load_config


def train_qa(data_dir: Path, output_dir: Path, config_path: Path) -> None:
    config = load_config(config_path)
    set_seed(int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    rows = load_jsonl(data_dir / "qa_train.jsonl")
    texts = [qa_prompt(row["document"], row["question"], row["answer"]) for row in rows]
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    dataset = Dataset.from_dict({"text": texts}).map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=int(config["max_length"])), batched=True, remove_columns=["text"])
    args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=float(config["num_train_epochs"]), per_device_train_batch_size=int(config["train_batch_size"]), gradient_accumulation_steps=int(config["gradient_accumulation_steps"]), learning_rate=float(config["learning_rate"]), report_to="none")
    Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)).train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/legal-gpt2-qa"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    train_qa(args.data_dir, args.output_dir, args.config)