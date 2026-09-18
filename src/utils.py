from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarization_prompt(document: str) -> str:
    return f"### TASK: Summarize the legal document.\n\n### DOCUMENT:\n{document}\n\n### SUMMARY:\n"


def qa_prompt(document: str, question: str, answer: str | None = None) -> str:
    prompt = f"### TASK: Answer the legal question.\n\n### DOCUMENT:\n{document}\n\n### QUESTION:\n{question}\n\n### ANSWER:\n"
    return prompt if answer is None else prompt + answer