from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from .inference import summarize
    from .utils import load_jsonl, qa_prompt
except ImportError:
    from inference import summarize
    from utils import load_jsonl, qa_prompt


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = prediction.lower().split(), reference.lower().split()
    overlap = len(set(predicted) & set(expected))
    if not predicted or not expected or not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def evaluate_qa(path: Path, model_path: str) -> dict[str, float]:
    rows = load_jsonl(path)
    predictions = []
    for row in rows:
        predictions.append(summarize(qa_prompt(row["document"], row["question"]), model_path))
    exact = [prediction.strip().lower() == row["answer"].strip().lower() for prediction, row in zip(predictions, rows)]
    f1 = [token_f1(prediction, row["answer"]) for prediction, row in zip(predictions, rows)]
    result = {"exact_match": sum(exact) / len(exact) if exact else 0.0, "token_f1": sum(f1) / len(f1) if f1 else 0.0}
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("data/processed/qa_test.jsonl"))
    parser.add_argument("--model", default="outputs/legal-gpt2-qa")
    args = parser.parse_args()
    evaluate_qa(args.test, args.model)