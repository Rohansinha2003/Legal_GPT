from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate

try:
    from .inference import summarize
    from .utils import load_jsonl
except ImportError:
    from inference import summarize
    from utils import load_jsonl


def evaluate_model(test_path: Path, model_path: str, output_path: Path) -> dict:
    rows = load_jsonl(test_path)
    predictions = [summarize(row["document"], model_path) for row in rows]
    metrics = evaluate.load("rouge").compute(predictions=predictions, references=[row["summary"] for row in rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row, prediction in zip(rows, predictions):
            handle.write(json.dumps({"original": row["document"], "reference_summary": row["summary"], "model_summary": prediction}, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--model", default="outputs/legal-gpt2")
    parser.add_argument("--output", type=Path, default=Path("evaluation/summarization_predictions.jsonl"))
    args = parser.parse_args()
    evaluate_model(args.test, args.model, args.output)