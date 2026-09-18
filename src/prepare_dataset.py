"""Validate summarization JSONL and create deterministic leakage-resistant splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "summarization.jsonl"


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    rows, malformed = [], 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
                else:
                    malformed += 1
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def validate_and_split(input_path: Path = DEFAULT_INPUT, output_dir: Path | None = None, seed: int = 42) -> dict:
    output_dir = output_dir or input_path.parent
    rows, malformed = read_jsonl(input_path)
    valid = [row for row in rows if str(row.get("document", "")).strip() and str(row.get("summary", "")).strip()]
    documents = [str(row["document"]).strip() for row in valid]
    pairs = [(doc, str(row["summary"]).strip()) for doc, row in zip(documents, valid)]
    unique = list(dict.fromkeys((doc, summary) for doc, summary in pairs))
    stats = {
        "total": len(rows), "malformed_json": malformed,
        "empty_documents": sum(not str(row.get("document", "")).strip() for row in rows),
        "empty_summaries": sum(not str(row.get("summary", "")).strip() for row in rows),
        "duplicate_documents": len(documents) - len(set(documents)),
        "duplicate_pairs": len(pairs) - len(set(pairs)),
        "minimum_document_length": min(map(len, documents), default=0),
        "maximum_document_length": max(map(len, documents), default=0),
        "average_document_length": sum(map(len, documents)) / len(documents) if documents else 0,
        "average_summary_length": sum(len(summary) for _, summary in pairs) / len(pairs) if pairs else 0,
    }
    if not unique:
        raise ValueError(f"No valid examples found in {input_path}. Statistics: {stats}")
    if len(unique) < 3:
        raise ValueError("At least 3 unique examples are required for non-empty train/validation/test splits.")
    train, remainder = train_test_split(unique, test_size=0.2, random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for document, summary in split:
                handle.write(json.dumps({"document": document, "summary": summary}, ensure_ascii=False) + "\n")
    print(json.dumps({**stats, "valid": len(unique), "train": len(train), "validation": len(validation), "test": len(test)}, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_and_split(args.input, args.output_dir, args.seed)