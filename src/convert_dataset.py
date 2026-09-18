"""Convert supported local legal data to normalized summarization JSONL.

The converter intentionally does not guess arbitrary text boundaries. JSON/JSONL
records must expose document and summary fields; plain text is accepted only
when it uses an explicit ``SUMMARY:`` separator.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "summarization.jsonl"
FIELD_ALIASES = {
    "document": ("document", "doc", "text", "content", "body"),
    "summary": ("summary", "summarization", "abstract", "short_summary"),
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _from_record(record: dict[str, Any]) -> tuple[str, str] | None:
    lowered = {str(key).lower(): value for key, value in record.items()}
    values = {}
    for target, aliases in FIELD_ALIASES.items():
        values[target] = next((lowered[key] for key in aliases if key in lowered), "")
    document, summary = normalize_text(values["document"]), normalize_text(values["summary"])
    return (document, summary) if document and summary else None


def iter_examples(path: Path) -> Iterator[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8-sig") as handle:
            if suffix == ".json":
                payload = json.load(handle)
                records = payload if isinstance(payload, list) else [payload]
            else:
                records = (json.loads(line) for line in handle if line.strip())
            for record in records:
                if isinstance(record, dict):
                    example = _from_record(record)
                    if example:
                        yield example
        return
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        matches = re.split(r"\n\s*(?:={3,}|-{3,})\s*\n", text)
        for block in matches:
            parts = re.split(r"\n\s*SUMMARY\s*:\s*\n", block, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                document, summary = normalize_text(parts[0]), normalize_text(parts[1])
                if document and summary:
                    yield document, summary


def convert(input_dir: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    sample_dirs = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and (path / "EN_Judgment.txt").exists()
    )
    if sample_dirs:
        return _convert_sample_directories(sample_dirs, output_path)

    files = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".txt"})
    if not files:
        raise FileNotFoundError(f"No supported dataset files found in {input_dir}. Add JSON, JSONL, or explicitly separated TXT data.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    stats = {"files": len(files), "processed": 0, "skipped": 0, "duplicates": 0}
    with output_path.open("w", encoding="utf-8") as output:
        for path in files:
            try:
                examples = iter_examples(path)
                for document, summary in examples:
                    pair = (document, summary)
                    if pair in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(pair)
                    output.write(json.dumps({"document": document, "summary": summary}, ensure_ascii=False) + "\n")
                    stats["processed"] += 1
            except (UnicodeError, json.JSONDecodeError, OSError) as error:
                stats["skipped"] += 1
                print(f"Skipped malformed file {path}: {error}")
    print(f"Files: {stats['files']} | processed: {stats['processed']} | skipped: {stats['skipped']} | duplicates: {stats['duplicates']}")
    return stats


def _convert_sample_directories(sample_dirs: list[Path], output_path: Path) -> dict[str, int]:
    """Convert the observed judgment/summary directory layout.

    Hindi summaries are intentionally excluded because the target task is
    English legal-document summarization.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    stats = {"files": len(sample_dirs), "processed": 0, "skipped": 0, "duplicates": 0}
    with output_path.open("w", encoding="utf-8") as output:
        for sample_dir in sample_dirs:
            judgment_path = sample_dir / "EN_Judgment.txt"
            summary_path = sample_dir / "EN_Summary.txt"
            if not summary_path.exists():
                stats["skipped"] += 1
                print(f"Skipped {sample_dir}: missing EN_Summary.txt")
                continue
            try:
                document = normalize_text(judgment_path.read_text(encoding="utf-8-sig"))
                summary = normalize_text(summary_path.read_text(encoding="utf-8-sig"))
            except (UnicodeError, OSError) as error:
                stats["skipped"] += 1
                print(f"Skipped malformed sample {sample_dir}: {error}")
                continue
            if not document or not summary:
                stats["skipped"] += 1
                print(f"Skipped {sample_dir}: empty judgment or summary")
                continue
            pair = (document, summary)
            if pair in seen:
                stats["duplicates"] += 1
                continue
            seen.add(pair)
            output.write(json.dumps({"document": document, "summary": summary}, ensure_ascii=False) + "\n")
            stats["processed"] += 1
    print(f"Sample directories: {stats['files']} | processed: {stats['processed']} | skipped: {stats['skipped']} | duplicates: {stats['duplicates']}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.input_dir, args.output)