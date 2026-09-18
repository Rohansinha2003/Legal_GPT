import json

import pytest

from src.convert_dataset import convert
from src.prepare_dataset import read_jsonl, validate_and_split


def test_conversion_and_validation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "examples.json").write_text(json.dumps([
        {"document": " A legal document. ", "summary": " A summary. "},
        {"document": " A legal document. ", "summary": " A summary. "},
    ]), encoding="utf-8")
    processed = tmp_path / "processed.jsonl"
    result = convert(raw, processed)
    assert result["processed"] == 1
    with pytest.raises(ValueError, match="At least 3 unique examples"):
        validate_and_split(processed, tmp_path / "splits")


def test_empty_raw_directory_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="No supported dataset files"):
        convert(tmp_path, tmp_path / "out.jsonl")


def test_sample_directory_layout_pairs_english_files(tmp_path):
    sample = tmp_path / "Sample_1"
    sample.mkdir()
    (sample / "EN_Judgment.txt").write_text("The court decided the matter.", encoding="utf-8")
    (sample / "EN_Summary.txt").write_text("The court issued a decision.", encoding="utf-8")
    (sample / "HI_Summary.txt").write_text("हिंदी सारांश", encoding="utf-8")
    output = tmp_path / "processed.jsonl"
    stats = convert(tmp_path, output)
    assert stats["processed"] == 1
    rows, malformed = read_jsonl(output)
    assert malformed == 0
    assert rows[0]["summary"] == "The court issued a decision."