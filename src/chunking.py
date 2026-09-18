from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    page: int | None = None


def token_aware_chunks(text: str, tokenizer: Any, max_tokens: int = 768, overlap: int = 64, page: int | None = None) -> list[TextChunk]:
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("max_tokens must be positive and overlap must be in [0, max_tokens).")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, start = [], 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunks.append(TextChunk(len(chunks), tokenizer.decode(token_ids[start:end], skip_special_tokens=True), page))
        if end == len(token_ids):
            break
        start = end - overlap
    return chunks