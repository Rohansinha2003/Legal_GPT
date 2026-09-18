from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .chunking import token_aware_chunks
    from .utils import summarization_prompt
except ImportError:
    from chunking import token_aware_chunks
    from utils import summarization_prompt


def summarize(document: str, model_path: str | Path, max_new_tokens: int = 150, temperature: float = 0.7,
              top_p: float = 0.9, do_sample: bool = True, no_repeat_ngram_size: int = 3) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
    inputs = tokenizer(summarization_prompt(document), return_tensors="pt", truncation=True, max_length=min(tokenizer.model_max_length, 1024)).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                                do_sample=do_sample, no_repeat_ngram_size=no_repeat_ngram_size, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def hierarchical_summarize(document: str, model_path: str | Path, max_tokens: int = 768, overlap: int = 64, **kwargs) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chunks = token_aware_chunks(document, tokenizer, max_tokens=max_tokens, overlap=overlap)
    if len(chunks) <= 1:
        return summarize(document, model_path, **kwargs)
    return summarize("\n\n".join(summarize(chunk.text, model_path, **kwargs) for chunk in chunks), model_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/legal-gpt2")
    parser.add_argument("--text")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()
    document = args.text if args.text is not None else args.file.read_text(encoding="utf-8") if args.file else input("Legal document: ")
    print("SUMMARY:\n" + summarize(document, args.model, max_new_tokens=args.max_new_tokens))