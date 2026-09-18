from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .chunking import token_aware_chunks
except ImportError:
    from chunking import token_aware_chunks


INSUFFICIENT = "The provided document does not contain enough information to answer this question."


class LegalRAG:
    def __init__(self, model_path: str = "outputs/legal-gpt2-qa", embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path).eval()
        self.records = []
        self.index = None

    @staticmethod
    def extract(path: Path) -> list[tuple[int, str]]:
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as document:
                return [(page.number + 1, page.get_text()) for page in document]
        if path.suffix.lower() == ".txt":
            return [(1, path.read_text(encoding="utf-8"))]
        raise ValueError("Only PDF and TXT documents are supported.")

    def add_document(self, path: Path) -> None:
        for page, text in self.extract(path):
            for chunk in token_aware_chunks(text, self.tokenizer, page=page):
                self.records.append({"document": path.name, "page": page, "chunk_id": chunk.chunk_id, "text": chunk.text})
        vectors = self.encoder.encode([record["text"] for record in self.records], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.asarray(vectors, dtype="float32"))

    def question(self, question: str, top_k: int = 4) -> dict:
        if self.index is None or not self.records:
            return {"answer": INSUFFICIENT, "sources": []}
        query = self.encoder.encode([question], normalize_embeddings=True)
        _, indices = self.index.search(np.asarray(query, dtype="float32"), min(top_k, len(self.records)))
        sources = [self.records[index] for index in indices[0] if index >= 0]
        context = "\n\n".join(source["text"] for source in sources)
        prompt = f"### TASK:\nAnswer the question using ONLY the provided legal context.\n\n### CONTEXT:\n{context}\n\n### QUESTION:\n{question}\n\n### ANSWER:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output = self.model.generate(**inputs, max_new_tokens=160, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        answer = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip() or INSUFFICIENT
        return {"answer": answer, "sources": sources}