# Legal-GPT2

## Overview

Legal-GPT2 is a research and education pipeline for GPT-2 legal-document summarization, document-grounded Q&A, token-aware long-document processing, retrieval, and an HTTP/UI layer. This checkout contains code and tests, but `data/raw/` is intentionally empty: no dataset, license, metrics, or trained checkpoint is fabricated.

## Features

- Discovery-based JSON, JSONL, and explicitly separated TXT conversion
- UTF-8 JSONL validation, duplicate detection, and deterministic 80/10/10 splits
- Configurable GPT-2 causal-language-model fine-tuning
- Hierarchical token-aware summarization
- ROUGE evaluation and qualitative prediction output
- Q&A training/evaluation and PDF/TXT RAG with page-aware sources
- FastAPI endpoints and a Vite React client
- Colab notebook calling the same project scripts

## Architecture

`raw data -> conversion -> validation/splits -> GPT-2 -> evaluation -> chunking/RAG -> FastAPI -> React`

## Dataset and Dataset License

Place only licensed data in `data/raw/`. JSON records must expose document and summary fields (common aliases are supported). TXT requires `SUMMARY:` between document and summary, with `===` or `---` separating examples. Record the source, version, jurisdiction, time period, and license before training. Raw data is ignored by Git.

## Installation

Python 3.10+ is recommended. The current development environment uses Python 3.9.6 and the dataset modules remain compatible with it.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
cd frontend && npm install
```

## Local Development

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m compileall -q src app
```

## Dataset Preparation

```bash
./.venv/bin/python src/convert_dataset.py
./.venv/bin/python src/prepare_dataset.py
```

The converter never modifies raw files. It stops if no supported source exists or fewer than three unique examples can form non-empty splits.

## Training and Evaluation

```bash
./.venv/bin/python src/train.py
./.venv/bin/python src/evaluate.py
```

Training parameters live in `configs/training.yaml`. Checkpoints and outputs are Git-ignored. Do not report ROUGE or QA results until these commands have run on the actual licensed dataset.

## Google Colab

Open `notebooks/legal_gpt2_colab.ipynb` in Colab, replace the placeholder repository URL, upload or fetch licensed data, and run the cells in order. It runs `src/convert_dataset.py`, `src/prepare_dataset.py`, `src/train.py`, and `src/evaluate.py` directly.

## Inference, Summarization, and Q&A

```bash
./.venv/bin/python src/inference.py --model outputs/legal-gpt2 --file document.txt
./.venv/bin/python src/train_qa.py
./.venv/bin/python src/evaluate_qa.py
```

Long documents use token-aware chunks and hierarchical summarization. Q&A is intended to answer only from the supplied document.

## RAG

`src/rag.py` supports PDF page extraction with PyMuPDF, TXT input, SentenceTransformer embeddings, FAISS retrieval, and GPT-2 generation. Every returned source retains document name, page, chunk ID, and source text. The API currently exposes index construction as an application integration point and returns HTTP 503 until an index is configured.

## API and Frontend

```bash
./.venv/bin/python -m uvicorn app.main:app --reload
cd frontend && npm run dev
```

Endpoints are `/health`, `/summarize`, `/question`, `/upload`, and `/rag/question`. The frontend runs at the Vite URL and calls the local API.

## Results and Hugging Face

No results are included because no dataset or training run exists in this checkout. After evaluation, publish only a reviewed checkpoint and model card to `YOUR_USERNAME/legal-gpt2`; never put tokens in Git or notebooks. The model card must include dataset license, parameters, metrics, jurisdiction, time period, limitations, and known risks.

## Limitations and Responsible Use

This system is intended for research and educational purposes. Generated outputs may contain errors or hallucinations, and legal information may become outdated. Different jurisdictions have different laws. The system is not a substitute for qualified legal advice. Verify important claims against authoritative primary sources and, where appropriate, qualified legal professionals. Do not present generated answers as legal advice or as authoritative citations.

## License

Code is MIT licensed. Dataset and model licenses remain separate and must be documented before use.