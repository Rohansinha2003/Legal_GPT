from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import AnswerResponse, QuestionRequest, TextRequest
from src.inference import hierarchical_summarize, summarize

app = FastAPI(title="Legal-GPT2 API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize")
def summarize_document(request: TextRequest):
    return {"summary": summarize(request.document, "outputs/legal-gpt2")}


@app.post("/question", response_model=AnswerResponse)
def answer_question(request: QuestionRequest):
    answer = summarize(request.document + "\n\nQuestion: " + request.question, "outputs/legal-gpt2-qa")
    return {"answer": answer, "sources": []}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in {".txt", ".pdf"}:
        raise HTTPException(status_code=415, detail="Only PDF and TXT uploads are supported.")
    content = await file.read()
    return {"filename": file.filename, "text": content.decode("utf-8", errors="replace") if file.filename.lower().endswith(".txt") else "PDF text extraction is available through the RAG pipeline."}


@app.post("/rag/question", response_model=AnswerResponse)
def rag_question(request: QuestionRequest):
    raise HTTPException(status_code=503, detail="Build a RAG index from a document before using this endpoint.")