import os
import tempfile
from typing import Literal
from .llm_router import route_answer
from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .db import Base, engine, get_db
from .ingest import ingest_pdf
from .retrieval import retrieve
from .llm import answer_with_ollama

os.makedirs("./data", exist_ok=True)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Local RAG (SQLite + FAISS + Ollama Gemma3)", version="1.0.0")
from fastapi.staticfiles import StaticFiles



class QueryIn(BaseModel):
    question: str
    top_k: int = 5
    mode: Literal["auto", "local", "api"] = "auto"  # yeni

@app.post("/v1/ingest/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    doc_id = ingest_pdf(db, tmp_path, title=file.filename or "pdf", source=file.filename or "")
    return {"document_id": doc_id}

@app.post("/v1/query")
def query_endpoint(payload: QueryIn, db: Session = Depends(get_db)):
    hits = retrieve(db, payload.question, top_k=payload.top_k)

    answer, provider, model = route_answer(
        question=payload.question,
        hits=hits,
        mode=payload.mode
    )

    citations = [{
        "n": i + 1,
        "document_id": h["document_id"],
        "title": h["title"],
        "source": h["source"],
        "chunk_index": h["chunk_index"],
        "score": h["score"],
    } for i, h in enumerate(hits)]

    return {
        "answer": answer,
        "provider": provider,   # "local" | "api" | "none"
        "model": model,         # hangi model kullanıldı
        "citations": citations,
        "context": hits,
    }
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# UI dosyalarını /ui altında sun
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

# Ana sayfada index.html göster
@app.get("/")
def root():
    return FileResponse("ui/index.html")

