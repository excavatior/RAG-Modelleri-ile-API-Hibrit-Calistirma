from pypdf import PdfReader
from sqlalchemy.orm import Session
from .models import Document, Chunk
from .chunking import chunk_text
from .embeddings import embed_texts
from .faiss_store import store

def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def ingest_pdf(db: Session, pdf_path: str, title: str, source: str) -> int:
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)

    doc = Document(title=title or "pdf", source=source or "")
    db.add(doc)
    db.flush()  # doc.id

    chunk_rows: list[Chunk] = []
    for idx, t in enumerate(chunks):
        row = Chunk(document_id=doc.id, chunk_index=idx, text=t)
        db.add(row)
        chunk_rows.append(row)

    db.flush()  # chunk id'leri

    if chunk_rows:
        embs = embed_texts([c.text for c in chunk_rows])
        ids = [c.id for c in chunk_rows]
        store.add_with_ids(embs, ids)

    db.commit()
    return doc.id
