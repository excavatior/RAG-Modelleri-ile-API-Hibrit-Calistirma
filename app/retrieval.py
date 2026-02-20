from sqlalchemy.orm import Session
from sqlalchemy import select
from .models import Chunk, Document
from .embeddings import embed_query
from .faiss_store import store

def retrieve(db: Session, question: str, top_k: int = 5):
    q = embed_query(question)
    scores, ids = store.search(q, top_k)
    if not ids:
        return []

    stmt = (
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(ids))
    )
    rows = db.execute(stmt).all()

    by_id = {chunk.id: (chunk, doc) for chunk, doc in rows}

    # FAISS sırasını koru
    results = []
    for score, cid in zip(scores, ids):
        if cid not in by_id:
            continue
        chunk, doc = by_id[cid]
        results.append({
            "score": score,
            "chunk_id": chunk.id,
            "document_id": doc.id,
            "title": doc.title,
            "source": doc.source,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
        })
    return results
