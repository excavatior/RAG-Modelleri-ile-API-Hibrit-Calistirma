from sentence_transformers import SentenceTransformer
import numpy as np
from .settings import settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBED_MODEL)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embs = model.encode(texts, normalize_embeddings=True)
    if isinstance(embs, np.ndarray):
        embs = embs.astype("float32").tolist()
    return embs

def embed_query(q: str) -> list[float]:
    return embed_texts([q])[0]
