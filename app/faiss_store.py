import os
import faiss
import numpy as np
from .settings import settings

class FaissStore:
    def __init__(self, path: str, dim: int):
        self.path = path
        self.dim = dim
        self.index = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.path):
            idx = faiss.read_index(self.path)
            if not isinstance(idx, faiss.IndexIDMap):
                idx = faiss.IndexIDMap2(idx)
            return idx

        base = faiss.IndexFlatIP(self.dim)  # normalize embeddings => IP ~ cosine
        return faiss.IndexIDMap2(base)

    def add_with_ids(self, vectors: list[list[float]], ids: list[int]):
        if not vectors:
            return
        v = np.asarray(vectors, dtype="float32")
        i = np.asarray(ids, dtype="int64")
        self.index.add_with_ids(v, i)
        faiss.write_index(self.index, self.path)

    def search(self, query_vector: list[float], top_k: int):
        if self.index.ntotal == 0:
            return [], []
        q = np.asarray([query_vector], dtype="float32")
        D, I = self.index.search(q, top_k)
        pairs = [(float(d), int(i)) for d, i in zip(D[0].tolist(), I[0].tolist()) if i != -1]
        if not pairs:
            return [], []
        scores, ids = zip(*pairs)
        return list(scores), list(ids)

store = FaissStore(settings.FAISS_INDEX_PATH, settings.EMBED_DIM)
