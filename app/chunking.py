def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i + chunk_size])
        i += max(1, chunk_size - overlap)
    return out
