from openai import OpenAI
from .settings import settings

client = OpenAI(
    base_url=settings.OLLAMA_BASE_URL,
    api_key=settings.OLLAMA_API_KEY,  # required but ignored :contentReference[oaicite:6]{index=6}
)

SYSTEM_PROMPT = (
    "Sen bir şirket içi bilgi asistanısın.\n"
    "- SADECE verilen KAYNAKLAR'a dayanarak cevap ver.\n"
    "- Kaynaklarda yoksa: 'Bunu kaynaklarda bulamadım.' de.\n"
    "- Cevap içinde iddialarını [1], [2] şeklinde kaynak numarasıyla belirt.\n"
    "- Düşünme notlarını yazma; sadece sonucu yaz.\n"
)

def build_context(hits: list[dict], max_chars: int = 12000) -> str:
    blocks = []
    total = 0
    for i, h in enumerate(hits, start=1):
        title = h.get("title") or "doc"
        chunk = (h.get("text") or "").strip()
        block = f"[{i}] {title} (chunk={h.get('chunk_index')}):\n{chunk}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)

def answer_with_ollama(question: str, hits: list[dict]) -> str:
    context = build_context(hits)
    user_prompt = (
        f"KAYNAKLAR:\n{context}\n\n"
        f"SORU: {question}\n\n"
        "CEVAP:"
    )

    resp = client.chat.completions.create(
        model=settings.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""
