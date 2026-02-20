import re
from typing import Literal, Tuple
from openai import OpenAI
from .settings import settings

SYSTEM_PROMPT = (
    "Sen bir şirket içi bilgi asistanısın.\n"
    "- SADECE verilen KAYNAKLAR'a dayanarak cevap ver.\n"
    "- Kaynaklarda yoksa: 'Bunu kaynaklarda bulamadım.' de.\n"
    "- Cevap içinde iddialarını [1], [2] şeklinde kaynak numarasıyla belirt.\n"
    "- Düşünme notlarını yazma; sadece sonucu yaz.\n"
)

def _build_context(hits: list[dict], max_chars: int = 12000) -> str:
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

def _has_citations(answer: str) -> bool:
    # [1], [2] ... gibi referanslar var mı?
    return bool(re.search(r"\[\d+\]", answer or ""))

def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)

_local_client = _make_client(settings.OLLAMA_BASE_URL, settings.OLLAMA_API_KEY)
_api_client = _make_client(settings.API_BASE_URL, settings.API_KEY) if settings.API_ENABLED else None

def _generate(client: OpenAI, model: str, question: str, hits: list[dict]) -> str:
    context = _build_context(hits)
    user_prompt = f"KAYNAKLAR:\n{context}\n\nSORU: {question}\n\nCEVAP:"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

def route_answer(
    question: str,
    hits: list[dict],
    mode: Literal["auto", "local", "api"] = "auto",
) -> Tuple[str, str, str]:
    """
    Returns: (answer, provider, model)
    provider: "local" | "api" | "none"
    """
    if not hits:
        return "Bu soruya uygun kaynak parçası bulamadım.", "none", ""

    top_score = float(hits[0].get("score", 0.0))
    if top_score < settings.ROUTER_MIN_SCORE:
        return "Kaynaklar bu soruya yeterince yakın görünmüyor. Bunu kaynaklarda bulamadım.", "none", ""

    # Zorla API istenmiş ama API yoksa
    if mode == "api":
        if not settings.API_ENABLED or not settings.API_BASE_URL or not settings.API_KEY or not settings.API_MODEL:
            return "API modu seçildi ama API ayarları eksik. Lütfen .env dosyasını doldurun.", "none", ""
        try:
            ans = _generate(_api_client, settings.API_MODEL, question, hits)
            return ans, "api", settings.API_MODEL
        except Exception as e:
            return f"API çağrısı hata verdi: {e}", "none", ""

    # Zorla local istenmişse
    if mode == "local":
        try:
            ans = _generate(_local_client, settings.OLLAMA_MODEL, question, hits)
            return ans, "local", settings.OLLAMA_MODEL
        except Exception as e:
            return f"Yerel model hata verdi: {e}", "none", ""

    # AUTO: önce local dene
    try:
        local_ans = _generate(_local_client, settings.OLLAMA_MODEL, question, hits)
    except Exception:
        local_ans = ""

    need_fallback = False
    if not local_ans:
        need_fallback = True
    elif settings.ROUTER_REQUIRE_CITATIONS and not _has_citations(local_ans):
        # kaynakları kullanmamış gibi davranıyorsa API ile tekrar dene
        need_fallback = True
    elif "kaynaklarda bulamadım" in (local_ans or "").lower() and top_score >= settings.ROUTER_MIN_SCORE:
        # kaynak aslında var ama local "bulamadım" dedi -> API ile dene
        need_fallback = True

    if need_fallback and settings.API_ENABLED and settings.API_BASE_URL and settings.API_KEY and settings.API_MODEL:
        try:
            api_ans = _generate(_api_client, settings.API_MODEL, question, hits)
            # API cevabı boşsa local'a dön
            if api_ans:
                return api_ans, "api", settings.API_MODEL
        except Exception:
            pass

    # fallback yoksa ya da başarısızsa local dön
    return local_ans or "Cevap üretilemedi.", "local", settings.OLLAMA_MODEL