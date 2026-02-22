# Yerel RAG Asistanı (Ollama + Gemma3:4b) — API Hibrit

PDF’leri indexleyip (RAG) **yerel LLM** ile cevap üreten, gerektiğinde **API LLM’e fallback** yapabilen basit bir MVP.

## Özellikler
- PDF yükle & indexle (FAISS + embeddings)
- Yerel cevap üretimi (Ollama + `gemma3:4b`)
- Kaynaklar (citations)
- Türkçe modern web arayüzü
- Hibrit mod: `auto` / `local` / `api`

  <img width="1453" height="808" alt="image" src="https://github.com/user-attachments/assets/3acd23f2-531a-4457-887a-989fe2a9f3b0" />


## Kurulum

### 1) Ollama + Model
```bash
ollama pull gemma3:4b
2) Python ortamı
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Git Bash:
# source .venv/Scripts/activate

pip install -r requirements.txt
3) (Opsiyonel) API fallback için .env

.env repoya eklenmez (gitignore).

API_ENABLED=1
API_BASE_URL=https://api.openai.com/v1/
API_KEY=YOUR_API_KEY
API_MODEL=gpt-4o-mini
Çalıştırma
python -m uvicorn app.main:app --reload
Kullanım

UI: http://127.0.0.1:8000/

API:

POST /v1/ingest/pdf (PDF upload)

POST /v1/query (soru sor)

Örnek query:

curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Bu dokümanda ana konu nedir?","top_k":5,"mode":"auto"}'

mode: auto (varsayılan), local, api
