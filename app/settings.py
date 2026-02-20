from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Lokal DB + FAISS
    DATABASE_URL: str = "sqlite:///./data/rag.db"
    FAISS_INDEX_PATH: str = "./data/faiss.index"

    # Embeddings
    EMBED_MODEL: str = "intfloat/multilingual-e5-base"
    EMBED_DIM: int = 768

    # Yerel LLM (Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1/"
    OLLAMA_API_KEY: str = "ollama"      # dummy
    OLLAMA_MODEL: str = "gemma3:4b"

    # API LLM (Hibritin ikinci bacağı)
    API_ENABLED: bool = False
    API_BASE_URL: str = "https://api.openai.com/v1/"             # örn: https://api.openai.com/v1/
    API_KEY: str = ""                  # API key
    API_MODEL: str = "gpt-4o-mini"                # örn: gpt-4o-mini

    # Router eşikleri
    ROUTER_MIN_SCORE: float = 0.22     # retrieval zayıfsa LLM’i boşuna çağırmayalım
    ROUTER_REQUIRE_CITATIONS: bool = True  # cevapta [1] [2] yoksa fallback dene

settings = Settings()