"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Clinical RAG Platform.

    All values can be overridden via environment variables or a .env file.
    Supports two LLM backends: Groq (default, free) or Ollama (local).
    Embeddings are always local via sentence-transformers (no API cost).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Backend ──────────────────────────────────────────────────────────
    # Set LLM_BACKEND="ollama" to use local Ollama instead of Groq.
    LLM_BACKEND: str = "groq"  # "groq" | "ollama"

    # Groq (free tier — https://console.groq.com)
    GROQ_API_KEY: str = ""
    CHAT_MODEL: str = "llama-3.3-70b-versatile"  # best free Groq model

    # Ollama (local — https://ollama.ai)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ── Embeddings (always local, zero API cost) ──────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # ~80MB, no API key needed
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" | "cuda" | "mps"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "clinical_docs"
    # Optional Qdrant Cloud API key (leave empty for local)
    QDRANT_API_KEY: str = ""

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── Logging / Runtime ────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    MAX_RETRIES: int = 3

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_STRATEGY: str = "recursive"

    # ── API ───────────────────────────────────────────────────────────────────
    APP_VERSION: str = "0.1.0"


settings = Settings()
