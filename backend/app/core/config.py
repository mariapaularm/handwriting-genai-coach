"""Configuración central de la aplicación (cargada desde variables de entorno / .env)."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Se lee primero el .env de la raíz del repo y luego backend/.env (este último prevalece).
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Handwriting GenAI Coach API"
    APP_VERSION: str = "0.1.0"

    # Base de datos
    DATABASE_URL: str = "sqlite+aiosqlite:///./handwriting.db"
    DATABASE_ECHO: bool = False

    # Subida de imágenes
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE_MB: int = 10
    MAX_PROCESSED_DIMENSION: int = 1024

    # CORS (frontend Vite)
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Gemini (LangChain)
    GEMINI_API_KEY: SecretStr | None = None
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GEMINI_TEMPERATURE: float = 0.2
    # Nivel de razonamiento para modelos Gemini 3.x ("low", "medium", "high"; vacío = el del modelo).
    # "low" reduce latencia (medido en nov-2026: gemini-3.5-flash ~30 s/petición; gemini-3.8-flash 60-100 s).
    GEMINI_THINKING_LEVEL: str | None = "low"
    # Plazo por intento: se envía a Gemini como X-Server-Timeout (si vence, responde 504 DEADLINE_EXCEEDED).
    GEMINI_TIMEOUT_SECONDS: float = 180.0
    # Margen extra del timeout local para que llegue antes el error limpio del servidor.
    GEMINI_TIMEOUT_GRACE_SECONDS: float = 5.0
    # Reintentos adicionales ante 429 / 5xx / timeouts (backoff exponencial con jitter)...
    GEMINI_MAX_RETRIES: int = 4
    # ...sin superar este presupuesto total por operación (evita esperas de varios minutos).
    GEMINI_MAX_TOTAL_SECONDS: float = 240.0
    GEMINI_RETRY_INITIAL_SECONDS: float = 1.0
    GEMINI_RETRY_MAX_SECONDS: float = 20.0
    # Lado mayor con el que se envía la imagen original al VLM (reduce tokens y latencia).
    VLM_MAX_IMAGE_DIMENSION: int = 1600

    # RAG (ChromaDB)
    KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "knowledge_base"
    CHROMA_DIR: Path = Path("chroma_db")
    CHROMA_COLLECTION: str = "wellness_knowledge"
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 150
    RAG_TOP_K: int = 3

    # MCP
    MCP_HISTORY_LIMIT: int = 5
    MCP_TREND_WINDOW: int = 10

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
