"""Configuración central de la aplicación (cargada desde variables de entorno / .env)."""

from functools import lru_cache
from pathlib import Path

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

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
