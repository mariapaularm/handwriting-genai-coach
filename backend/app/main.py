"""Punto de entrada de la API FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.database import engine, init_db
from app.core.gemini import GeminiError, GeminiRateLimitError
from app.services.image_service import ImageValidationError


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield
    await engine.dispose()


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ImageValidationError)
async def image_validation_exception_handler(_request: Request, exc: ImageValidationError) -> JSONResponse:
    """Convierte los errores de validación de imágenes en respuestas HTTP (400 por defecto)."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(GeminiError)
async def gemini_exception_handler(_request: Request, exc: GeminiError) -> JSONResponse:
    """Errores de Gemini tras agotar reintentos: 429 (rate limit), 502, 503 (config/caída) o 504 (timeout)."""
    headers = {"Retry-After": "30"} if isinstance(exc, GeminiRateLimitError) else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "operation": exc.operation, "provider_detail": exc.provider_detail},
        headers=headers,
    )


app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}
