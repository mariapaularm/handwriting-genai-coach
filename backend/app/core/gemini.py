"""Clientes Gemini (LangChain) y capa de resiliencia común.

Todas las llamadas a Gemini (chat multimodal y embeddings) pasan por `call_gemini`, que:
- aplica un timeout duro por intento (`GEMINI_TIMEOUT_SECONDS`);
- reintenta con backoff exponencial + jitter (tenacity) ante 429, 5xx, timeouts y cortes de red;
- traduce el error final a una excepción tipada (`GeminiError`) con su status HTTP.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import TypeVar

import httpx
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    OutputParserException,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.core.config import settings

logger = logging.getLogger("handwriting.gemini")

T = TypeVar("T")

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_MARKERS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL")
_PLACEHOLDER_KEYS = {"", "pega_aqui_tu_api_key", "your-api-key"}


# ---------------------------------------------------------------------------
# Errores tipados
# ---------------------------------------------------------------------------


class GeminiError(Exception):
    """Fallo de Gemini tras agotar reintentos; `status_code` es el HTTP a devolver."""

    status_code = 502

    def __init__(self, message: str, *, operation: str) -> None:
        super().__init__(message)
        self.message = message
        self.operation = operation


class GeminiConfigurationError(GeminiError):
    """API key ausente/inválida o modelo inexistente: reintentar no sirve."""

    status_code = 503


class GeminiRateLimitError(GeminiError):
    status_code = 429


class GeminiTimeoutError(GeminiError):
    status_code = 504


class GeminiUnavailableError(GeminiError):
    status_code = 503


class GeminiResponseError(GeminiError):
    """Respuesta inesperada o no parseable."""

    status_code = 502


# ---------------------------------------------------------------------------
# Clasificación de excepciones (robusta a cómo cada SDK envuelve los errores)
# ---------------------------------------------------------------------------


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> int | None:
    for err in _exception_chain(exc):
        for attr in ("code", "status_code"):
            value = getattr(err, attr, None)
            if isinstance(value, int) and 100 <= value <= 599:
                return value
        response_status = getattr(getattr(err, "response", None), "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def _is_timeout(exc: BaseException) -> bool:
    return any(isinstance(e, (TimeoutError, httpx.TimeoutException)) for e in _exception_chain(exc))


def _is_connection_error(exc: BaseException) -> bool:
    return any(isinstance(e, (httpx.TransportError, ConnectionError)) for e in _exception_chain(exc))


def _has_marker(exc: BaseException, *markers: str) -> bool:
    return any(marker in str(e) for e in _exception_chain(exc) for marker in markers)


def _find(exc: BaseException, *types: type[BaseException]) -> bool:
    return any(isinstance(e, types) for e in _exception_chain(exc))


def is_retryable_error(exc: BaseException) -> bool:
    """True para errores transitorios: 429, 5xx, timeouts, red o salida estructurada inválida."""
    if isinstance(exc, GeminiConfigurationError):
        return False
    if _is_timeout(exc) or _is_connection_error(exc):
        return True
    if _find(exc, OutputParserException):
        return True
    status = _status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    model_errors = [e for e in _exception_chain(exc) if isinstance(e, ModelError)]
    if model_errors:
        return any(e.is_retryable for e in model_errors)
    return _has_marker(exc, *_RETRYABLE_MARKERS)


def classify_error(exc: BaseException, operation: str) -> GeminiError:
    """Convierte la excepción final en un `GeminiError` con mensaje accionable."""
    if isinstance(exc, GeminiError):
        return exc
    status = _status_code(exc)

    if _is_timeout(exc):
        return GeminiTimeoutError(
            f"Gemini no respondió a tiempo durante '{operation}' "
            f"(límite {settings.GEMINI_TIMEOUT_SECONDS:.0f}s por intento).",
            operation=operation,
        )
    if status == 429 or _find(exc, ModelRateLimitError) or _has_marker(exc, "RESOURCE_EXHAUSTED"):
        return GeminiRateLimitError(
            f"Se alcanzó el límite de peticiones de Gemini durante '{operation}'. "
            "Espera unos segundos y vuelve a intentarlo.",
            operation=operation,
        )
    if status in (401, 403) or _find(exc, ModelAuthenticationError, ModelPermissionDeniedError) or _has_marker(
        exc, "API_KEY_INVALID", "PERMISSION_DENIED"
    ):
        return GeminiConfigurationError(
            "GEMINI_API_KEY es inválida o no tiene permisos para este modelo.", operation=operation
        )
    if status == 404 or _find(exc, ModelNotFoundError):
        return GeminiConfigurationError(
            f"Modelo no disponible durante '{operation}'. Revisa GEMINI_MODEL "
            f"('{settings.GEMINI_MODEL}') y GEMINI_EMBEDDING_MODEL ('{settings.GEMINI_EMBEDDING_MODEL}'). "
            "Nota: 'models/text-embedding-004' fue retirado; usa 'models/gemini-embedding-001'.",
            operation=operation,
        )
    if (status is not None and status >= 500) or _is_connection_error(exc):
        return GeminiUnavailableError(
            f"Gemini no está disponible temporalmente durante '{operation}'.", operation=operation
        )
    if _find(exc, OutputParserException):
        return GeminiResponseError(
            f"Gemini devolvió una respuesta con formato inválido durante '{operation}'.", operation=operation
        )
    return GeminiResponseError(
        f"Error inesperado de Gemini durante '{operation}': {type(exc).__name__}.", operation=operation
    )


# ---------------------------------------------------------------------------
# Ejecución resiliente
# ---------------------------------------------------------------------------


def _log_retry(operation: str) -> Callable[[RetryCallState], None]:
    def _before_sleep(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0
        logger.warning(
            "Gemini '%s' falló (intento %d): %s. Reintentando en %.1fs",
            operation,
            state.attempt_number,
            type(exc).__name__ if exc else "?",
            wait,
        )

    return _before_sleep


async def call_gemini(operation: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Ejecuta `fn` con timeout por intento y reintentos exponenciales ante errores transitorios."""
    retrying = AsyncRetrying(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(max(1, settings.GEMINI_MAX_RETRIES + 1)),
        wait=wait_random_exponential(
            multiplier=settings.GEMINI_RETRY_INITIAL_SECONDS, max=settings.GEMINI_RETRY_MAX_SECONDS
        ),
        before_sleep=_log_retry(operation),
        reraise=True,
    )
    try:
        async for attempt in retrying:
            with attempt:
                return await asyncio.wait_for(fn(), timeout=settings.GEMINI_TIMEOUT_SECONDS)
    except Exception as exc:
        raise classify_error(exc, operation) from exc
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Fábricas de clientes
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = settings.GEMINI_API_KEY.get_secret_value().strip() if settings.GEMINI_API_KEY else ""
    if key in _PLACEHOLDER_KEYS:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY no está configurada. Añádela al archivo .env de la raíz del proyecto.",
            operation="configuración",
        )
    return key


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Modelo multimodal (visión + texto). Los reintentos los gestiona `call_gemini`."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=_api_key(),
        temperature=settings.GEMINI_TEMPERATURE,
        timeout=settings.GEMINI_TIMEOUT_SECONDS,
        max_retries=1,  # 1 = sin reintentos internos del SDK (0 activaría los suyos por defecto)
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL, google_api_key=_api_key())
