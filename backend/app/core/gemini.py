"""Clientes Gemini (LangChain) y capa de resiliencia común.

Todas las llamadas a Gemini (chat multimodal y embeddings) pasan por `call_gemini`, que:
- aplica un plazo por intento (`GEMINI_TIMEOUT_SECONDS`, enviado también a Gemini como X-Server-Timeout);
- reintenta con backoff exponencial + jitter (tenacity) ante 429, 5xx, timeouts y cortes de red,
  sin superar un presupuesto total (`GEMINI_MAX_TOTAL_SECONDS`);
- traduce el error final a una excepción tipada (`GeminiError`) que conserva el mensaje real de Gemini.

Las llamadas al SDK se hacen en modo **síncrono dentro de `asyncio.to_thread`** (`in_thread`): así
google-genai usa httpx y no aiohttp. Con aiohttp, langchain-core intenta leer `error.response.json()`
sin `await` al registrar errores (RuntimeWarning: coroutine 'ClientResponse.json' was never awaited).
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, TypeVar

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
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

from app.core.config import settings

logger = logging.getLogger("handwriting.gemini")

T = TypeVar("T")

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_MARKERS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL")
_PLACEHOLDER_KEYS = {"", "pega_aqui_tu_api_key", "your-api-key"}
_MAX_DETAIL_CHARS = 300


# ---------------------------------------------------------------------------
# Errores tipados
# ---------------------------------------------------------------------------


class GeminiError(Exception):
    """Fallo de Gemini tras agotar reintentos; `status_code` es el HTTP a devolver."""

    status_code = 502

    def __init__(self, message: str, *, operation: str, provider_detail: str | None = None) -> None:
        full = f"{message} Detalle de Gemini: {provider_detail}" if provider_detail else message
        super().__init__(full)
        self.message = full
        self.operation = operation
        self.provider_detail = provider_detail


class GeminiConfigurationError(GeminiError):
    """API key ausente/inválida, modelo inexistente o parámetro no soportado: reintentar no sirve."""

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
        # Solo atributos ya resueltos: nunca se llama a métodos del objeto `response`.
        response_status = getattr(getattr(err, "response", None), "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def describe_error(exc: BaseException) -> str:
    """Mensaje real y limpio del proveedor (p. ej. '504 DEADLINE_EXCEEDED: Deadline expired...')."""
    for err in _exception_chain(exc):
        code, status, message = getattr(err, "code", None), getattr(err, "status", None), getattr(err, "message", None)
        if isinstance(code, int) and isinstance(message, str) and message:  # google.genai.errors.APIError
            text = " ".join(str(p) for p in (code, status if isinstance(status, str) else None) if p)
            return _clip(f"{text}: {message}")
    for err in _exception_chain(exc):
        if text := str(err).strip():
            return _clip(f"{type(err).__name__}: {text}")
    return type(exc).__name__


def _clip(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= _MAX_DETAIL_CHARS else text[: _MAX_DETAIL_CHARS - 1] + "…"


def _is_timeout(exc: BaseException) -> bool:
    if any(isinstance(e, (TimeoutError, httpx.TimeoutException)) for e in _exception_chain(exc)):
        return True
    return _status_code(exc) == 504 or _has_marker(exc, "DEADLINE_EXCEEDED")


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
    """Convierte la excepción final en un `GeminiError` con mensaje accionable y el detalle real."""
    if isinstance(exc, GeminiError):
        return exc
    status = _status_code(exc)
    detail = describe_error(exc)

    def build(cls: type[GeminiError], message: str) -> GeminiError:
        return cls(message, operation=operation, provider_detail=detail)

    if _is_timeout(exc):
        return build(
            GeminiTimeoutError,
            f"Gemini superó el plazo de {settings.GEMINI_TIMEOUT_SECONDS:.0f}s por intento durante "
            f"'{operation}' (modelo '{settings.GEMINI_MODEL}'). Vuelve a intentarlo, aumenta "
            "GEMINI_TIMEOUT_SECONDS o usa un modelo más rápido.",
        )
    if status == 429 or _find(exc, ModelRateLimitError) or _has_marker(exc, "RESOURCE_EXHAUSTED"):
        return build(
            GeminiRateLimitError,
            f"Se alcanzó el límite de peticiones de Gemini durante '{operation}'. Espera unos segundos.",
        )
    if status in (401, 403) or _find(exc, ModelAuthenticationError, ModelPermissionDeniedError) or _has_marker(
        exc, "API_KEY_INVALID", "PERMISSION_DENIED"
    ):
        return build(GeminiConfigurationError, "GEMINI_API_KEY es inválida o no tiene permisos para este modelo.")
    if status == 404 or _find(exc, ModelNotFoundError):
        return build(
            GeminiConfigurationError,
            f"Modelo no disponible durante '{operation}'. Revisa GEMINI_MODEL ('{settings.GEMINI_MODEL}') y "
            f"GEMINI_EMBEDDING_MODEL ('{settings.GEMINI_EMBEDDING_MODEL}'). Nota: 'models/text-embedding-004' "
            "fue retirado; usa 'models/gemini-embedding-001'.",
        )
    if status == 400:
        return build(
            GeminiConfigurationError,
            f"Gemini rechazó la petición de '{operation}' por un parámetro inválido (revisa GEMINI_MODEL y "
            "GEMINI_THINKING_LEVEL).",
        )
    if (status is not None and status >= 500) or _is_connection_error(exc):
        return build(GeminiUnavailableError, f"Gemini no está disponible temporalmente durante '{operation}'.")
    if _find(exc, OutputParserException):
        return build(GeminiResponseError, f"Gemini devolvió una respuesta con formato inválido durante '{operation}'.")
    return build(GeminiResponseError, f"Error inesperado de Gemini durante '{operation}'.")


# ---------------------------------------------------------------------------
# Ejecución resiliente
# ---------------------------------------------------------------------------


def _log_retry(operation: str) -> Callable[[RetryCallState], None]:
    def _before_sleep(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0
        logger.warning(
            "Gemini '%s' falló (intento %d, %.1fs transcurridos): %s. Reintentando en %.1fs",
            operation,
            state.attempt_number,
            state.seconds_since_start or 0.0,
            describe_error(exc) if exc else "?",
            wait,
        )

    return _before_sleep


def in_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> Callable[[], Awaitable[T]]:
    """Fábrica para `call_gemini`: ejecuta una llamada síncrona del SDK en un hilo (httpx, sin aiohttp)."""
    return lambda: asyncio.to_thread(fn, *args, **kwargs)


async def call_gemini(operation: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Ejecuta `fn` con plazo por intento y reintentos exponenciales ante errores transitorios."""
    retrying = AsyncRetrying(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(max(1, settings.GEMINI_MAX_RETRIES + 1))
        | stop_after_delay(settings.GEMINI_MAX_TOTAL_SECONDS),
        wait=wait_random_exponential(
            multiplier=settings.GEMINI_RETRY_INITIAL_SECONDS, max=settings.GEMINI_RETRY_MAX_SECONDS
        ),
        before_sleep=_log_retry(operation),
        reraise=True,
    )
    # El SDK ya corta a GEMINI_TIMEOUT_SECONDS; el timeout local es una red de seguridad que además
    # nunca supera lo que queda del presupuesto total (GEMINI_MAX_TOTAL_SECONDS).
    loop = asyncio.get_running_loop()
    deadline = loop.time() + settings.GEMINI_MAX_TOTAL_SECONDS
    per_attempt = settings.GEMINI_TIMEOUT_SECONDS + settings.GEMINI_TIMEOUT_GRACE_SECONDS
    try:
        async for attempt in retrying:
            with attempt:
                remaining = max(0.01, deadline - loop.time())
                return await asyncio.wait_for(fn(), timeout=min(per_attempt, remaining))
    except Exception as exc:
        error = classify_error(exc, operation)
        logger.error("Gemini '%s' falló definitivamente (%s): %s", operation, type(error).__name__, error.message)
        raise error from exc
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Fábricas de clientes
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = settings.GEMINI_API_KEY.get_secret_value().strip() if settings.GEMINI_API_KEY else ""
    if key in _PLACEHOLDER_KEYS:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY no está configurada. Añádela al archivo .env.", operation="configuración"
        )
    return key


def _thinking_kwargs() -> dict[str, Any]:
    level = (settings.GEMINI_THINKING_LEVEL or "").strip().lower()
    # thinking_level solo existe en Gemini 3.x; en otras familias se deja el valor del modelo.
    if level and settings.GEMINI_MODEL.removeprefix("models/").startswith("gemini-3"):
        return {"thinking_level": level}
    return {}


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Modelo multimodal (visión + texto). Los reintentos los gestiona `call_gemini`."""
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=_api_key(),
        temperature=settings.GEMINI_TEMPERATURE,
        timeout=settings.GEMINI_TIMEOUT_SECONDS,
        max_retries=1,  # 1 = sin reintentos internos del SDK (0 activaría los suyos por defecto)
        **_thinking_kwargs(),
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL, google_api_key=_api_key())
