"""Validación, almacenamiento y preprocesado de imágenes de escritura a mano."""

import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.models.db_models import DEFAULT_PROFILE_NAME, PROFILE_NAME_PATTERN

# Formato detectado por Pillow -> extensión con la que se guarda.
ALLOWED_FORMATS: dict[str, str] = {"PNG": "png", "JPEG": "jpg"}
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/pjpeg"}

# Protección frente a "decompression bombs" (≈ 8000x6000 px).
MAX_IMAGE_PIXELS = 50_000_000
_READ_CHUNK = 1024 * 1024
_PROFILE_NAME_RE = re.compile(PROFILE_NAME_PATTERN)


class ImageValidationError(Exception):
    """Archivo rechazado; se traduce a una respuesta HTTP 400 (ver handler en main.py)."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    format: str  # "PNG" | "JPEG"
    extension: str  # "png" | "jpg"
    width: int
    height: int


@dataclass(frozen=True)
class StoredImage:
    image_id: str
    original_path: Path
    processed_path: Path
    width: int
    height: int
    processed_width: int
    processed_height: int


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Lee el archivo por bloques y corta en cuanto supera `max_bytes`."""
    buffer = bytearray()
    while chunk := await file.read(_READ_CHUNK):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise ImageValidationError(
                f"La imagen supera el tamaño máximo de {max_bytes // (1024 * 1024)} MB"
            )
    return bytes(buffer)


def validate_image_bytes(
    data: bytes, content_type: str | None = None, max_bytes: int | None = None
) -> ValidatedImage:
    """Comprueba tamaño, tipo real (PNG/JPEG) y que la imagen se pueda decodificar."""
    max_bytes = max_bytes or settings.max_upload_size_bytes

    if not data:
        raise ImageValidationError("El archivo está vacío")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"La imagen supera el tamaño máximo de {max_bytes // (1024 * 1024)} MB"
        )

    declared = (content_type or "").split(";")[0].strip().lower()
    if declared and declared not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError("Formato no soportado. Solo se aceptan imágenes PNG o JPEG")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            detected = probe.format
            probe.verify()  # integridad estructural (p. ej. CRC en PNG)
        if detected not in ALLOWED_FORMATS:
            raise ImageValidationError("Formato no soportado. Solo se aceptan imágenes PNG o JPEG")
        if probe.width * probe.height > MAX_IMAGE_PIXELS:
            raise ImageValidationError("La imagen tiene demasiados píxeles")
        # verify() invalida el objeto: reabrimos y decodificamos por completo.
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
    except ImageValidationError:
        raise
    except UnidentifiedImageError:
        raise ImageValidationError("El archivo no es una imagen válida") from None
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ImageValidationError("La imagen está dañada o no se puede leer") from None

    if width == 0 or height == 0:
        raise ImageValidationError("La imagen no tiene dimensiones válidas")

    return ValidatedImage(
        data=data,
        format=detected,
        extension=ALLOWED_FORMATS[detected],
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Preprocesado
# ---------------------------------------------------------------------------


def _load_grayscale(data: bytes) -> np.ndarray:
    """Decodifica con Pillow respetando la orientación EXIF y aplanando transparencias sobre blanco."""
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba)
        rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _resize_max_dimension(image: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def preprocess_image(data: bytes, max_dim: int | None = None) -> np.ndarray:
    """Escala de grises -> redimensionado (máx. `max_dim` px) -> GaussianBlur -> umbral adaptativo."""
    gray = _load_grayscale(data)
    resized = _resize_max_dimension(gray, max_dim or settings.MAX_PROCESSED_DIMENSION)
    denoised = cv2.GaussianBlur(resized, (5, 5), 0)
    return cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # tamaño de bloque (impar)
        15,  # constante restada a la media ponderada
    )


# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------


def validate_profile_name(profile_name: str | None) -> str:
    """Devuelve un nombre de perfil seguro para usar como carpeta (evita path traversal)."""
    name = (profile_name or DEFAULT_PROFILE_NAME).strip()
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ImageValidationError(
            "profile_name inválido: usa 1-64 caracteres alfanuméricos, '-' o '_'"
        )
    return name


def save_and_preprocess(
    image: ValidatedImage, profile_name: str = DEFAULT_PROFILE_NAME, upload_root: Path | None = None
) -> StoredImage:
    """Guarda `uploads/{profile_name}/{image_id}.{ext}` y `uploads/{profile_name}/{image_id}_processed.png`."""
    root = Path(upload_root or settings.UPLOAD_DIR)
    profile_dir = root / validate_profile_name(profile_name)
    profile_dir.mkdir(parents=True, exist_ok=True)

    image_id = uuid.uuid4().hex
    original_path = profile_dir / f"{image_id}.{image.extension}"
    processed_path = profile_dir / f"{image_id}_processed.png"

    processed = preprocess_image(image.data)
    ok, encoded = cv2.imencode(".png", processed)
    if not ok:
        raise ImageValidationError(
            "No se pudo generar la imagen preprocesada", status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    try:
        original_path.write_bytes(image.data)
        processed_path.write_bytes(encoded.tobytes())
    except OSError:
        original_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        raise

    return StoredImage(
        image_id=image_id,
        original_path=original_path,
        processed_path=processed_path,
        width=image.width,
        height=image.height,
        processed_width=processed.shape[1],
        processed_height=processed.shape[0],
    )


def remove_stored_image(stored: StoredImage) -> None:
    """Elimina los archivos de una imagen guardada (p. ej. si el análisis se rechaza o falla)."""
    stored.original_path.unlink(missing_ok=True)
    stored.processed_path.unlink(missing_ok=True)


async def validate_upload(file: UploadFile, max_bytes: int | None = None) -> ValidatedImage:
    max_bytes = max_bytes or settings.max_upload_size_bytes
    data = await read_upload_limited(file, max_bytes)
    return await run_in_threadpool(validate_image_bytes, data, file.content_type, max_bytes)


async def process_upload(
    file: UploadFile, profile_name: str = DEFAULT_PROFILE_NAME, upload_root: Path | None = None
) -> StoredImage:
    """Valida un `UploadFile`, lo guarda y genera su versión preprocesada."""
    profile_name = validate_profile_name(profile_name)
    validated = await validate_upload(file)
    return await run_in_threadpool(save_and_preprocess, validated, profile_name, upload_root)
