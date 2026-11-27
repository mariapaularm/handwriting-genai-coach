"""Pruebas del núcleo: /health, /api/v1/history y validación de imágenes.

Se ejecutan con el plugin de pytest de AnyIO (incluido con FastAPI/Starlette),
usando una base de datos SQLite temporal por test.
"""

import io
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi import UploadFile
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.datastructures import Headers

from app.core.config import settings
from app.core.database import build_engine, get_db, init_db
from app.main import app, image_validation_exception_handler
from app.models.db_models import AnalysisRecord
from app.services.image_service import (
    ImageValidationError,
    process_upload,
    validate_image_bytes,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fixtures y utilidades
# ---------------------------------------------------------------------------


@dataclass
class Env:
    client: AsyncClient
    sessions: async_sessionmaker[AsyncSession]
    upload_root: Path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Env]:
    """App con una base SQLite y una carpeta uploads/ aisladas en `tmp_path`."""
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(settings, "UPLOAD_DIR", upload_root)

    test_engine = build_engine(f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    await init_db(test_engine)
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield Env(client=client, sessions=sessions, upload_root=upload_root)
    app.dependency_overrides.clear()
    await test_engine.dispose()


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (200, 100), mode: str = "RGB") -> bytes:
    color = (255, 255, 255, 0) if mode == "RGBA" else "white"
    img = Image.new(mode, size, color)
    # Unos "trazos" para que la umbralización tenga algo que detectar.
    for x in range(10, size[0] - 10):
        img.putpixel((x, size[1] // 2), (0, 0, 0, 255) if mode == "RGBA" else (0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def make_upload(data: bytes, filename: str, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


async def add_record(env: Env, profile_name: str, minutes_ago: int, **fields) -> AnalysisRecord:
    record = AnalysisRecord(
        profile_name=profile_name,
        image_path=fields.pop("image_path", f"uploads/{profile_name}/x.png"),
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        **fields,
    )
    async with env.sessions() as session:
        session.add(record)
        await session.commit()
    return record


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


async def test_health(env: Env) -> None:
    response = await env.client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_cors_allows_vite_origin(env: Env) -> None:
    response = await env.client.options(
        "/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ---------------------------------------------------------------------------
# /api/v1/history
# ---------------------------------------------------------------------------


async def test_history_is_empty_initially(env: Env) -> None:
    response = await env.client.get("/api/v1/history")
    assert response.status_code == 200
    assert response.json() == []


async def test_history_lists_newest_first(env: Env) -> None:
    oldest = await add_record(env, "default", minutes_ago=30)
    newest = await add_record(
        env,
        "maria",
        minutes_ago=1,
        emotion="ansiedad",
        confidence=0.81,
        visual_indicators_json={"pressure": 0.7},
        rag_sources_json=[{"source": "guia.pdf"}],
    )
    middle = await add_record(env, "default", minutes_ago=10)

    response = await env.client.get("/api/v1/history")
    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body] == [newest.id, middle.id, oldest.id]

    first = body[0]
    assert first["profile_name"] == "maria"
    assert first["emotion"] == "ansiedad"
    assert first["visual_indicators"] == {"pressure": 0.7}
    assert first["rag_sources"] == [{"source": "guia.pdf"}]


async def test_history_filters_by_profile_name(env: Env) -> None:
    await add_record(env, "default", minutes_ago=5)
    maria = await add_record(env, "maria", minutes_ago=3)

    response = await env.client.get("/api/v1/history", params={"profile_name": "maria"})
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [maria.id]

    invalid = await env.client.get("/api/v1/history", params={"profile_name": "../etc"})
    assert invalid.status_code == 422


async def test_delete_history_removes_all_records_and_files(env: Env) -> None:
    profile_dir = env.upload_root / "default"
    profile_dir.mkdir()
    original = profile_dir / "abc.png"
    processed = profile_dir / "abc_processed.png"
    original.write_bytes(b"x")
    processed.write_bytes(b"x")
    await add_record(
        env, "default", minutes_ago=2, image_path=str(original), processed_image_path=str(processed)
    )
    await add_record(env, "maria", minutes_ago=1)

    response = await env.client.delete("/api/v1/history")
    assert response.status_code == 200
    assert response.json()["deleted"] == 2
    assert not original.exists() and not processed.exists()

    assert (await env.client.get("/api/v1/history")).json() == []


async def test_delete_history_by_profile_keeps_other_profiles(env: Env) -> None:
    await add_record(env, "default", minutes_ago=2)
    maria = await add_record(env, "maria", minutes_ago=1)

    response = await env.client.delete("/api/v1/history", params={"profile_name": "default"})
    assert response.status_code == 200
    assert response.json() == {"deleted": 1, "profile_name": "default"}

    remaining = (await env.client.get("/api/v1/history")).json()
    assert [item["id"] for item in remaining] == [maria.id]


# ---------------------------------------------------------------------------
# Validación y preprocesado de imágenes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "content_type", "extension"),
    [("PNG", "image/png", "png"), ("JPEG", "image/jpeg", "jpg")],
)
async def test_valid_image_is_saved_under_profile_and_preprocessed(
    tmp_path: Path, fmt: str, content_type: str, extension: str
) -> None:
    data = make_image_bytes(fmt, size=(2048, 1024))
    upload = make_upload(data, f"muestra.{extension}", content_type)

    stored = await process_upload(upload, profile_name="maria", upload_root=tmp_path)

    assert stored.original_path == tmp_path / "maria" / f"{stored.image_id}.{extension}"
    assert stored.processed_path == tmp_path / "maria" / f"{stored.image_id}_processed.png"
    assert stored.original_path.read_bytes() == data

    with Image.open(stored.processed_path) as processed:
        assert processed.format == "PNG"
        assert processed.mode == "L"  # escala de grises / binarizada
        assert processed.size == (1024, 512)  # proporción 2:1 conservada
        assert set(np.unique(np.asarray(processed)).tolist()) <= {0, 255}  # umbralización adaptativa


async def test_default_profile_and_alpha_flattening(tmp_path: Path) -> None:
    data = make_image_bytes("PNG", size=(300, 120), mode="RGBA")
    stored = await process_upload(make_upload(data, "a.png", "image/png"), upload_root=tmp_path)
    assert stored.original_path.parent == tmp_path / "default"
    with Image.open(stored.processed_path) as processed:
        assert processed.size == (300, 120)  # no se agranda
        assert processed.getpixel((5, 5)) == 255  # el fondo transparente queda blanco


INVALID_FILES = {
    "no_es_imagen": (b"esto no es una imagen", "falso.png", "image/png"),
    "gif": (make_image_bytes("GIF"), "img.gif", "image/gif"),
    "gif_disfrazado_de_png": (make_image_bytes("GIF"), "img.png", "image/png"),
    "content_type_pdf": (make_image_bytes("PNG"), "doc.pdf", "application/pdf"),
    "mayor_10mb": (make_image_bytes("PNG") + b"\0" * (10 * 1024 * 1024), "grande.png", "image/png"),
    "jpeg_truncado": (make_image_bytes("JPEG", size=(400, 400))[:500], "roto.jpg", "image/jpeg"),
    "vacio": (b"", "vacio.png", "image/png"),
}


@pytest.mark.parametrize("case", INVALID_FILES.values(), ids=INVALID_FILES.keys())
async def test_invalid_files_are_rejected_with_400(tmp_path: Path, case: tuple[bytes, str, str]) -> None:
    data, filename, content_type = case
    with pytest.raises(ImageValidationError) as exc:
        await process_upload(make_upload(data, filename, content_type), upload_root=tmp_path)
    assert exc.value.status_code == 400

    # El handler global lo convierte en una respuesta HTTP 400 con `detail`.
    response = await image_validation_exception_handler(None, exc.value)
    assert response.status_code == 400
    assert b'"detail"' in response.body

    assert not any(tmp_path.rglob("*"))  # no se guardó nada


@pytest.mark.parametrize("profile_name", ["../fuera", "a/b", "con espacio", "x" * 65])
async def test_unsafe_profile_name_is_rejected(tmp_path: Path, profile_name: str) -> None:
    upload = make_upload(make_image_bytes("PNG"), "a.png", "image/png")
    with pytest.raises(ImageValidationError) as exc:
        await process_upload(upload, profile_name=profile_name, upload_root=tmp_path)
    assert exc.value.status_code == 400
    assert not any(tmp_path.rglob("*"))


def test_validate_image_bytes_accepts_png() -> None:
    validated = validate_image_bytes(make_image_bytes("PNG"), "image/png")
    assert (validated.format, validated.extension) == ("PNG", "png")
