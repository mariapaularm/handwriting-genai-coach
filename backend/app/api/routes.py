"""Endpoints abiertos (sin autenticación) para la demo."""

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import delete, select, true
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.agent.graph import run_analysis
from app.agent.schemas import AnalysisOutcome
from app.core.config import settings
from app.core.database import get_db
from app.models.db_models import DEFAULT_PROFILE_NAME, AnalysisRecord
from app.models.schemas import (
    PROFILE_NAME_PATTERN,
    AnalysisResponse,
    AnalyzeResponse,
    ErrorResponse,
    HistoryContext,
    HistoryDeleteResponse,
    NoHandwritingResponse,
    RagSource,
)
from app.services.image_service import StoredImage, process_upload, remove_stored_image

router = APIRouter()

DbSession = Annotated[AsyncSession, Depends(get_db)]
ProfileFilter = Annotated[
    str | None,
    Query(pattern=PROFILE_NAME_PATTERN, description="Filtra por perfil (opcional)"),
]


@router.get("/history", response_model=list[AnalysisResponse], tags=["history"])
async def list_history(
    db: DbSession,
    profile_name: ProfileFilter = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AnalysisRecord]:
    """Lista los análisis del más reciente al más antiguo."""
    query = select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
    if profile_name is not None:
        query = query.where(AnalysisRecord.profile_name == profile_name)
    result = await db.scalars(query.limit(limit).offset(offset))
    return list(result)


def _remove_files(paths: list[str]) -> None:
    """Borra las imágenes asociadas, solo si están dentro de UPLOAD_DIR."""
    upload_root = settings.UPLOAD_DIR.resolve()
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_relative_to(upload_root):
            path.unlink(missing_ok=True)


@router.delete("/history", response_model=HistoryDeleteResponse, tags=["history"])
async def clear_history(db: DbSession, profile_name: ProfileFilter = None) -> HistoryDeleteResponse:
    """Borra los análisis (todos, o solo los de `profile_name`) y sus imágenes en uploads/."""
    condition = AnalysisRecord.profile_name == profile_name if profile_name is not None else true()

    rows = await db.execute(
        select(AnalysisRecord.image_path, AnalysisRecord.processed_image_path).where(condition)
    )
    files = [path for row in rows for path in row if path]

    result = await db.execute(delete(AnalysisRecord).where(condition))
    await db.commit()

    await run_in_threadpool(_remove_files, files)
    return HistoryDeleteResponse(deleted=result.rowcount or 0, profile_name=profile_name)


# ---------------------------------------------------------------------------
# POST /api/v1/analyze
# ---------------------------------------------------------------------------

_SNIPPET_CHARS = 400


def _rag_sources(outcome: AnalysisOutcome) -> list[RagSource]:
    cited = set(outcome.cited_sources)
    sources = []
    for chunk in outcome.rag_chunks:
        snippet = chunk.content[:_SNIPPET_CHARS].rstrip()
        if len(chunk.content) > _SNIPPET_CHARS:
            snippet += "..."
        sources.append(
            RagSource(
                source=chunk.source,
                title=chunk.title,
                section=chunk.section,
                score=chunk.score,
                snippet=snippet,
                cited=chunk.source in cited,
            )
        )
    return sources


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    summary="Analiza una muestra de escritura manuscrita",
    responses={
        400: {"model": ErrorResponse, "description": "Archivo inválido (no PNG/JPEG, > 10 MB o dañado)"},
        422: {"model": NoHandwritingResponse, "description": "No se detectó escritura manuscrita"},
        429: {"model": ErrorResponse, "description": "Límite de peticiones de Gemini agotado tras reintentos"},
        502: {"model": ErrorResponse, "description": "Respuesta inválida de Gemini"},
        503: {"model": ErrorResponse, "description": "Gemini no configurado o no disponible"},
        504: {"model": ErrorResponse, "description": "Timeout de Gemini"},
    },
)
async def analyze(
    db: DbSession,
    file: Annotated[UploadFile, File(description="Imagen PNG o JPEG (máx. 10 MB) con texto manuscrito")],
    profile_name: Annotated[
        str, Form(pattern=PROFILE_NAME_PATTERN, description="Perfil al que se asocia el análisis")
    ] = DEFAULT_PROFILE_NAME,
) -> AnalyzeResponse | JSONResponse:
    started = time.perf_counter()
    stored: StoredImage = await process_upload(file, profile_name)

    try:
        outcome = await run_analysis(profile_name, stored.original_path, stored.processed_path)
    except Exception:
        await run_in_threadpool(remove_stored_image, stored)
        raise  # GeminiError -> handler global (429/502/503/504)

    if not outcome.contains_handwriting or outcome.visual is None or outcome.feedback is None:
        await run_in_threadpool(remove_stored_image, stored)
        reason = outcome.rejection_reason or "No se detectó escritura manuscrita legible."
        body = NoHandwritingResponse(
            detail=(
                "No se detectó escritura manuscrita en la imagen. Sube una foto nítida de un texto "
                f"escrito a mano. Motivo: {reason}"
            ),
            reason=reason,
            trace=outcome.trace,
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=body.model_dump(mode="json"))

    visual = outcome.visual
    rag_sources = _rag_sources(outcome)

    record = AnalysisRecord(
        id=stored.image_id,
        profile_name=profile_name,
        image_path=stored.original_path.as_posix(),
        processed_image_path=stored.processed_path.as_posix(),
        transcribed_text=visual.transcribed_text,
        emotion=visual.emotion,
        confidence=visual.confidence,
        visual_indicators_json=visual.visual_indicators.model_dump(mode="json"),
        visual_summary=visual.visual_summary,
        agent_feedback=outcome.feedback,
        rag_sources_json=[s.model_dump(mode="json") for s in rag_sources],
        execution_trace_json=[t.model_dump(mode="json") for t in outcome.trace],
    )
    db.add(record)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        await run_in_threadpool(remove_stored_image, stored)
        raise

    history = outcome.history
    return AnalyzeResponse(
        id=record.id,
        profile_name=profile_name,
        created_at=record.created_at,
        transcribed_text=visual.transcribed_text,
        emotion=visual.emotion,
        confidence=visual.confidence,
        visual_indicators=visual.visual_indicators,
        visual_summary=visual.visual_summary,
        biomarkers=visual.visual_indicators.scores(),
        feedback=outcome.feedback,
        rag_sources=rag_sources,
        cited_sources=outcome.cited_sources,
        history=HistoryContext(
            previous_analyses=history.recent.count if history else 0,
            trend=history.trend if history else None,
        ),
        risk_flags=outcome.risk_flags,
        trace=outcome.trace,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        models={"vision": settings.GEMINI_MODEL, "embeddings": settings.GEMINI_EMBEDDING_MODEL},
        image_path=record.image_path,
        processed_image_path=record.processed_image_path or "",
    )
