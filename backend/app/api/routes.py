"""Endpoints abiertos (sin autenticación) para la demo."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete, select, true
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.database import get_db
from app.models.db_models import AnalysisRecord
from app.models.schemas import PROFILE_NAME_PATTERN, AnalysisResponse, HistoryDeleteResponse

router = APIRouter(tags=["history"])

DbSession = Annotated[AsyncSession, Depends(get_db)]
ProfileFilter = Annotated[
    str | None,
    Query(pattern=PROFILE_NAME_PATTERN, description="Filtra por perfil (opcional)"),
]


@router.get("/history", response_model=list[AnalysisResponse])
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


@router.delete("/history", response_model=HistoryDeleteResponse)
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
