"""Modelos ORM (SQLAlchemy 2.x)."""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

DEFAULT_PROFILE_NAME = "default"
# `profile_name` se usa como carpeta en uploads/, así que solo se admiten caracteres seguros.
PROFILE_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return uuid.uuid4().hex


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    # El id coincide con el `image_id` usado en las rutas de uploads/.
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_uuid)
    profile_name: Mapped[str] = mapped_column(
        String(64),
        default=DEFAULT_PROFILE_NAME,
        server_default=DEFAULT_PROFILE_NAME,
        index=True,
        nullable=False,
    )
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    processed_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    transcribed_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    emotion: Mapped[str | None] = mapped_column(String(50), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_indicators_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    visual_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    rag_sources_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    execution_trace_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True, nullable=False
    )
