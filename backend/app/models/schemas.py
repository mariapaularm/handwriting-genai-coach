"""Esquemas Pydantic v2 para requests y responses."""

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# `profile_name` se usa como carpeta en uploads/, así que solo se admiten caracteres seguros.
PROFILE_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    profile_name: str
    image_path: str
    processed_image_path: str | None = None
    transcribed_text: str | None = None
    emotion: str | None = None
    confidence: float | None = None
    visual_indicators: dict[str, Any] | None = Field(
        default=None, validation_alias=AliasChoices("visual_indicators", "visual_indicators_json")
    )
    agent_feedback: str | None = None
    rag_sources: list[Any] | None = Field(
        default=None, validation_alias=AliasChoices("rag_sources", "rag_sources_json")
    )
    created_at: datetime


class HistoryDeleteResponse(BaseModel):
    deleted: int
    profile_name: str | None = None
