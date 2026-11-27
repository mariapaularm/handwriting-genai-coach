"""Esquemas Pydantic v2 para requests y responses."""

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.agent.schemas import EmotionLabel, NodeTrace, VisualIndicators
from app.mcp_server.server import EmotionalTrendResult
from app.models.db_models import PROFILE_NAME_PATTERN

__all__ = [
    "PROFILE_NAME_PATTERN",
    "AnalysisResponse",
    "AnalyzeResponse",
    "ErrorResponse",
    "HistoryDeleteResponse",
    "NoHandwritingResponse",
    "RagSource",
]


class AnalysisResponse(BaseModel):
    """Registro de historial tal como se guarda en SQLite."""

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
    visual_summary: str | None = None
    agent_feedback: str | None = None
    rag_sources: list[Any] | None = Field(
        default=None, validation_alias=AliasChoices("rag_sources", "rag_sources_json")
    )
    trace: list[Any] | None = Field(
        default=None, validation_alias=AliasChoices("trace", "execution_trace_json")
    )
    created_at: datetime


class HistoryDeleteResponse(BaseModel):
    deleted: int
    profile_name: str | None = None


# ---------------------------------------------------------------------------
# POST /api/v1/analyze
# ---------------------------------------------------------------------------


class RagSource(BaseModel):
    source: str = Field(description="Documento de knowledge_base/")
    title: str
    section: str | None = None
    score: float
    snippet: str
    cited: bool = Field(description="True si la retroalimentación cita este documento")


class HistoryContext(BaseModel):
    previous_analyses: int
    trend: EmotionalTrendResult | None = None


class AnalyzeResponse(BaseModel):
    id: str
    profile_name: str
    created_at: datetime
    contains_handwriting: Literal[True] = True
    transcribed_text: str
    emotion: EmotionLabel
    confidence: float
    visual_indicators: VisualIndicators
    visual_summary: str
    biomarkers: dict[str, int] = Field(description="Puntajes 0-100 planos (compatibles con el frontend)")
    feedback: str
    rag_sources: list[RagSource]
    cited_sources: list[str]
    history: HistoryContext
    risk_flags: list[str]
    trace: list[NodeTrace]
    latency_ms: float
    models: dict[str, str]
    image_path: str
    processed_image_path: str


class NoHandwritingResponse(BaseModel):
    detail: str
    contains_handwriting: Literal[False] = False
    reason: str
    trace: list[NodeTrace]


class ErrorResponse(BaseModel):
    detail: str
    operation: str | None = None
