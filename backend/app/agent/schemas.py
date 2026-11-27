"""Esquemas Pydantic del agente: salida estructurada del VLM, traza y resultado final."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.mcp_server.server import ProfileContext
from app.rag.retriever import RetrievedChunk

EmotionLabel = Literal["neutral", "anxiety", "stress", "depression"]

EMOTION_LABELS_ES: dict[str, str] = {
    "neutral": "neutral",
    "anxiety": "ansiedad",
    "stress": "estrés",
    "depression": "ánimo bajo",
}


class IndicatorAssessment(BaseModel):
    score: int = Field(ge=0, le=100, description="Puntaje 0-100 según la escala del indicador")
    observation: str = Field(description="Observación breve (una frase) de lo que se ve en la imagen")

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, value: Any) -> int:
        try:
            return int(round(min(100.0, max(0.0, float(value)))))
        except (TypeError, ValueError):
            return 50


class VisualIndicators(BaseModel):
    pressure: IndicatorAssessment = Field(description="Presión: 0 trazo muy tenue, 100 muy marcado")
    rhythm: IndicatorAssessment = Field(description="Ritmo: 0 muy entrecortado, 100 muy fluido y regular")
    slant: IndicatorAssessment = Field(description="Inclinación: 0 izquierda, 50 vertical, 100 derecha")
    spacing: IndicatorAssessment = Field(description="Espaciado: 0 muy apretado, 100 muy amplio")
    stroke_variability: IndicatorAssessment = Field(
        description="Variabilidad del trazo: 0 muy uniforme, 100 muy variable"
    )

    def scores(self) -> dict[str, int]:
        return {name: getattr(self, name).score for name in type(self).model_fields}


class VisualAnalysisOutput(BaseModel):
    """Salida estructurada del análisis visual multimodal (Gemini)."""

    contains_handwriting: bool = Field(
        description=(
            "True solo si la imagen contiene escritura hecha a mano y legible. False para fotos en blanco, "
            "paisajes, objetos, personas, capturas de pantalla o texto impreso/tipografiado."
        )
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Si contains_handwriting es False, motivo breve en español; si es True, null.",
    )
    transcribed_text: str = Field(description="Transcripción literal del texto manuscrito ('' si no hay)")
    emotion: EmotionLabel = Field(description="Estado emocional orientativo más compatible")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza 0.0-1.0 (máx. 0.85 salvo evidencia clara)")
    visual_indicators: VisualIndicators
    visual_summary: str = Field(description="Resumen de 2-3 frases de los rasgos del trazo más relevantes")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number > 1.0:  # el modelo a veces devuelve porcentajes
            number /= 100.0
        return round(min(1.0, max(0.0, number)), 3)


class NodeTrace(BaseModel):
    node: str
    status: Literal["ok", "stopped", "degraded"]
    duration_ms: float
    detail: str | None = None


class AnalysisOutcome(BaseModel):
    """Resultado completo de una ejecución del grafo."""

    contains_handwriting: bool
    rejection_reason: str | None = None
    visual: VisualAnalysisOutput | None = None
    history: ProfileContext | None = None
    rag_chunks: list[RetrievedChunk] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    feedback: str | None = None
    risk_flags: list[str] = Field(default_factory=list)
    trace: list[NodeTrace] = Field(default_factory=list)
    total_duration_ms: float = 0.0
