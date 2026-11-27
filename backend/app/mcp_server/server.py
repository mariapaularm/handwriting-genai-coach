"""Servidor MCP (FastMCP) con herramientas sobre el historial de `AnalysisRecord`.

Dos formas de uso:
1. **Servidor MCP independiente** (stdio), p. ej. para MCP Inspector o Claude Desktop, desde `backend/`:
       python -m app.mcp_server.server
2. **En proceso desde el grafo de LangGraph**: `fetch_profile_context()` / `call_mcp_tool()` abren un
   cliente FastMCP en memoria (sin subprocesos ni red). Todo es `async` y usa SQLAlchemy async,
   así que no bloquea el event loop de FastAPI.
"""

import asyncio
from collections import Counter
from datetime import datetime
from typing import Annotated, Any, Literal

from fastmcp import Client, FastMCP
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core import database
from app.core.config import settings
from app.models.db_models import DEFAULT_PROFILE_NAME, PROFILE_NAME_PATTERN, AnalysisRecord

EMOTIONS = ("neutral", "anxiety", "stress", "depression")
ACTIVATION_EMOTIONS = {"anxiety", "stress"}
INDICATORS = ("pressure", "rhythm", "slant", "spacing", "stroke_variability")

ProfileNameArg = Annotated[str, Field(pattern=PROFILE_NAME_PATTERN, description="Perfil a consultar")]

mcp = FastMCP(
    "HandwritingWellnessMCP",
    instructions=(
        "Herramientas de solo lectura sobre el historial de análisis de escritura manuscrita "
        "y la escala de referencia de indicadores del trazo. Uso orientativo y preventivo."
    ),
)


# ---------------------------------------------------------------------------
# Esquemas de salida
# ---------------------------------------------------------------------------


class AnalysisSummary(BaseModel):
    id: str
    created_at: datetime
    emotion: str | None
    confidence: float | None
    visual_summary: str | None
    indicator_scores: dict[str, int] = Field(default_factory=dict)


class RecentAnalysesResult(BaseModel):
    profile_name: str
    count: int
    analyses: list[AnalysisSummary]


class EmotionalTrendResult(BaseModel):
    profile_name: str
    window_size: int
    total_analyses: int
    distribution: dict[str, int]
    percentages: dict[str, float]
    dominant_emotion: str | None
    stress_anxiety_count: int
    stress_anxiety_ratio: float
    consecutive_stress_anxiety: int = Field(description="Racha desde el análisis más reciente")
    depression_count: int
    recurrent_stress_anxiety: bool
    alert_level: Literal["none", "watch", "elevated"]
    summary: str


class ScoreBand(BaseModel):
    min: int
    max: int
    label: Literal["bajo", "medio", "alto"]
    interpretation: str


class IndicatorRule(BaseModel):
    indicator: str
    name_es: str
    observed_in_image: str
    scale: str
    bands: list[ScoreBand]


class GraphologyReferenceRules(BaseModel):
    scale: str
    indicators: list[IndicatorRule]
    combined_patterns: dict[str, str]
    caveats: list[str]


class ProfileContext(BaseModel):
    """Contexto que el grafo obtiene vía MCP para la síntesis."""

    recent: RecentAnalysesResult
    trend: EmotionalTrendResult
    rules: GraphologyReferenceRules


# ---------------------------------------------------------------------------
# Escala de referencia (coherente con knowledge_base/04_indicadores_grafologicos_trazo.md)
# ---------------------------------------------------------------------------


def _bands(low: str, mid: str, high: str) -> list[ScoreBand]:
    return [
        ScoreBand(min=0, max=33, label="bajo", interpretation=low),
        ScoreBand(min=34, max=66, label="medio", interpretation=mid),
        ScoreBand(min=67, max=100, label="alto", interpretation=high),
    ]


GRAPHOLOGY_RULES = GraphologyReferenceRules(
    scale="Puntaje 0-100 por indicador. Bandas: bajo 0-33, medio 34-66, alto 67-100.",
    indicators=[
        IndicatorRule(
            indicator="pressure",
            name_es="Presión",
            observed_in_image="Grosor y oscuridad del trazo, carga de tinta, surcos en el papel.",
            scale="0 = trazo muy ligero o tenue; 100 = muy marcado, grueso u oscuro.",
            bands=_bands(
                "Trazo tenue o fino. Puede acompañar cansancio, baja energía o desánimo; depende también del bolígrafo.",
                "Presión estable y moderada; patrón habitual sin señales destacables.",
                "Trazo grueso, oscuro o con surcos. Puede asociarse a tensión, activación o urgencia.",
            ),
        ),
        IndicatorRule(
            indicator="rhythm",
            name_es="Ritmo",
            observed_in_image="Fluidez y regularidad del movimiento, continuidad, retoques y cortes.",
            scale="0 = muy entrecortado o irregular; 100 = muy fluido y regular.",
            bands=_bands(
                "Cortes, retoques o cambios bruscos de velocidad/tamaño. Puede reflejar tensión, prisa o cansancio.",
                "Alguna irregularidad puntual; habitual en escritura espontánea.",
                "Escritura fluida y homogénea; sugiere estabilidad al escribir.",
            ),
        ),
        IndicatorRule(
            indicator="slant",
            name_es="Inclinación",
            observed_in_image="Ángulo de los trazos verticales respecto a la línea base.",
            scale="0 = muy inclinada a la izquierda; 50 = vertical; 100 = muy inclinada a la derecha.",
            bands=_bands(
                "Inclinación a la izquierda; rasgo personal (lateralidad, aprendizaje). Relevante solo si cambia.",
                "Vertical o con inclinación ligera; rango habitual.",
                "Inclinación marcada a la derecha; por sí sola no indica un estado emocional. Valorar inestabilidad.",
            ),
        ),
        IndicatorRule(
            indicator="spacing",
            name_es="Espaciado",
            observed_in_image="Distancia entre letras, palabras y líneas; uso de márgenes.",
            scale="0 = muy apretado; 100 = muy amplio o disperso.",
            bands=_bands(
                "Escritura apretada y márgenes reducidos. Puede acompañar presión, prisa o sobrecarga.",
                "Espaciado regular y legible.",
                "Espacios amplios o irregulares. Puede acompañar lentitud, distracción o retraimiento.",
            ),
        ),
        IndicatorRule(
            indicator="stroke_variability",
            name_es="Variabilidad del trazo",
            observed_in_image="Cambios de tamaño, grosor, alineación con la línea base y temblor.",
            scale="0 = muy uniforme; 100 = muy variable.",
            bands=_bands(
                "Trazo uniforme en tamaño, grosor y alineación.",
                "Variaciones moderadas, normales en escritura espontánea.",
                "Cambios notables o temblor. Puede relacionarse con activación, fatiga o nerviosismo.",
            ),
        ),
    ],
    combined_patterns={
        "activation": "Presión alta + variabilidad alta + ritmo bajo + espaciado bajo: compatible con estrés o ansiedad.",
        "low_energy": "Presión baja + ritmo lento + espaciado amplio + líneas descendentes: compatible con ánimo bajo.",
        "stable": "Valores en banda media, ritmo fluido y variabilidad baja o media: patrón neutral.",
    },
    caveats=[
        "Son estimaciones a partir de una foto: influyen bolígrafo, papel, luz, postura y prisa.",
        "Ningún rasgo aislado basta: se requieren al menos dos indicios coherentes con el contenido del texto.",
        "La grafología tradicional no tiene validez diagnóstica; usar lenguaje condicional y nunca diagnosticar.",
        "La tendencia del historial de la misma persona es más informativa que un único análisis.",
    ],
)


# ---------------------------------------------------------------------------
# Implementaciones (async, reutilizables fuera de MCP)
# ---------------------------------------------------------------------------


def _indicator_scores(record: AnalysisRecord) -> dict[str, int]:
    scores: dict[str, int] = {}
    for name, value in (record.visual_indicators_json or {}).items():
        score = value.get("score") if isinstance(value, dict) else value
        if isinstance(score, (int, float)):
            scores[name] = int(score)
    return scores


async def _fetch_records(profile_name: str, limit: int) -> list[AnalysisRecord]:
    async with database.AsyncSessionLocal() as session:
        result = await session.scalars(
            select(AnalysisRecord)
            .where(AnalysisRecord.profile_name == profile_name)
            .order_by(AnalysisRecord.created_at.desc())
            .limit(limit)
        )
        return list(result)


async def recent_analyses_impl(profile_name: str = DEFAULT_PROFILE_NAME, limit: int = 5) -> RecentAnalysesResult:
    records = await _fetch_records(profile_name, limit)
    return RecentAnalysesResult(
        profile_name=profile_name,
        count=len(records),
        analyses=[
            AnalysisSummary(
                id=r.id,
                created_at=r.created_at,
                emotion=r.emotion,
                confidence=r.confidence,
                visual_summary=r.visual_summary,
                indicator_scores=_indicator_scores(r),
            )
            for r in records
        ],
    )


def compute_emotional_trend(profile_name: str, emotions: list[str | None], window: int) -> EmotionalTrendResult:
    """Distribución y recurrencia a partir de emociones ordenadas de la más reciente a la más antigua."""
    valid = [e for e in emotions if e in EMOTIONS]
    counts = Counter(valid)
    total = len(valid)
    distribution = {emotion: counts.get(emotion, 0) for emotion in EMOTIONS}
    percentages = {e: round(100 * c / total, 1) if total else 0.0 for e, c in distribution.items()}

    activation = sum(distribution[e] for e in ACTIVATION_EMOTIONS)
    ratio = round(activation / total, 3) if total else 0.0
    streak = 0
    for emotion in valid:
        if emotion not in ACTIVATION_EMOTIONS:
            break
        streak += 1
    depression = distribution["depression"]

    recurrent = (activation >= 3 and ratio >= 0.5) or streak >= 3
    if streak >= 3 or (recurrent and ratio >= 0.7) or depression >= 3:
        alert: Literal["none", "watch", "elevated"] = "elevated"
    elif recurrent or activation >= 2 or depression >= 2:
        alert = "watch"
    else:
        alert = "none"

    dominant = max(distribution, key=lambda e: distribution[e]) if total else None

    if total == 0:
        summary = "Sin análisis previos para este perfil: no hay tendencia que comparar."
    else:
        summary = (
            f"En los últimos {total} análisis predomina '{dominant}'. "
            f"Estrés/ansiedad aparecen en {activation} ({ratio:.0%}); racha actual: {streak}. "
        )
        if recurrent:
            summary += "Se detecta recurrencia de estrés/ansiedad."
        elif depression >= 2:
            summary += "Se repiten indicios de ánimo bajo."
        else:
            summary += "No se detecta recurrencia significativa."

    return EmotionalTrendResult(
        profile_name=profile_name,
        window_size=window,
        total_analyses=total,
        distribution=distribution,
        percentages=percentages,
        dominant_emotion=dominant,
        stress_anxiety_count=activation,
        stress_anxiety_ratio=ratio,
        consecutive_stress_anxiety=streak,
        depression_count=depression,
        recurrent_stress_anxiety=recurrent,
        alert_level=alert,
        summary=summary,
    )


async def emotional_trend_impl(profile_name: str = DEFAULT_PROFILE_NAME, window: int = 10) -> EmotionalTrendResult:
    records = await _fetch_records(profile_name, window)
    return compute_emotional_trend(profile_name, [r.emotion for r in records], window)


# ---------------------------------------------------------------------------
# Herramientas MCP
# ---------------------------------------------------------------------------


@mcp.tool
async def get_recent_analyses(
    profile_name: ProfileNameArg = DEFAULT_PROFILE_NAME,
    limit: Annotated[int, Field(ge=1, le=50)] = 5,
) -> RecentAnalysesResult:
    """Devuelve los últimos análisis del perfil (más reciente primero) con emoción, confianza e indicadores."""
    return await recent_analyses_impl(profile_name, limit)


@mcp.tool
async def get_emotional_trend(
    profile_name: ProfileNameArg = DEFAULT_PROFILE_NAME,
    window: Annotated[int, Field(ge=1, le=100)] = 10,
) -> EmotionalTrendResult:
    """Distribución de emociones recientes del perfil y detección de recurrencia de estrés/ansiedad."""
    return await emotional_trend_impl(profile_name, window)


@mcp.tool
async def get_graphology_reference_rules() -> GraphologyReferenceRules:
    """Escala estándar (0-100, bandas bajo/medio/alto) para interpretar los indicadores del trazo."""
    return GRAPHOLOGY_RULES


# ---------------------------------------------------------------------------
# Helpers para el grafo (cliente MCP en memoria)
# ---------------------------------------------------------------------------


async def call_mcp_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Invoca una herramienta del servidor por protocolo MCP (transporte en memoria)."""
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments or {})
    return result.structured_content or {}


async def fetch_profile_context(
    profile_name: str = DEFAULT_PROFILE_NAME, limit: int | None = None, window: int | None = None
) -> ProfileContext:
    """Ejecuta las 3 herramientas en una sola sesión MCP y valida sus salidas."""
    async with Client(mcp) as client:
        recent, trend, rules = await asyncio.gather(
            client.call_tool(
                "get_recent_analyses",
                {"profile_name": profile_name, "limit": limit or settings.MCP_HISTORY_LIMIT},
            ),
            client.call_tool(
                "get_emotional_trend",
                {"profile_name": profile_name, "window": window or settings.MCP_TREND_WINDOW},
            ),
            client.call_tool("get_graphology_reference_rules", {}),
        )
    return ProfileContext(
        recent=RecentAnalysesResult.model_validate(recent.structured_content),
        trend=EmotionalTrendResult.model_validate(trend.structured_content),
        rules=GraphologyReferenceRules.model_validate(rules.structured_content),
    )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(database.init_db())
    mcp.run()
