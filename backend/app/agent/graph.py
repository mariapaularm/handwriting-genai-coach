"""Grafo agéntico multimodal (LangGraph).

    START → visual_analysis ─┬─(sin escritura)──────────────────────────→ END
                             └─→ mcp_history ─┐
                             └─→ rag_retrieval ┴─→ synthesis_guardrail → END

- `visual_analysis` y `synthesis_guardrail` son críticos: si Gemini falla tras los reintentos se
  propaga un `GeminiError` (la API lo traduce a 429/503/504).
- `mcp_history` y `rag_retrieval` se ejecutan en paralelo y son degradables: si fallan, el flujo
  continúa y la traza lo registra con `status="degraded"`.
"""

import asyncio
import base64
import functools
import io
import logging
import operator
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageOps

from app.agent import prompts
from app.agent.schemas import AnalysisOutcome, NodeTrace, VisualAnalysisOutput
from app.core import gemini
from app.core.config import settings
from app.mcp_server import server as mcp_server
from app.mcp_server.server import ProfileContext
from app.rag import retriever
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger("handwriting.agent")

NODE_VISUAL = "visual_analysis"
NODE_MCP = "mcp_history"
NODE_RAG = "rag_retrieval"
NODE_SYNTHESIS = "synthesis_guardrail"

_DEFAULT_REJECTION = "No se detectó escritura manuscrita legible en la imagen."
_CITATION_RE = re.compile(r"\[([\w\-]+\.md)\]")
_CRISIS_PATTERNS = [
    r"suicid",
    r"quitarme la vida",
    r"no quiero (seguir )?vivir",
    r"quiero morir(me)?",
    r"matarme",
    r"hacerme dano",
    r"autolesion",
    r"no vale la pena vivir",
    r"acabar con todo",
    r"desaparecer para siempre",
]


class AgentState(TypedDict, total=False):
    profile_name: str
    original_image_path: str
    processed_image_path: str
    visual: VisualAnalysisOutput
    stop_reason: str | None
    history: ProfileContext | None
    rag_chunks: list[RetrievedChunk]
    feedback: str
    cited_sources: list[str]
    risk_flags: list[str]
    trace: Annotated[list[NodeTrace], operator.add]


NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]


def traced(node_name: str) -> Callable[[NodeFn], NodeFn]:
    """Mide la duración del nodo y añade una entrada a `trace` (claves especiales `_status`/`_detail`)."""

    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: AgentState) -> dict[str, Any]:
            started = time.perf_counter()
            update = await fn(state)
            update["trace"] = [
                NodeTrace(
                    node=node_name,
                    status=update.pop("_status", "ok"),
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    detail=update.pop("_detail", None),
                )
            ]
            return update

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _to_data_url(path: str, max_dim: int | None, fmt: str) -> str:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba)
        img = img.convert("RGB" if fmt == "JPEG" else "L")
        if max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def encode_images(original_path: str, processed_path: str) -> tuple[str, str]:
    """Original reescalada a JPEG (≤ VLM_MAX_IMAGE_DIMENSION) + preprocesada en PNG."""
    return (
        _to_data_url(original_path, settings.VLM_MAX_IMAGE_DIMENSION, "JPEG"),
        _to_data_url(processed_path, None, "PNG"),
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def detect_risk_signals(text: str) -> list[str]:
    """Guardrail determinista: detecta expresiones de riesgo de autolesión en el texto transcrito."""
    normalized = _normalize(text)
    return [pattern for pattern in _CRISIS_PATTERNS if re.search(pattern, normalized)]


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if callable(text) and not isinstance(text, str):  # langchain-core < 1.0: .text() era un método
        text = text()
    if text is None:
        text = message.content if isinstance(getattr(message, "content", None), str) else ""
    return str(text).strip()


def apply_guardrails(feedback: str, risk_flags: list[str], available_sources: set[str]) -> tuple[str, list[str]]:
    """Añade aviso obligatorio (y de crisis si aplica) y valida que las citas existan en el RAG."""
    cited: list[str] = []
    for source in _CITATION_RE.findall(feedback):
        if source in available_sources and source not in cited:
            cited.append(source)
    # Elimina citas a documentos que no fueron recuperados (alucinaciones de fuente).
    feedback = _CITATION_RE.sub(lambda m: m.group(0) if m.group(1) in available_sources else "", feedback)

    parts = []
    if risk_flags:
        parts.append(prompts.CRISIS_MESSAGE)
    parts.append(feedback.strip())
    parts.append(prompts.WELLNESS_DISCLAIMER)
    return "\n\n".join(p for p in parts if p), cited


def _invoke_structured(runnable: Any, messages: list[BaseMessage]) -> VisualAnalysisOutput:
    """Llamada síncrona (se ejecuta en un hilo vía `gemini.in_thread`)."""
    result = runnable.invoke(messages)
    if result is None:
        raise OutputParserException("Gemini no devolvió una salida estructurada.")
    return result if isinstance(result, VisualAnalysisOutput) else VisualAnalysisOutput.model_validate(result)


def _invoke_text(model: Any, messages: list[BaseMessage]) -> str:
    """Llamada síncrona (se ejecuta en un hilo vía `gemini.in_thread`)."""
    text = _message_text(model.invoke(messages))
    if not text:
        raise OutputParserException("Gemini devolvió una retroalimentación vacía.")
    return text


# ---------------------------------------------------------------------------
# Nodos
# ---------------------------------------------------------------------------


@traced(NODE_VISUAL)
async def visual_analysis_node(state: AgentState) -> dict[str, Any]:
    original_url, processed_url = await asyncio.to_thread(
        encode_images, state["original_image_path"], state["processed_image_path"]
    )
    messages = prompts.build_visual_messages(original_url, processed_url)
    structured = gemini.get_chat_model().with_structured_output(VisualAnalysisOutput)
    visual = await gemini.call_gemini("análisis visual", gemini.in_thread(_invoke_structured, structured, messages))

    if not visual.contains_handwriting or not visual.transcribed_text.strip():
        reason = (visual.rejection_reason or "").strip() or _DEFAULT_REJECTION
        return {"visual": visual, "stop_reason": reason, "_status": "stopped", "_detail": reason}

    detail = f"emoción={visual.emotion} confianza={visual.confidence:.2f}"
    return {"visual": visual, "stop_reason": None, "_detail": detail}


@traced(NODE_MCP)
async def mcp_history_node(state: AgentState) -> dict[str, Any]:
    try:
        history = await mcp_server.fetch_profile_context(state["profile_name"])
    except Exception as exc:  # degradable: el análisis sigue sin historial
        logger.warning("MCP no disponible: %s", exc, exc_info=True)
        return {"history": None, "_status": "degraded", "_detail": f"Historial no disponible ({type(exc).__name__})"}
    trend = history.trend
    detail = f"{history.recent.count} análisis previos; alerta={trend.alert_level}; recurrencia={trend.recurrent_stress_anxiety}"
    return {"history": history, "_detail": detail}


@traced(NODE_RAG)
async def rag_retrieval_node(state: AgentState) -> dict[str, Any]:
    visual = state["visual"]
    try:
        chunks = await retriever.retrieve_wellness_context(visual.emotion, visual.visual_summary, k=settings.RAG_TOP_K)
    except Exception as exc:  # degradable: la síntesis usa recomendaciones generales
        message = getattr(exc, "message", None) or gemini.describe_error(exc)
        logger.warning("RAG no disponible: %s", message, exc_info=True)
        return {"rag_chunks": [], "_status": "degraded", "_detail": f"RAG no disponible: {message}"}
    sources = sorted({c.source for c in chunks})
    return {"rag_chunks": chunks, "_detail": f"{len(chunks)} fragmentos de {', '.join(sources) or 'ninguna fuente'}"}


@traced(NODE_SYNTHESIS)
async def synthesis_guardrail_node(state: AgentState) -> dict[str, Any]:
    visual = state["visual"]
    chunks = state.get("rag_chunks") or []
    risk_flags = detect_risk_signals(visual.transcribed_text)

    messages = prompts.build_synthesis_messages(visual, state.get("history"), chunks, risk_flags)
    model = gemini.get_chat_model()
    draft = await gemini.call_gemini(
        "síntesis de retroalimentación", gemini.in_thread(_invoke_text, model, messages)
    )

    feedback, cited = apply_guardrails(draft, risk_flags, {c.source for c in chunks})
    detail = f"{len(cited)} fuentes citadas" + ("; protocolo de crisis activado" if risk_flags else "")
    return {"feedback": feedback, "cited_sources": cited, "risk_flags": risk_flags, "_detail": detail}


def route_after_visual(state: AgentState) -> str | list[str]:
    return END if state.get("stop_reason") else [NODE_MCP, NODE_RAG]


# ---------------------------------------------------------------------------
# Construcción y ejecución
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node(NODE_VISUAL, visual_analysis_node)
    graph.add_node(NODE_MCP, mcp_history_node)
    graph.add_node(NODE_RAG, rag_retrieval_node)
    graph.add_node(NODE_SYNTHESIS, synthesis_guardrail_node)

    graph.add_edge(START, NODE_VISUAL)
    graph.add_conditional_edges(NODE_VISUAL, route_after_visual, [NODE_MCP, NODE_RAG, END])
    graph.add_edge([NODE_MCP, NODE_RAG], NODE_SYNTHESIS)  # espera a ambas ramas
    graph.add_edge(NODE_SYNTHESIS, END)
    return graph.compile()


@functools.lru_cache(maxsize=1)
def get_analysis_graph() -> Any:
    return build_graph()


async def run_analysis(profile_name: str, original_image_path: Path | str, processed_image_path: Path | str) -> AnalysisOutcome:
    started = time.perf_counter()
    final: AgentState = await get_analysis_graph().ainvoke(
        {
            "profile_name": profile_name,
            "original_image_path": str(original_image_path),
            "processed_image_path": str(processed_image_path),
            "trace": [],
        }
    )
    stop_reason = final.get("stop_reason")
    return AnalysisOutcome(
        contains_handwriting=not stop_reason,
        rejection_reason=stop_reason,
        visual=final.get("visual"),
        history=final.get("history"),
        rag_chunks=final.get("rag_chunks") or [],
        cited_sources=final.get("cited_sources") or [],
        feedback=final.get("feedback"),
        risk_flags=final.get("risk_flags") or [],
        trace=final.get("trace") or [],
        total_duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
