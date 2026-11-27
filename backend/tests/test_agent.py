"""Pruebas del núcleo GenAI con Gemini mockeado (sin red ni API key real).

- /api/v1/analyze: flujo completo, persistencia, rechazo 422, reintentos 429, timeouts y guardrails.
- Herramientas FastMCP invocadas por protocolo MCP (cliente en memoria).
- Retriever RAG sobre la knowledge_base real con ChromaDB y embeddings deterministas locales.
"""

import asyncio
import io
import re
import shutil
import time
import unicodedata
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from google.genai.errors import ClientError, ServerError
from httpx import ASGITransport, AsyncClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from PIL import Image
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent import graph as agent_graph
from app.agent.prompts import CRISIS_MESSAGE, WELLNESS_DISCLAIMER
from app.agent.schemas import VisualAnalysisOutput
from app.core import database, gemini
from app.core.config import settings
from app.core.database import build_engine, init_db
from app.main import app
from app.mcp_server import server as mcp_server
from app.models.db_models import AnalysisRecord
from app.rag import retriever as rag_module
from app.rag.retriever import WellnessRetriever

pytestmark = pytest.mark.anyio

KB_SOURCES = {
    "01_regulacion_ansiedad.md",
    "02_manejo_estres_academico_laboral.md",
    "03_bienestar_estado_animo_bajo.md",
    "04_indicadores_grafologicos_trazo.md",
    "05_limites_eticos_y_criterio_clinico.md",
}
REAL_GET_CHAT_MODEL = gemini.get_chat_model
NODES = ["visual_analysis", "mcp_history", "rag_retrieval", "synthesis_guardrail"]


# ---------------------------------------------------------------------------
# Dobles de prueba
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


class KeywordEmbeddings(Embeddings):
    """Embeddings deterministas por conteo de términos: semántica mínima sin llamar a Gemini."""

    VOCAB = [
        # ansiedad
        "ansiedad", "respiracion", "preocupacion", "fisiologica", "anclaje", "relajacion", "panico",
        # estrés
        "estres", "academico", "laboral", "pausas", "plazos", "sobrecarga", "pomodoro", "cognitiva",
        # ánimo bajo
        "animo", "conductual", "rutina", "apatia", "placer", "logro",
        # trazo y ética
        "presion", "inclinacion", "espaciado", "diagnostico", "etica", "crisis",
    ]

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def _vector(self, text: str) -> list[float]:
        normalized = _normalize(text)
        return [float(normalized.count(term)) for term in self.VOCAB] + [0.5]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)


class NoReindexEmbeddings(KeywordEmbeddings):
    """Falla si se intenta volver a generar embeddings de documentos."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("No debería re-indexar: el índice existente debe reutilizarse")


def make_visual(**overrides: Any) -> VisualAnalysisOutput:
    data: dict[str, Any] = {
        "contains_handwriting": True,
        "rejection_reason": None,
        "transcribed_text": "Estoy muy nervioso por el examen de mañana, no paro de pensar en eso",
        "emotion": "anxiety",
        "confidence": 0.74,
        "visual_indicators": {
            "pressure": {"score": 78, "observation": "Trazo grueso y oscuro"},
            "rhythm": {"score": 31, "observation": "Cortes frecuentes"},
            "slant": {"score": 62, "observation": "Inclinación ligera a la derecha"},
            "spacing": {"score": 25, "observation": "Palabras muy juntas"},
            "stroke_variability": {"score": 70, "observation": "Tamaño de letra irregular"},
        },
        "visual_summary": "Escritura apretada con presión alta y ritmo entrecortado.",
    }
    data.update(overrides)
    return VisualAnalysisOutput.model_validate(data)


NO_HANDWRITING = {
    "contains_handwriting": False,
    "rejection_reason": "La imagen muestra un paisaje sin texto manuscrito.",
    "transcribed_text": "",
    "emotion": "neutral",
    "confidence": 0.0,
    "visual_summary": "Fotografía de un paisaje.",
}

FAKE_FEEDBACK = (
    "### Lo que muestra tu escritura\nTu trazo firme y apretado podría reflejar tensión.\n"
    "### Para hoy\n- Prueba la respiración diafragmática [01_regulacion_ansiedad.md]\n"
    "- Una técnica inventada [documento_inexistente.md]\n"
    "### Un último pensamiento\nVas paso a paso."
)


class FakeGemini:
    """Sustituye a ChatGoogleGenerativeAI: registra llamadas y permite inyectar fallos."""

    def __init__(self) -> None:
        self.visual_output: VisualAnalysisOutput = make_visual()
        self.feedback = FAKE_FEEDBACK
        self.visual_errors: list[BaseException] = []
        self.visual_delay = 0.0
        self.visual_calls = 0
        self.visual_messages: list[Any] = []
        self.synthesis_messages: list[Any] = []

    def with_structured_output(self, schema: Any, **_: Any) -> "_FakeStructured":
        assert schema is VisualAnalysisOutput
        return _FakeStructured(self)

    def invoke(self, messages: Any, **_: Any) -> AIMessage:
        self.synthesis_messages.append(messages)
        return AIMessage(content=self.feedback)


class _FakeStructured:
    def __init__(self, parent: FakeGemini) -> None:
        self.parent = parent

    def invoke(self, messages: Any, **_: Any) -> VisualAnalysisOutput:
        parent = self.parent
        parent.visual_calls += 1
        parent.visual_messages.append(messages)
        if parent.visual_delay:
            time.sleep(parent.visual_delay)
        if parent.visual_errors:
            raise parent.visual_errors.pop(0)
        return parent.visual_output


def rate_limit_error() -> GoogleRateLimitError:
    return GoogleRateLimitError("Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 quota")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class Api:
    client: AsyncClient
    fake: FakeGemini
    sessions: async_sessionmaker[AsyncSession]
    upload_root: Path
    embeddings: KeywordEmbeddings


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def db_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """SQLite temporal; tanto la API como las herramientas MCP usan `database.AsyncSessionLocal`."""
    engine = build_engine(f"sqlite+aiosqlite:///{(tmp_path / 'agent.db').as_posix()}")
    await init_db(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "AsyncSessionLocal", sessions)
    yield sessions
    await engine.dispose()


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GEMINI_RETRY_INITIAL_SECONDS", 0.0)
    monkeypatch.setattr(settings, "GEMINI_RETRY_MAX_SECONDS", 0.0)
    monkeypatch.setattr(settings, "GEMINI_MAX_RETRIES", 2)


@pytest.fixture
async def api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db_sessions: async_sessionmaker[AsyncSession], fast_retries: None
) -> AsyncIterator[Api]:
    upload_root = tmp_path / "uploads"
    monkeypatch.setattr(settings, "UPLOAD_DIR", upload_root)

    embeddings = KeywordEmbeddings()
    rag = WellnessRetriever(embeddings, persist_dir=tmp_path / "chroma", embedding_model_name="fake-keyword")
    monkeypatch.setattr(rag_module, "get_retriever", lambda: rag)

    fake = FakeGemini()
    monkeypatch.setattr(gemini, "get_chat_model", lambda: fake)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield Api(client=client, fake=fake, sessions=db_sessions, upload_root=upload_root, embeddings=embeddings)


def handwriting_png() -> bytes:
    img = Image.new("RGB", (600, 300), "white")
    for y in (80, 150, 220):
        for x in range(40, 560):
            img.putpixel((x, y + (x // 40) % 3), (20, 20, 20))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


async def post_analyze(api: Api, profile_name: str = "maria", data: bytes | None = None) -> Any:
    return await api.client.post(
        "/api/v1/analyze",
        files={"file": ("nota.png", data if data is not None else handwriting_png(), "image/png")},
        data={"profile_name": profile_name},
    )


def stored_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


# ---------------------------------------------------------------------------
# /api/v1/analyze
# ---------------------------------------------------------------------------


async def test_analyze_full_flow_and_history_persistence(api: Api) -> None:
    response = await post_analyze(api)
    assert response.status_code == 200, response.text
    body = response.json()

    # Análisis visual
    assert body["profile_name"] == "maria"
    assert body["emotion"] == "anxiety"
    assert body["confidence"] == pytest.approx(0.74)
    assert body["visual_indicators"]["pressure"] == {"score": 78, "observation": "Trazo grueso y oscuro"}
    assert body["biomarkers"] == {"pressure": 78, "rhythm": 31, "slant": 62, "spacing": 25, "stroke_variability": 70}

    # Gemini recibió la imagen original y la preprocesada
    content = api.fake.visual_messages[0][-1].content
    images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert len(images) == 2
    assert images[0].startswith("data:image/jpeg;base64,") and images[1].startswith("data:image/png;base64,")

    # Retroalimentación + guardrails
    assert body["feedback"].rstrip().endswith(WELLNESS_DISCLAIMER)
    assert "[01_regulacion_ansiedad.md]" in body["feedback"]
    assert "documento_inexistente.md" not in body["feedback"]  # cita inventada eliminada
    assert body["risk_flags"] == []

    # Fuentes RAG
    assert len(body["rag_sources"]) == settings.RAG_TOP_K
    assert {s["source"] for s in body["rag_sources"]} <= KB_SOURCES
    assert body["rag_sources"][0]["source"] == "01_regulacion_ansiedad.md"
    assert body["cited_sources"] == ["01_regulacion_ansiedad.md"]
    assert any(s["cited"] for s in body["rag_sources"])

    # Traza de LangGraph
    trace = body["trace"]
    assert [t["node"] for t in trace][0] == "visual_analysis"
    assert sorted(t["node"] for t in trace) == sorted(NODES)
    assert trace[-1]["node"] == "synthesis_guardrail"
    assert all(t["status"] == "ok" and t["duration_ms"] >= 0 for t in trace)

    # Archivos guardados
    assert Path(body["image_path"]).exists() and Path(body["processed_image_path"]).exists()
    assert Path(body["image_path"]).parent.name == "maria"

    # Persistencia en el historial
    history = (await api.client.get("/api/v1/history", params={"profile_name": "maria"})).json()
    assert len(history) == 1
    record = history[0]
    assert record["id"] == body["id"]
    assert record["emotion"] == "anxiety"
    assert record["agent_feedback"] == body["feedback"]
    assert record["visual_summary"] == body["visual_summary"]
    assert record["visual_indicators"]["rhythm"]["score"] == 31
    assert {s["source"] for s in record["rag_sources"]} <= KB_SOURCES
    assert [t["node"] for t in record["trace"]] == [t["node"] for t in trace]


async def test_second_analysis_uses_mcp_history_and_reuses_rag_index(api: Api) -> None:
    assert (await post_analyze(api)).status_code == 200
    second = await post_analyze(api)
    assert second.status_code == 200
    body = second.json()

    assert body["history"]["previous_analyses"] == 1
    assert body["history"]["trend"]["distribution"]["anxiety"] == 1
    # El contexto MCP llegó al prompt de síntesis
    synthesis_prompt = api.fake.synthesis_messages[-1][-1].content
    assert '"analisis_previos": 1' in synthesis_prompt
    # El índice RAG se construyó una sola vez
    assert api.embeddings.document_calls == 1


@pytest.mark.parametrize(
    "visual_overrides",
    [NO_HANDWRITING, {"contains_handwriting": True, "transcribed_text": "   "}],
    ids=["vlm_sin_escritura", "transcripcion_vacia"],
)
async def test_no_handwriting_returns_422_without_persisting(api: Api, visual_overrides: dict) -> None:
    api.fake.visual_output = make_visual(**visual_overrides)

    response = await post_analyze(api)
    assert response.status_code == 422
    body = response.json()
    assert body["contains_handwriting"] is False
    assert "No se detectó escritura manuscrita" in body["detail"]
    assert body["reason"]
    assert [t["node"] for t in body["trace"]] == ["visual_analysis"]
    assert body["trace"][0]["status"] == "stopped"

    # El flujo se detuvo: ni MCP, ni RAG, ni síntesis
    assert api.fake.synthesis_messages == []
    assert api.embeddings.query_calls == 0
    assert (await api.client.get("/api/v1/history")).json() == []
    assert stored_files(api.upload_root) == []


async def test_rate_limit_is_retried_with_backoff_then_succeeds(api: Api) -> None:
    api.fake.visual_errors = [rate_limit_error(), rate_limit_error()]
    response = await post_analyze(api)
    assert response.status_code == 200, response.text
    assert api.fake.visual_calls == 3  # 1 intento + 2 reintentos


async def test_rate_limit_exhausted_returns_429(api: Api) -> None:
    api.fake.visual_errors = [rate_limit_error() for _ in range(10)]
    response = await post_analyze(api)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert response.json()["operation"] == "análisis visual"
    assert api.fake.visual_calls == settings.GEMINI_MAX_RETRIES + 1
    assert (await api.client.get("/api/v1/history")).json() == []
    assert stored_files(api.upload_root) == []


async def test_gemini_timeout_returns_504(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GEMINI_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(settings, "GEMINI_TIMEOUT_GRACE_SECONDS", 0.0)
    monkeypatch.setattr(settings, "GEMINI_MAX_RETRIES", 1)
    api.fake.visual_delay = 0.3
    response = await post_analyze(api)
    assert response.status_code == 504
    assert api.fake.visual_calls == 2
    assert stored_files(api.upload_root) == []


async def test_missing_api_key_returns_503(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini, "get_chat_model", REAL_GET_CHAT_MODEL)  # cliente real...
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("pega_aqui_tu_api_key"))  # ...sin key válida
    response = await post_analyze(api)
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]
    assert stored_files(api.upload_root) == []


async def test_crisis_language_triggers_safety_guardrail(api: Api) -> None:
    api.fake.visual_output = make_visual(
        transcribed_text="Ya no puedo más, a veces pienso que no quiero vivir", emotion="depression"
    )
    response = await post_analyze(api)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_flags"]
    assert body["feedback"].startswith(CRISIS_MESSAGE)
    assert body["feedback"].rstrip().endswith(WELLNESS_DISCLAIMER)
    assert '"senales_de_riesgo": [' in api.fake.synthesis_messages[-1][-1].content


async def test_rag_failure_degrades_gracefully(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken_retrieve(*_: Any, **__: Any) -> list:
        raise gemini.GeminiConfigurationError("Modelo de embeddings no disponible", operation="consulta RAG")

    monkeypatch.setattr(rag_module, "retrieve_wellness_context", broken_retrieve)
    response = await post_analyze(api)
    assert response.status_code == 200
    body = response.json()
    rag_trace = next(t for t in body["trace"] if t["node"] == "rag_retrieval")
    assert rag_trace["status"] == "degraded"
    assert "embeddings" in rag_trace["detail"]
    assert body["rag_sources"] == []
    assert body["cited_sources"] == []  # sin fuentes recuperadas no se aceptan citas


async def test_invalid_file_is_rejected_with_400_before_calling_gemini(api: Api) -> None:
    response = await post_analyze(api, data=b"esto no es una imagen")
    assert response.status_code == 400
    assert api.fake.visual_calls == 0


async def test_invalid_profile_name_is_rejected(api: Api) -> None:
    response = await post_analyze(api, profile_name="../fuera")
    assert response.status_code == 422
    assert api.fake.visual_calls == 0


# ---------------------------------------------------------------------------
# Herramientas FastMCP
# ---------------------------------------------------------------------------


async def seed_records(sessions: async_sessionmaker[AsyncSession], profile: str, emotions: list[str]) -> list[str]:
    """Inserta registros; `emotions` va del más reciente al más antiguo."""
    now = datetime.now(timezone.utc)
    ids = []
    async with sessions() as session:
        for minutes, emotion in enumerate(emotions):
            record = AnalysisRecord(
                id=uuid.uuid4().hex,
                profile_name=profile,
                image_path=f"uploads/{profile}/{minutes}.png",
                emotion=emotion,
                confidence=0.7,
                visual_summary=f"resumen {minutes}",
                visual_indicators_json={"pressure": {"score": 60 + minutes, "observation": "x"}},
                created_at=now - timedelta(minutes=minutes),
            )
            session.add(record)
            ids.append(record.id)
        await session.commit()
    return ids


async def test_mcp_server_exposes_three_tools() -> None:
    async with Client(mcp_server.mcp) as client:
        names = {tool.name for tool in await client.list_tools()}
    assert names == {"get_recent_analyses", "get_emotional_trend", "get_graphology_reference_rules"}


async def test_mcp_get_recent_analyses(db_sessions: async_sessionmaker[AsyncSession]) -> None:
    ids = await seed_records(db_sessions, "maria", ["stress", "anxiety", "neutral", "stress", "neutral", "stress"])
    await seed_records(db_sessions, "otro", ["depression"])

    result = await mcp_server.call_mcp_tool("get_recent_analyses", {"profile_name": "maria", "limit": 3})
    parsed = mcp_server.RecentAnalysesResult.model_validate(result)
    assert parsed.count == 3
    assert [a.id for a in parsed.analyses] == ids[:3]  # más reciente primero
    assert parsed.analyses[0].indicator_scores == {"pressure": 60}
    assert all(a.emotion != "depression" for a in parsed.analyses)


async def test_mcp_emotional_trend_detects_recurrence(db_sessions: async_sessionmaker[AsyncSession]) -> None:
    await seed_records(db_sessions, "maria", ["stress", "anxiety", "stress", "neutral", "anxiety"])
    await seed_records(db_sessions, "tranquilo", ["neutral", "neutral", "stress"])

    trend = mcp_server.EmotionalTrendResult.model_validate(
        await mcp_server.call_mcp_tool("get_emotional_trend", {"profile_name": "maria"})
    )
    assert trend.total_analyses == 5
    assert trend.distribution == {"neutral": 1, "anxiety": 2, "stress": 2, "depression": 0}
    assert trend.stress_anxiety_count == 4
    assert trend.consecutive_stress_anxiety == 3
    assert trend.recurrent_stress_anxiety is True
    assert trend.alert_level == "elevated"

    calm = mcp_server.EmotionalTrendResult.model_validate(
        await mcp_server.call_mcp_tool("get_emotional_trend", {"profile_name": "tranquilo"})
    )
    assert calm.recurrent_stress_anxiety is False
    assert calm.alert_level == "none"
    assert calm.dominant_emotion == "neutral"

    empty = mcp_server.EmotionalTrendResult.model_validate(
        await mcp_server.call_mcp_tool("get_emotional_trend", {"profile_name": "nuevo"})
    )
    assert empty.total_analyses == 0 and empty.dominant_emotion is None


async def test_mcp_graphology_rules_cover_full_scale() -> None:
    rules = mcp_server.GraphologyReferenceRules.model_validate(
        await mcp_server.call_mcp_tool("get_graphology_reference_rules")
    )
    assert [r.indicator for r in rules.indicators] == list(mcp_server.INDICATORS)
    for rule in rules.indicators:
        assert [(b.min, b.max) for b in rule.bands] == [(0, 33), (34, 66), (67, 100)]


async def test_mcp_fetch_profile_context_helper(db_sessions: async_sessionmaker[AsyncSession]) -> None:
    await seed_records(db_sessions, "maria", ["stress", "stress"])
    context = await mcp_server.fetch_profile_context("maria")
    assert context.recent.count == 2
    assert context.trend.stress_anxiety_count == 2
    assert len(context.rules.indicators) == 5


async def test_mcp_rejects_invalid_profile_name() -> None:
    with pytest.raises(Exception):
        await mcp_server.call_mcp_tool("get_recent_analyses", {"profile_name": "../etc"})


# ---------------------------------------------------------------------------
# Retriever RAG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("emotion", "expected_source"),
    [
        ("anxiety", "01_regulacion_ansiedad.md"),
        ("stress", "02_manejo_estres_academico_laboral.md"),
        ("depression", "03_bienestar_estado_animo_bajo.md"),
    ],
)
async def test_retriever_returns_relevant_chunks_with_source(tmp_path: Path, emotion: str, expected_source: str) -> None:
    rag = WellnessRetriever(KeywordEmbeddings(), persist_dir=tmp_path / "chroma", embedding_model_name="fake")
    chunks = await rag.retrieve(emotion, "", k=3)

    assert len(chunks) == 3
    assert chunks[0].source == expected_source
    assert all(c.source in KB_SOURCES and c.title and c.content for c in chunks)
    assert chunks[0].score >= chunks[-1].score


async def test_retrieve_wellness_context_module_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rag = WellnessRetriever(KeywordEmbeddings(), persist_dir=tmp_path / "chroma", embedding_model_name="fake")
    monkeypatch.setattr(rag_module, "get_retriever", lambda: rag)
    chunks = await rag_module.retrieve_wellness_context("stress", "letras apretadas", k=2)
    assert len(chunks) == 2
    assert chunks[0].source == "02_manejo_estres_academico_laboral.md"


async def test_retriever_reuses_persisted_index_without_new_embeddings(tmp_path: Path) -> None:
    persist = tmp_path / "chroma"
    first = WellnessRetriever(KeywordEmbeddings(), persist_dir=persist, embedding_model_name="fake")
    collection = await first.ensure_index()
    total = collection.count()
    assert total > 5  # varios chunks por documento

    reused = WellnessRetriever(NoReindexEmbeddings(), persist_dir=persist, embedding_model_name="fake")
    chunks = await reused.retrieve("anxiety", "", k=1)
    assert (await reused.ensure_index()).count() == total
    assert chunks[0].source == "01_regulacion_ansiedad.md"


async def test_retriever_rebuilds_index_when_knowledge_base_changes(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    shutil.copytree(settings.KNOWLEDGE_BASE_DIR, kb)
    persist = tmp_path / "chroma"
    embeddings = KeywordEmbeddings()

    await WellnessRetriever(embeddings, knowledge_base_dir=kb, persist_dir=persist, embedding_model_name="fake").ensure_index()
    (kb / "06_nuevo.md").write_text("# Nuevo documento\n\nContenido sobre pomodoro y pausas.", encoding="utf-8")
    rebuilt = WellnessRetriever(embeddings, knowledge_base_dir=kb, persist_dir=persist, embedding_model_name="fake")
    await rebuilt.ensure_index()

    assert embeddings.document_calls == 2
    sources = {m["source"] for m in (await rebuilt.ensure_index()).get(include=["metadatas"])["metadatas"]}
    assert "06_nuevo.md" in sources


def test_chunks_keep_source_title_and_section() -> None:
    rag = WellnessRetriever(KeywordEmbeddings(), embedding_model_name="fake")
    chunks = rag.split_documents(rag.load_documents())
    assert {c.metadata["source"] for c in chunks} == KB_SOURCES
    assert all(len(c.page_content) <= settings.RAG_CHUNK_SIZE for c in chunks)
    first = next(c for c in chunks if c.metadata["source"] == "01_regulacion_ansiedad.md")
    assert first.metadata["title"] == "Regulación de la ansiedad y la activación fisiológica"
    documents = {d.metadata["source"]: d.page_content for d in rag.load_documents()}
    for chunk in chunks:  # cada sección es un encabezado real del documento fuente
        assert re.search(rf"^#+ {re.escape(chunk.metadata['section'])}$", documents[chunk.metadata["source"]], re.M)
    assert len({c.metadata["section"] for c in chunks if c.metadata["source"] == "01_regulacion_ansiedad.md"}) > 1


# ---------------------------------------------------------------------------
# Resiliencia y guardrails (unitarias)
# ---------------------------------------------------------------------------


def _wrapped(exc: BaseException) -> GoogleGenerativeAIError:
    try:
        raise GoogleGenerativeAIError("Error embedding content") from exc
    except GoogleGenerativeAIError as wrapped:
        return wrapped


def test_error_classification_walks_wrapped_sdk_errors() -> None:
    quota = ClientError(429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}})
    bad_request = ClientError(400, {"error": {"code": 400, "message": "Bad", "status": "INVALID_ARGUMENT"}})
    not_found = ClientError(404, {"error": {"code": 404, "message": "models/text-embedding-004 not found", "status": "NOT_FOUND"}})

    assert gemini.is_retryable_error(_wrapped(quota)) is True
    assert gemini.is_retryable_error(TimeoutError()) is True
    assert gemini.is_retryable_error(_wrapped(bad_request)) is False
    assert isinstance(gemini.classify_error(_wrapped(quota), "x"), gemini.GeminiRateLimitError)
    assert isinstance(gemini.classify_error(TimeoutError(), "x"), gemini.GeminiTimeoutError)
    config_error = gemini.classify_error(_wrapped(not_found), "x")
    assert isinstance(config_error, gemini.GeminiConfigurationError)
    assert "gemini-embedding-001" in config_error.message


async def test_call_gemini_retries_transient_errors(fast_retries: None) -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _wrapped(ClientError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}))
        return "ok"

    assert await gemini.call_gemini("prueba", flaky) == "ok"
    assert attempts == 3


async def test_call_gemini_does_not_retry_permanent_errors(fast_retries: None) -> None:
    attempts = 0

    async def invalid() -> str:
        nonlocal attempts
        attempts += 1
        raise _wrapped(ClientError(403, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}))

    with pytest.raises(gemini.GeminiConfigurationError):
        await gemini.call_gemini("prueba", invalid)
    assert attempts == 1


def test_guardrails_add_disclaimer_and_strip_unknown_citations() -> None:
    text, cited = agent_graph.apply_guardrails(
        "Respira [01_regulacion_ansiedad.md] y [falso.md].", [], {"01_regulacion_ansiedad.md"}
    )
    assert cited == ["01_regulacion_ansiedad.md"]
    assert "falso.md" not in text
    assert text.endswith(WELLNESS_DISCLAIMER)
    assert agent_graph.detect_risk_signals("Pienso en QUITARME LA VIDA") != []
    assert agent_graph.detect_risk_signals("Hoy estudié para el examen") == []


DEADLINE_JSON = {"error": {"code": 504, "message": "Deadline expired before operation could complete.", "status": "DEADLINE_EXCEEDED"}}


def deadline_error() -> GoogleAPIError:
    """Reproduce el error real observado con gemini-3.8-flash (504 DEADLINE_EXCEEDED)."""
    server = ServerError(504, DEADLINE_JSON)
    try:
        raise GoogleAPIError(code=504, response_json=server.details, response=None) from server
    except GoogleAPIError as exc:
        return exc


def test_deadline_exceeded_is_classified_as_timeout_with_real_message() -> None:
    err = deadline_error()
    assert gemini.describe_error(err) == "504 DEADLINE_EXCEEDED: Deadline expired before operation could complete."
    error = gemini.classify_error(err, "análisis visual")
    assert isinstance(error, gemini.GeminiTimeoutError)
    assert error.status_code == 504
    assert "Deadline expired before operation could complete." in error.message
    assert gemini.is_retryable_error(err) is True


async def test_deadline_exceeded_returns_504_with_provider_detail(api: Api) -> None:
    api.fake.visual_errors = [deadline_error() for _ in range(10)]
    response = await post_analyze(api)
    assert response.status_code == 504
    body = response.json()
    assert "Deadline expired before operation could complete." in body["detail"]
    assert body["provider_detail"] == "504 DEADLINE_EXCEEDED: Deadline expired before operation could complete."
    assert stored_files(api.upload_root) == []


async def test_retries_respect_total_time_budget(fast_retries: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GEMINI_MAX_RETRIES", 50)
    monkeypatch.setattr(settings, "GEMINI_MAX_TOTAL_SECONDS", 0.2)
    attempts = 0

    async def slow_failure() -> None:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.08)
        raise deadline_error()

    with pytest.raises(gemini.GeminiTimeoutError):
        await gemini.call_gemini("prueba", slow_failure)
    assert attempts <= 4  # el presupuesto total corta antes de los 51 intentos


def test_thinking_level_only_applies_to_gemini_3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("test-key"))
    monkeypatch.setattr(settings, "GEMINI_THINKING_LEVEL", "low")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.8-flash")
    model = REAL_GET_CHAT_MODEL()
    assert model.reasoning_effort == "low"
    assert model.timeout == settings.GEMINI_TIMEOUT_SECONDS
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-2.0-flash")
    assert REAL_GET_CHAT_MODEL().reasoning_effort is None


async def test_attempt_timeout_never_exceeds_total_budget(fast_retries: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GEMINI_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(settings, "GEMINI_MAX_TOTAL_SECONDS", 0.2)

    async def hangs() -> None:
        await asyncio.sleep(5)

    started = time.perf_counter()
    with pytest.raises(gemini.GeminiTimeoutError):
        await gemini.call_gemini("prueba", hangs)
    assert time.perf_counter() - started < 1.0
