"""Pipeline RAG: knowledge_base/*.md -> chunks -> embeddings Gemini -> ChromaDB persistente.

- El índice vive en `CHROMA_DIR` (por defecto `chroma_db/`) y se reutiliza entre arranques.
- Se guarda una *huella* (hash del contenido de la base, modelo de embeddings y parámetros de
  chunking) en los metadatos de la colección: si coincide, no se vuelve a llamar a la API de
  embeddings; si los documentos cambian, el índice se reconstruye automáticamente.

Uso por línea de comandos (desde `backend/`) para precalcular el índice:
    python -m app.rag.retriever
"""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.core import gemini
from app.core.config import settings

logger = logging.getLogger("handwriting.rag")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_ADD_BATCH = 64

# Consulta enriquecida por emoción: orienta la búsqueda hacia el documento de intervención adecuado.
EMOTION_QUERY_HINTS: dict[str, str] = {
    "anxiety": "ansiedad, preocupación y activación fisiológica: respiración diafragmática, anclaje y relajación",
    "stress": "estrés académico o laboral, sobrecarga y plazos: reestructuración cognitiva, organización y pausas activas",
    "depression": "estado de ánimo bajo, desánimo y apatía: activación conductual, rutinas y apoyo social",
    "neutral": "bienestar general y prevención: hábitos saludables, descanso y autocuidado",
}


class RetrievedChunk(BaseModel):
    """Fragmento recuperado de la base de conocimiento, con su documento fuente."""

    chunk_id: str
    source: str = Field(description="Nombre del archivo fuente, p. ej. 01_regulacion_ansiedad.md")
    title: str
    section: str | None = None
    content: str
    score: float = Field(ge=0.0, le=1.0, description="Similitud coseno (1 = idéntico)")


class WellnessRetriever:
    def __init__(
        self,
        embeddings: Embeddings,
        *,
        knowledge_base_dir: Path | None = None,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        embedding_model_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.knowledge_base_dir = Path(knowledge_base_dir or settings.KNOWLEDGE_BASE_DIR)
        self.persist_dir = Path(persist_dir or settings.CHROMA_DIR)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self.embedding_model_name = embedding_model_name or settings.GEMINI_EMBEDDING_MODEL
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self._collection: Any = None
        self._lock = asyncio.Lock()

    # ---------------- Carga y chunking ----------------

    def load_documents(self) -> list[Document]:
        if not self.knowledge_base_dir.is_dir():
            raise FileNotFoundError(f"No existe la base de conocimiento: {self.knowledge_base_dir}")
        loader = DirectoryLoader(
            str(self.knowledge_base_dir),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = sorted(loader.load(), key=lambda d: Path(d.metadata["source"]).name)
        if not documents:
            raise FileNotFoundError(f"No hay archivos .md en {self.knowledge_base_dir}")
        for doc in documents:
            doc.metadata["source"] = Path(doc.metadata["source"]).name
            title = next((m.group(2) for m in _HEADING_RE.finditer(doc.page_content) if m.group(1) == "#"), None)
            doc.metadata["title"] = title or doc.metadata["source"]
        return documents

    def split_documents(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        chunks: list[Document] = []
        for doc in documents:
            headings = [(m.start(), m.group(2)) for m in _HEADING_RE.finditer(doc.page_content)]
            for index, chunk in enumerate(splitter.split_documents([doc])):
                start = chunk.metadata.get("start_index", 0)
                section = next((text for pos, text in reversed(headings) if pos <= start), None)
                chunk.metadata.update(
                    {
                        "chunk_id": f"{doc.metadata['source']}::{index}",
                        "section": section or "",
                    }
                )
                chunks.append(chunk)
        return chunks

    def _fingerprint(self, documents: list[Document]) -> str:
        payload = {
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "docs": [(d.metadata["source"], hashlib.sha256(d.page_content.encode()).hexdigest()) for d in documents],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # ---------------- Índice ----------------

    def _client(self) -> Any:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(self.persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )

    async def ensure_index(self, force_rebuild: bool = False) -> Any:
        """Devuelve la colección; solo genera embeddings si no hay índice válido."""
        async with self._lock:
            if self._collection is not None and not force_rebuild:
                return self._collection

            documents = await asyncio.to_thread(self.load_documents)
            chunks = self.split_documents(documents)
            fingerprint = self._fingerprint(documents)
            client = await asyncio.to_thread(self._client)

            existing = await asyncio.to_thread(self._get_collection, client)
            if (
                existing is not None
                and not force_rebuild
                and (existing.metadata or {}).get("fingerprint") == fingerprint
                and await asyncio.to_thread(existing.count) == len(chunks)
            ):
                logger.info("Reutilizando índice RAG existente (%d chunks) en %s", len(chunks), self.persist_dir)
                self._collection = existing
                return existing

            logger.info("Construyendo índice RAG: %d documentos, %d chunks", len(documents), len(chunks))
            texts = [chunk.page_content for chunk in chunks]
            vectors = await gemini.call_gemini(
                "indexación RAG (embeddings)", gemini.in_thread(self.embeddings.embed_documents, texts)
            )
            self._collection = await asyncio.to_thread(
                self._write_collection, client, chunks, vectors, fingerprint
            )
            return self._collection

    def _get_collection(self, client: Any) -> Any:
        try:
            return client.get_collection(self.collection_name, embedding_function=None)
        except Exception:  # la colección aún no existe
            return None

    def _write_collection(
        self, client: Any, chunks: list[Document], vectors: list[list[float]], fingerprint: str
    ) -> Any:
        if self._get_collection(client) is not None:
            client.delete_collection(self.collection_name)
        collection = client.create_collection(
            self.collection_name,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "fingerprint": fingerprint,
                "embedding_model": self.embedding_model_name,
            },
        )
        for start in range(0, len(chunks), _ADD_BATCH):
            batch = chunks[start : start + _ADD_BATCH]
            collection.add(
                ids=[c.metadata["chunk_id"] for c in batch],
                documents=[c.page_content for c in batch],
                embeddings=vectors[start : start + _ADD_BATCH],
                metadatas=[
                    {
                        "source": c.metadata["source"],
                        "title": c.metadata["title"],
                        "section": c.metadata["section"],
                    }
                    for c in batch
                ],
            )
        return collection

    # ---------------- Consulta ----------------

    @staticmethod
    def build_query(emotion: str, visual_summary: str) -> str:
        hint = EMOTION_QUERY_HINTS.get(emotion, EMOTION_QUERY_HINTS["neutral"])
        return f"{hint}. Rasgos observados en la escritura: {visual_summary}".strip()

    async def retrieve(self, emotion: str, visual_summary: str, k: int = 3) -> list[RetrievedChunk]:
        collection = await self.ensure_index()
        query = self.build_query(emotion, visual_summary)
        vector = await gemini.call_gemini(
            "consulta RAG (embedding)", gemini.in_thread(self.embeddings.embed_query, query)
        )
        total = await asyncio.to_thread(collection.count)
        if total == 0:
            return []
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[vector],
            n_results=min(max(1, k), total),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source=meta["source"],
                    title=meta["title"],
                    section=meta.get("section") or None,
                    content=text,
                    score=round(min(1.0, max(0.0, 1.0 - float(distance))), 4),
                )
            )
        return chunks


# ---------------------------------------------------------------------------
# API de módulo
# ---------------------------------------------------------------------------

_default_retriever: WellnessRetriever | None = None


def get_retriever() -> WellnessRetriever:
    """Retriever compartido por el proceso (embeddings Gemini + ChromaDB persistente)."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = WellnessRetriever(gemini.get_embeddings())
    return _default_retriever


async def retrieve_wellness_context(emotion: str, visual_summary: str, k: int = 3) -> list[RetrievedChunk]:
    """Fragmentos de la base de conocimiento más relevantes para la emoción y el resumen visual."""
    return await get_retriever().retrieve(emotion, visual_summary, k)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    async def _main() -> None:
        retriever = get_retriever()
        collection = await retriever.ensure_index()
        print(f"Índice listo en {retriever.persist_dir} con {collection.count()} chunks.")

    asyncio.run(_main())
