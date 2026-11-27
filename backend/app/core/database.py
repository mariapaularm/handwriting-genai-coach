"""Engine async de SQLAlchemy, sesiones y creación de tablas."""

from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Clase base declarativa para todos los modelos ORM."""


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def build_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """Crea un engine async; en SQLite activa las claves foráneas."""
    async_engine = create_async_engine(database_url, echo=echo)
    if async_engine.dialect.name == "sqlite":
        event.listen(async_engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
    return async_engine


engine: AsyncEngine = build_engine(settings.DATABASE_URL, echo=settings.DATABASE_ECHO)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncIterator[AsyncSession]:
    """Dependencia de FastAPI que entrega una sesión por request."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db(target_engine: AsyncEngine | None = None) -> None:
    """Crea todas las tablas registradas en `Base.metadata` si no existen."""
    # Importar los modelos para que queden registrados en Base.metadata.
    from app.models import db_models  # noqa: F401

    async with (target_engine or engine).begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
