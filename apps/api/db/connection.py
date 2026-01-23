"""Database connection management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)

settings = get_settings()

async_engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.api_debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database session.

    Yields:
        AsyncSession for database operations.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize database tables.

    Creates all tables defined in models if they don't exist.
    """
    from apps.api.db.models import Base

    # Import monitor models so they register with the shared Base
    import apps.monitor.db  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized")
