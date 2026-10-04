"""Database engine, session factory and the FastAPI session dependency.

The API is read-mostly and I/O bound: no probability is ever computed here
(B1.1), so an async engine is the right shape -- requests spend their time
waiting on Postgres, not on CPU.

pool_pre_ping guards against connections killed underneath us by a pooler or a
restarted container (R-07).
"""

from sqlalchemy.ext.asyncio.session import AsyncSession


from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine
) 
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

class Base(DeclarativeBase):
    """Declarative e base for all ORM models."""

engine: AsyncEngine = create_async_engine(
    settings.async_database_url,
    echo = settings.db_echo,
    pool_size= settings.db_pool_size,
    max_overflow= settings.db_max_overflow,
    pool_pre_ping=settings.db_pool_pre_ping,
)

SessionLocal = async_sessionmaker(
    bind= engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

async def  get_session() -> AsyncGenerator[AsyncSession, None]:
    """ FastAPi dependency yielding a session that is always closed."""
    async with SessionLocal() as session:
        yield session


async def  dispose_engine() -> None:
    await engine.dispose()
