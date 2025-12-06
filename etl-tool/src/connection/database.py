from abc import ABC, abstractmethod
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.schema import MetaData
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager
from typing import Any, Dict, AsyncIterator

from src.config.connection import DBType


class BaseDatabase(ABC):
    @abstractmethod
    @property
    def db_type(self) -> DBType:
        pass

    def __init__(self, connection_string: str, engine_config: Dict[str, Any]):
        self._connection_string = connection_string
        self._engine: AsyncEngine = create_async_engine(
            connection_string,
            poolclass=NullPool,
            **engine_config,
        )
        self._sessionmaker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create_all(self, metadata: MetaData) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()


class SqlServerDatabase(BaseDatabase):
    def db_type(self) -> DBType:
        return DBType.SERVER


class PostgresDatabase(BaseDatabase):
    def db_type(self) -> DBType:
        return DBType.POSTGRES


class SqliteDatabase(BaseDatabase):
    def db_type(self) -> DBType:
        return DBType.SQLITE
