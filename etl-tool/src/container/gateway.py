from dependency_injector.containers import DeclarativeContainer
from dependency_injector import providers

from src.connection import database


class Gateway(DeclarativeContainer):
    config = providers.Configuration()

    source_db = providers.Selector(
        config.source_type,
        postgres=providers.Resource(
            database.PostgresDatabase,
            config.source_server,
            config.engine,
        ),
        server=providers.Resource(
            database.SqlServerDatabase,
            config.source_server,
            config.engine,
        ),
        lite=providers.Resource(
            database.SqliteDatabase,
            config.source_server,
            config.engine,
        ),
    )

    target_db = providers.Selector(
        config.target_type,
        postgres=providers.Resource(
            database.PostgresDatabase,
            config.target_server,
            config.engine,
        ),
        server=providers.Resource(
            database.SqlServerDatabase,
            config.target_server,
            config.engine,
        ),
        lite=providers.Resource(
            database.SqliteDatabase,
            config.target_server,
            config.engine,
        ),
    )
