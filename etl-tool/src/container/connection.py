from dependency_injector.containers import DeclarativeContainer
from dependency_injector import providers
from typing import Any, Dict

from src.config.connection import DBConnectionConfig
from src.connection import strings


def provide_connection(souce: Dict[str:Any]):
    model = DBConnectionConfig.model_validate(souce)
    connection = strings.map_con(model)
    return providers.Singleton(connection)


class Connection(DeclarativeContainer):
    config = providers.Configuration()

    sources = providers.List(*map(provide_connection, config.souces))
    target = provide_connection(config.target)
