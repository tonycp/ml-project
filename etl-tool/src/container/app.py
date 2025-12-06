from dependency_injector.containers import DeclarativeContainer
from dependency_injector import providers

from .connection import Connection
from .gateway import Gateway
from .logging import Logging


class Gateway(DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Container(
        Logging,
        config=config.logging,
    )

    connection = providers.Container(
        Connection,
        config=config.connection,
    )

    gateway = providers.Container(
        Gateway,
        config=config.gateway,
    )
