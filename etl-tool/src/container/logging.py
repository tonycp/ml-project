from dependency_injector import containers, providers

import logging.config


class Logging(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.settings,
    )
