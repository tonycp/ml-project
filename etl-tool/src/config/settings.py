from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

from .gateway import GatewayConfig
from .logging import LoggingConfig
from .connection import ConnectionConfig

__all__ = ["AppSettings"]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", yaml_file="app_settings.yaml")
    connection: ConnectionConfig = ConnectionConfig(env_prefix="CONNECTION_")
    gateway: GatewayConfig = GatewayConfig(env_prefix="GATEWAY_")
    logging: LoggingConfig = LoggingConfig(env_prefix="LOGGING_")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            YamlConfigSettingsSource(
                settings_cls,
                yaml_file=cls.model_config.get("yaml_file", "app_settings.yaml"),
            ),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
