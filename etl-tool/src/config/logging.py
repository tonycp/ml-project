from pydantic import BaseModel, ConfigDict
from pydantic_settings_logging import (
    LoggingSettings,
    FormatterConfig,
    StreamHandlerConfig,
    FileHandlerConfig,
    RootLoggerConfig,
)

__all__ = ["LoggingConfig"]


class LoggingConfig(BaseModel):
    model_config = ConfigDict(env_prefix="LOGGING_")

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = False
    file_path: str = "app.log"

    settings: LoggingSettings = LoggingSettings(
        formatters={
            "default": FormatterConfig(
                format=format,
                datefmt=date_format,
            ),
        },
        handlers={
            "console": StreamHandlerConfig(
                level=level,
                formatter="default",
            ),
            "file": FileHandlerConfig(
                level=level,
                formatter="default",
                filename=file_path,
            ),
        },
        root=RootLoggerConfig(
            level=level,
            handlers=["console", "file"] if log_to_file else ["console"],
        ),
    )
