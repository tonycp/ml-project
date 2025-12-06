from pydantic import BaseModel, ConfigDict
from typing import Optional

__all__ = ["EngineConfig", "GatewayConfig"]


class EngineConfig(BaseModel):
    model_config = ConfigDict(env_prefix="ENGINE_")

    engine_type: str = "default"
    max_workers: int = 4
    log_level: str = "INFO"
    retry_on_failure: bool = True
    timeout_seconds: int = 60


class GatewayConfig(BaseModel):
    model_config = ConfigDict(env_prefix="GATEWAY_")
    engine: EngineConfig = EngineConfig(env_prefix="ENGINE_")

    source_type: Optional[str] = None
    target_type: Optional[str] = None
    source_con: Optional[str] = None
    target_con: Optional[str] = None
