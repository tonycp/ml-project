from pydantic import BaseModel, ConfigDict
from typing import List
from enum import Enum

__all__ = ["DBType", "DBConnectionConfig", "ConnectionConfig"]


class DBType(Enum):
    SQLITE = 0
    SERVER = 1
    POSTGRES = 2


class DBConnectionConfig(BaseModel):
    model_config = ConfigDict(env_prefix="DB_SERVER_")

    host: str = "localhost"
    port: int = 5432
    username: str = "user"
    password: str = "password"
    database: str = "mydatabase"

    db_type: DBType = DBType.SQLITE


class ConnectionConfig(BaseModel):
    model_config = ConfigDict(env_prefix="CONNECTION_")

    sources: List[DBConnectionConfig] = []
    target: DBConnectionConfig = DBConnectionConfig(env_prefix="SQLITE_SERVER_")
