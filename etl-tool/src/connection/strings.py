from dataclass_mapper import map_to, mapper
from dataclasses import dataclass
from typing import Type
from urllib import parse

from src.config.connection import DBConnectionConfig, DBType


@dataclass
class BaseConnectionString:
    host: str
    port: int
    username: str
    password: str
    database: str


@mapper(DBConnectionConfig)
@dataclass
class SqlServerConnectionString(BaseConnectionString):
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = False

    def build_url(self) -> str:
        conn_str = f"mssql+pyodbc:///?odbc_connect={
            parse.quote_plus(
                f'DRIVER={{{self.driver}}};SERVER={self.host},{self.port};'
                f'DATABASE={self.database};UID={self.username};PWD={self.password};'
            )
        }"
        return conn_str


@mapper(DBConnectionConfig)
@dataclass
class PostgresConnectionString(BaseConnectionString):
    ssl_mode: str = "disable"

    def build_url(self) -> str:
        return f"postgresql+psycopg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"


@mapper(DBConnectionConfig)
@dataclass
class SqliteConnectionString(BaseConnectionString):
    def build_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.database}"


def get_type(db_type: DBType) -> Type[BaseConnectionString]:
    if db_type is DBType.SQLITE:
        return SqliteConnectionString
    elif db_type is DBType.SERVER:
        return SqlServerConnectionString
    elif db_type is DBType.POSTGRES:
        return PostgresConnectionString
    else:
        raise TypeError()


def map_con(s: DBConnectionConfig) -> BaseConnectionString:
    conn_type = get_type(s.db_type)
    return map_to(s, conn_type)
