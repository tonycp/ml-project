"""Configuración centralizada de la aplicación usando Pydantic"""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class DatabaseConfig(BaseSettings):
    """Configuración para conexión a SQL Server"""

    username: str = Field(default="sa", description="Usuario de SQL Server")
    password: str = Field(default="Meteorology2025!", description="Password de SQL Server")
    default_database: str = Field(default="master", description="Base de datos por defecto")
    connection_timeout: int = Field(default=300, description="Timeout de conexión en segundos")
    health_check_timeout: int = Field(default=120, description="Timeout de health check")
    health_check_interval: int = Field(default=1, description="Intervalo entre checks")

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


class PathConfig(BaseSettings):
    """Configuración de rutas del proyecto"""

    data_dir: Path = Field(default=Path(".data"), description="Directorio de datos")
    logs_dir: Path = Field(default=Path(".data/logs"), description="Directorio de logs")
    backup_dir: Path = Field(default=Path("backup"), description="Directorio de backups")
    setup_dir: Path = Field(default=Path("backup/start"), description="Directorio de setup")
    compose_file: Path = Field(
        default=Path("docker-compose.yml"), description="Archivo docker-compose"
    )

    @field_validator("*", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        """Convierte strings a Path"""
        if isinstance(v, str):
            return Path(v)
        return v

    model_config = SettingsConfigDict(
        env_prefix="PATH_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


class LoaderConfig(BaseSettings):
    """Configuración del cargador de datos"""

    batch_size: int = Field(default=100, ge=1, description="Tamaño de batch para SQL")
    max_workers: int = Field(default=5, ge=1, le=20, description="Workers paralelos")
    docker_startup_wait: int = Field(
        default=5, ge=0, description="Segundos a esperar tras Docker start"
    )

    model_config = SettingsConfigDict(
        env_prefix="LOADER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


class AppConfig(BaseSettings):
    """Configuración completa de la aplicación"""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "AppConfig":
        """Carga configuración desde archivo YAML"""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            return cls()
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)

    @classmethod
    def from_docker_compose(cls, compose_path: Optional[Path] = None) -> "AppConfig":
        """
        Crea configuración detectando servicios desde docker-compose.yml
        
        Esto no carga toda la config desde compose, sino que usa los paths
        para que _load_services lo lea después.
        """
        compose_path = compose_path or Path("docker-compose.yml")
        
        return cls(
            paths=PathConfig(compose_file=compose_path)
        )
