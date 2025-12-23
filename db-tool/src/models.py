"""Modelos de dominio"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceDefinition:
    """Define un servicio de base de datos"""
    
    name: str
    port: int
    sql_file: Path
    data_dir: Path
    setup_file: Path | None = None
    
    @property
    def host(self) -> str:
        return "localhost"
