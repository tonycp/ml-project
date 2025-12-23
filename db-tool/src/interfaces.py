"""Interfaces y contratos (Dependency Inversion Principle)"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Tuple


class ISQLExecutor(ABC):
    """Interfaz para ejecutores de SQL"""
    
    @abstractmethod
    def execute(self, sql: str, database: str = "master") -> Tuple[bool, str, str]:
        """Ejecuta SQL y retorna (success, stdout, stderr)"""
        pass


class IStatementParser(ABC):
    """Interfaz para parseadores de SQL"""
    
    @abstractmethod
    def parse(self, sql_path: Path) -> Generator[Tuple[str, float], None, None]:
        """Genera tuplas de (statement, progress_percent)"""
        pass


class IHealthChecker(ABC):
    """Interfaz para verificadores de salud"""
    
    @abstractmethod
    def is_healthy(self, host: str, port: int, timeout: int) -> bool:
        """Verifica si el servicio está saludable"""
        pass


class ILogger(ABC):
    """Interfaz para logging"""
    
    @abstractmethod
    def log(self, message: str, level: str = "info"):
        """Registra un mensaje"""
        pass
    
    @abstractmethod
    def log_batch(self, batch_label: str, stdout: str, stderr: str):
        """Registra resultado de un batch"""
        pass
