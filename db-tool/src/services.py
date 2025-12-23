"""Servicios de dominio (Business Logic)"""
from pathlib import Path
from typing import List
import re

from interfaces import ISQLExecutor, ILogger


class IdentityInsertService:
    """Servicio para manejar IDENTITY_INSERT automáticamente"""
    
    @staticmethod
    def wrap_batch(batch_sql: str) -> str:
        """Envuelve batch con SET IDENTITY_INSERT para tablas detectadas"""
        tables = IdentityInsertService._extract_tables(batch_sql)
        
        if not tables:
            return batch_sql
        
        prefix = "".join(f"SET IDENTITY_INSERT {table} ON;\n" for table in tables)
        suffix = "".join(f"SET IDENTITY_INSERT {table} OFF;\n" for table in tables)
        
        return prefix + batch_sql + "\n" + suffix
    
    @staticmethod
    def _extract_tables(sql: str) -> List[str]:
        """Extrae nombres de tablas de INSERT statements"""
        tables = set()
        pattern = r"INSERT\s+(?:INTO\s+)?(\[[\w\-]+\]\.\[[\w]+\]|\[[\w]+\])"
        
        for line in sql.split("\n"):
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                tables.add(match.group(1))
        
        return list(tables)


class SetupExecutor:
    """Servicio para ejecutar scripts de setup"""
    
    def __init__(self, executor: ISQLExecutor, logger: ILogger):
        self.executor = executor
        self.logger = logger
    
    def execute_setup(self, setup_path: Path) -> bool:
        """Ejecuta script de setup si existe"""
        if not setup_path.exists():
            return True
        
        with open(setup_path, "r", encoding="utf-8-sig", errors="replace") as f:
            setup_sql = f.read().strip()
        
        if not setup_sql:
            return True
        
        success, stdout, stderr = self.executor.execute(setup_sql)
        self.logger.log_batch("SETUP", stdout, stderr)
        
        if not success:
            self.logger.log(f"Setup failed: {stderr}", "error")
        
        return success
