"""Implementaciones concretas de interfaces (Single Responsibility)"""

from pathlib import Path
from typing import Tuple, Generator
import subprocess
import socket

from interfaces import ISQLExecutor, IStatementParser, IHealthChecker, ILogger
from config import DatabaseConfig


class SQLCmdExecutor(ISQLExecutor):
    """Ejecutor de SQL usando sqlcmd"""

    def __init__(self, host: str, port: int, config: DatabaseConfig):
        self.host = host
        self.port = port
        self.config = config

    def execute(self, sql: str, database: str = "master") -> Tuple[bool, str, str]:
        if not sql.strip():
            return True, "", ""

        cmd = [
            "sqlcmd",
            "-S",
            f"{self.host},{self.port}",
            "-U",
            self.config.username,
            "-P",
            self.config.password,
            "-C",
            "-d",
            database,
        ]

        result = subprocess.run(
            cmd,
            input=sql,
            text=True,
            capture_output=True,
            timeout=self.config.connection_timeout,
        )

        return result.returncode == 0, result.stdout, result.stderr


class StreamingSQLParser(IStatementParser):
    """Parser de SQL con streaming para archivos grandes"""

    def __init__(self, encoding_detector):
        self.encoding_detector = encoding_detector

    def parse(self, sql_path: Path) -> Generator[Tuple[str, float], None, None]:
        """Genera statements con progreso"""
        encoding = self.encoding_detector(sql_path)
        file_size = max(sql_path.stat().st_size, 1)
        bytes_read = 0
        buffer = ""
        in_block_comment = False

        with open(sql_path, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                bytes_read += len(line.encode(encoding, errors="replace"))
                progress = round(min(float((bytes_read / file_size) * 100), 99), 2)

                # Manejo de comentarios de bloque
                if in_block_comment:
                    if "*/" in line:
                        line = line.split("*/", 1)[1]
                        in_block_comment = False
                    else:
                        continue

                while "/*" in line:
                    idx = line.index("/*")
                    line = line[:idx]
                    if "*/" in line[idx:]:
                        end_idx = line.index("*/", idx)
                        line = line[:idx] + line[end_idx + 2 :]
                    else:
                        in_block_comment = True
                        break

                if in_block_comment:
                    continue

                # Comentarios de línea
                if "--" in line:
                    line = line[: line.index("--")]

                # BOM y espacios
                line = line.lstrip("\ufeff").rstrip()
                if not line:
                    continue

                buffer += line + "\n"

                # Separador GO
                if line.upper().strip() == "GO":
                    stmt = buffer.replace("\nGO\n", "").strip()
                    if stmt:
                        yield stmt, progress
                    buffer = ""
                # Terminador ;
                elif ";" in line:
                    parts = buffer.split(";")
                    for part in parts[:-1]:
                        stmt = part.strip()
                        if stmt:
                            yield stmt, progress
                    buffer = parts[-1]

            # Último statement
            if buffer.strip():
                stmt = buffer.strip()
                if stmt:
                    yield stmt, 99.0


class SocketHealthChecker(IHealthChecker):
    """Verificador de salud usando sockets TCP"""

    def is_healthy(self, host: str, port: int, timeout: int) -> bool:
        import time

        start = time.time()

        while time.time() - start < timeout:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return True
            except (OSError, socket.timeout):
                pass
            time.sleep(1)

        return False


class FileLogger(ILogger):
    """Logger que escribe a archivo"""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.touch(exist_ok=True)

    def log(self, message: str, level: str = "info"):
        with open(self.log_path, "a") as f:
            f.write(f"[{level.upper()}] {message}\n")

    def log_batch(self, batch_label: str, stdout: str, stderr: str):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- {batch_label} ---\n")
            if stdout:
                f.write(stdout)
            if stderr:
                f.write(f"\nSTDERR:\n{stderr}")
