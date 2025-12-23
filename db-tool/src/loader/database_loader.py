"""Cargador de bases de datos refactorizado con SOLID"""

from pathlib import Path
from typing import Dict, Optional

from config import AppConfig
from models import ServiceDefinition
from interfaces import ISQLExecutor, IStatementParser, ILogger
from implementations import (
    SQLCmdExecutor,
    StreamingSQLParser,
    SocketHealthChecker,
    FileLogger,
)
from services import IdentityInsertService, SetupExecutor
from progress.tracker import ProgressTracker

from ._utils import _detect_encoding, _load_services

import subprocess
import time


class DatabaseLoader:
    """
    Cargador de bases de datos (Facade Pattern + Dependency Injection)

    Responsibilities:
    - Orquestar la carga de servicios
    - Coordinar dependencias
    - Gestionar progreso
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.services = self._discover_services()
        self.trackers: Dict[str, ProgressTracker] = {}
        self.identity_service = IdentityInsertService()

    def _discover_services(self) -> Dict[str, ServiceDefinition]:
        """Descubre servicios desde docker-compose"""
        raw_services = _load_services(str(self.config.paths.compose_file))

        services = {}
        for name, (port, sql_file, data_dir) in raw_services.items():
            setup_file = self.config.paths.setup_dir / f"{Path(sql_file).stem}.sql"

            services[name] = ServiceDefinition(
                name=name,
                port=port,
                sql_file=Path(sql_file),
                data_dir=Path(data_dir),
                setup_file=setup_file if setup_file.exists() else None,
            )

        return services

    def setup(self):
        """Inicializa directorios y contenedores Docker"""
        self._create_directories()
        self._start_docker_compose()
        self._initialize_trackers()

    def _create_directories(self):
        """Crea estructura de directorios necesaria"""
        self.config.paths.data_dir.mkdir(exist_ok=True)
        self.config.paths.logs_dir.mkdir(exist_ok=True)

        for service in self.services.values():
            service.data_dir.mkdir(parents=True, exist_ok=True)
            service.sql_file.parent.mkdir(parents=True, exist_ok=True)

    def _start_docker_compose(self):
        """Levanta servicios Docker"""
        subprocess.run(
            ["docker-compose", "up", "-d"],
            check=True,
            cwd=self.config.paths.compose_file.parent,
        )
        time.sleep(self.config.loader.docker_startup_wait)

    def _initialize_trackers(self):
        """Inicializa trackers de progreso"""
        for name, service in self.services.items():
            self.trackers[name] = ProgressTracker(name, service.sql_file)

    def load_database(self, service_name: str, progress=None, task_id=None) -> bool:
        """
        Carga una base de datos específica

        Args:
            service_name: Nombre del servicio a cargar
            progress: Objeto de progreso Rich (opcional)
            task_id: ID de tarea Rich (opcional)

        Returns:
            True si la carga fue exitosa
        """
        service = self.services[service_name]
        tracker = self.trackers[service_name]

        # Crear dependencias (DI)
        health_checker = SocketHealthChecker()
        logger = FileLogger(self.config.paths.logs_dir / f"{service_name}.log")
        executor = SQLCmdExecutor(service.host, service.port, self.config.database)
        parser = StreamingSQLParser(_detect_encoding)

        # Verificar salud
        if not health_checker.is_healthy(
            service.host, service.port, self.config.database.health_check_timeout
        ):
            tracker.update(0, "error", "Timeout healthy check")
            return False

        tracker.update(0, "loading")

        try:
            # Ejecutar setup
            if service.setup_file:
                setup_executor = SetupExecutor(executor, logger)
                if not setup_executor.execute_setup(service.setup_file):
                    error = "Setup failed"
                    tracker.update(0, "error", error)
                    return False

            # Cargar datos
            return self._load_data(
                service, tracker, executor, parser, logger, progress, task_id
            )

        except Exception as e:
            tracker.update(0, "error", str(e))
            logger.log(f"Exception: {e}", "error")
            return False

    def _load_data(
        self,
        service: ServiceDefinition,
        tracker: ProgressTracker,
        executor: ISQLExecutor,
        parser: IStatementParser,
        logger: ILogger,
        progress,
        task_id,
    ) -> bool:
        """Carga datos desde archivo SQL"""
        # Inicializar progreso
        if progress is not None and task_id is not None:
            self._update_progress(progress, task_id, total=100)

        batch = []
        batch_index = 0
        last_progress_value = 0

        # Procesar statements
        for stmt, pct in parser.parse(service.sql_file):
            batch.append(stmt)

            if len(batch) < self.config.loader.batch_size:
                continue

            # Ejecutar batch
            batch_index += 1
            if not self._execute_batch(
                batch, batch_index, pct, executor, logger, tracker
            ):
                return False

            batch.clear()

            # Actualizar progreso
            if progress is not None and task_id is not None:
                advance = max(0, pct - last_progress_value)
                last_progress_value = pct
                self._update_progress(progress, task_id, advance=advance)

            tracker.update(pct, "loading")

        # Ejecutar último batch
        if batch:
            batch_index += 1
            if not self._execute_batch(
                batch, batch_index, 99, executor, logger, tracker, final=True
            ):
                return False

        # Finalizar progreso
        if progress is not None and task_id is not None:
            remaining = max(0, 100 - last_progress_value)
            self._update_progress(progress, task_id, advance=remaining)

        tracker.update(100, "done")
        return True

    def _execute_batch(
        self,
        batch: list,
        batch_index: int,
        progress_pct: float,
        executor: ISQLExecutor,
        logger: ILogger,
        tracker: ProgressTracker,
        final: bool = False,
    ) -> bool:
        """Ejecuta un batch de statements"""
        batch_sql = "\n".join(batch)
        wrapped_sql = self.identity_service.wrap_batch(batch_sql)

        label = f"Batch {batch_index} {'(final)' if final else f'(progreso ~{progress_pct}%)'}"
        success, stdout, stderr = executor.execute(wrapped_sql)
        logger.log_batch(label, stdout, stderr)

        if not success:
            error = stderr or "Unknown error"
            tracker.update(progress_pct, "error", error[:200])
            return False

        return True

    @staticmethod
    def _update_progress(progress, task_id, **kwargs):
        """Actualiza progreso de forma thread-safe"""
        try:
            if hasattr(progress, "call_from_thread"):
                progress.call_from_thread(progress.update, task_id, **kwargs)
            else:
                progress.update(task_id, **kwargs)
        except Exception:
            pass
