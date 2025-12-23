from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from pathlib import Path


from loader.database_loader import DatabaseLoader
from config import AppConfig

from .progress import create_progress_ui

import sys


class CLI:
    """Command Line Interface principal"""

    def __init__(self, config_path: str | Path | None = None):
        """
        Inicializa CLI con configuración

        Args:
            config_path: Ruta a config.yml. Si None, usa defaults + .env
        """
        if config_path and Path(config_path).exists():
            self.config = AppConfig.from_yaml(config_path)
        else:
            # Intenta cargar config.yml si existe, sino usa defaults + .env
            default_config = Path("config.yml")
            if default_config.exists():
                self.config = AppConfig.from_yaml(default_config)
            else:
                self.config = AppConfig()

        self.console = Console()

    def run(self, command: str = "load"):
        """Ejecuta comando especificado"""
        commands = {
            "load": self.cmd_load,
            "setup": self.cmd_setup,
            "list": self.cmd_list,
        }

        cmd_func = commands.get(command)
        if not cmd_func:
            self.console.print(f"[red]Comando desconocido: {command}[/red]")
            self.console.print(f"Comandos disponibles: {', '.join(commands.keys())}")
            sys.exit(1)

        try:
            cmd_func()
        except KeyboardInterrupt:
            self.console.print("\n[red]Interrumpido[/red]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    def cmd_setup(self):
        """Solo inicializa directorios y Docker"""
        loader = DatabaseLoader(self.config)
        self.console.print("[cyan]Inicializando...[/cyan]")
        loader.setup()
        self.console.print("[green]✅ Setup completo[/green]")

    def cmd_list(self):
        """Lista servicios disponibles"""
        loader = DatabaseLoader(self.config)

        self.console.print("\n[bold]Servicios disponibles:[/bold]\n")
        for name, service in loader.services.items():
            self.console.print(f"  • [cyan]{name}[/cyan]")
            self.console.print(f"    Puerto: {service.port}")
            self.console.print(f"    SQL: {service.sql_file}")
            self.console.print(f"    Setup: {service.setup_file or 'N/A'}")
            self.console.print()

    def cmd_load(self):
        """Carga todas las bases de datos"""
        loader = DatabaseLoader(self.config)
        loader.setup()

        with create_progress_ui(self.console) as progress:
            tasks = {name: progress.add_task(name, total=0) for name in loader.services}

            with ThreadPoolExecutor(
                max_workers=self.config.loader.max_workers
            ) as executor:
                futures = {
                    executor.submit(
                        loader.load_database, name, progress, tasks[name]
                    ): name
                    for name in loader.services
                }

                results = {}
                for future in as_completed(futures):
                    name = futures[future]
                    results[name] = future.result()

        # Resumen
        self._print_summary(loader, results)

    def _print_summary(self, loader: DatabaseLoader, results: dict):
        """Imprime resumen de resultados"""
        success = all(results.values())

        if success:
            self.console.print("[green]✅ Todas las DBs cargadas exitosamente[/green]")
            sys.exit(0)
        else:
            self.console.print("[red]❌ Errores encontrados:[/red]")
            for name, ok in results.items():
                if not ok:
                    tracker = loader.trackers[name]
                    state = tracker.get_state()
                    self.console.print(f"  [red]✗ {name}:[/red] {state.error}")
            sys.exit(1)
