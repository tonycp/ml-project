from .cli import CLI


def main():
    """Punto de entrada principal"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Herramienta de carga de bases de datos SQL Server"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="load",
        choices=["load", "setup", "list"],
        help="Comando a ejecutar (default: load)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Ruta a archivo de configuración YAML (default: config.yml si existe)",
    )
    parser.add_argument(
        "--start-percent",
        type=float,
        default=0.0,
        help="Porcentaje desde donde reanudar la carga (0-100, ej: 25.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Número de sentencias por lote (override config)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Número máximo de workers paralelos (override config)",
    )
    parser.add_argument(
        "--service",
        type=str,
        default=None,
        help="Cargar solo un servicio específico (ej: varadero, casablanca_2020)",
    )

    args = parser.parse_args()

    cli = CLI(
        config_path=args.config,
        start_percent=args.start_percent,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        service_filter=args.service,
    )
    cli.run(args.command)


if __name__ == "__main__":
    main()
