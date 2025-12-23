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

    args = parser.parse_args()

    cli = CLI(config_path=args.config)
    cli.run(args.command)


if __name__ == "__main__":
    main()
