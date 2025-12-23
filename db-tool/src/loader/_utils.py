from pathlib import Path
from typing import Dict

import yaml


def _detect_encoding(sql_path: Path) -> str:
    """Detecta encoding básico: UTF-16 (LE/BE) o UTF-8(-sig)."""
    with open(sql_path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if head.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if b"\x00" in head:
        # Muchas herramientas de SQL Server guardan en UTF-16-LE sin BOM
        return "utf-16-le"
    return "utf-8"


def _load_services(
    compose_file: str = "docker-compose.yml",
) -> Dict[str, tuple]:
    """Extrae configuración de servicios desde docker-compose.yml"""
    services_dict = {}

    try:
        with open(compose_file, "r") as f:
            compose = yaml.safe_load(f)

        if not compose or "services" not in compose:
            return services_dict

        for service_name, config in compose["services"].items():
            if service_name == "init-data":
                continue  # Skip init service

            if "ports" not in config:
                continue

            # Extrae puerto host (primer puerto listado)
            ports = config["ports"]
            if not ports:
                continue

            port_mapping = ports[0]  # e.g., "1433:1433"
            if isinstance(port_mapping, str):
                host_port = int(port_mapping.split(":")[0])
            else:
                host_port = port_mapping.get("target", 1433)

            # Extrae path del SQL desde volumes
            sql_file = None
            data_dir = None

            if "volumes" in config:
                for volume in config["volumes"]:
                    if isinstance(volume, str):
                        # Format: "./.data/varadero:/var/opt/mssql/data"
                        parts = volume.split(":")
                        if len(parts) == 2 and "/var/opt/mssql/data" in parts[1]:
                            data_dir = parts[0]

            # Busca el SQL file correspondiente en backup/
            if data_dir:
                # Intenta encontrar .sql usando service_name como pista
                candidates = [
                    f"backup/{service_name}.sql",
                    f"backup/cb_{service_name.split('_')[-1]}.sql",
                    f"backup/{service_name.replace('casablanca_', 'cb_')}.sql",
                    f"backup/{service_name.replace('_', '-')}.sql",
                ]

                for pattern in candidates:
                    if Path(pattern).exists():
                        sql_file = pattern
                        break

                # Si aún no lo encontró, busca cualquier .sql con similitud de nombre
                if not sql_file:
                    backup_dir = Path("backup")
                    if backup_dir.exists():
                        for sql_path in backup_dir.glob("*.sql"):
                            if (
                                service_name.lower() in sql_path.name.lower()
                                or sql_path.name.lower() in service_name.lower()
                            ):
                                sql_file = str(sql_path)
                                break

            if sql_file and data_dir:
                services_dict[service_name] = (host_port, sql_file, data_dir)

    except Exception as e:
        print(f"Error cargando docker-compose.yml: {e}")

    return services_dict
