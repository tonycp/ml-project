# db-tool — Carga de bases de datos SQL Server con progreso en vivo

Herramienta CLI para levantar contenedores de SQL Server (vía Docker Compose) y cargar backups `.sql` en paralelo mostrando una interfaz de progreso en la terminal. Es ideal para preparar entornos locales de datos a partir de varios dumps.

**Características clave:**

- Carga paralela de varias bases (`ThreadPoolExecutor`).
- Indicadores en vivo por servicio: esperando, cargando, ok, error.
- División inteligente de scripts: soporta `GO` y parsing con `sqlparse`.
- Logs detallados por servicio en `.data/logs/*.log`.
- Estructura de datos persistentes por servicio en `.data/<servicio>`.

## Estructura del paquete

- `main.py`: punto de entrada de la CLI.
- `src/loader/database_loader.py`: levanta Docker Compose, espera salud de puertos y ejecuta `sqlcmd` por lotes.
- `src/progress/renderer.py`: UI de progreso en terminal.
- `src/progress/tracker.py`: estado de progreso por servicio.
- `backup/`: directorio esperado de los archivos `.sql` a cargar.
- `.data/`: directorio generado para datos y logs.

## Requisitos

- Docker y Docker Compose (la herramienta invoca `docker-compose up -d`).
- `sqlcmd` (mssql-tools) disponible en el PATH.
- Python 3.10+ (recomendado) y el paquete `sqlparse`.

Notas:

- El código invoca explícitamente `docker-compose`. Si solo tienes `docker compose`, crea un alias o instala el binario `docker-compose`.
- La comprobación de salud usa `bash` y `/dev/tcp`, por lo que se requiere tener `bash` instalado (común en Linux).

## Instalación de dependencias (mínimas)

Si usas un entorno virtual con fish:

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install sqlparse
```

Si tu `pyproject.toml` ya lista dependencias, también puedes:

```fish
pip install -e .
```

## Configuración

La herramienta usa **Pydantic Settings** para la configuración, soportando múltiples fuentes en orden de prioridad:

1. Variables de entorno (`.env`)
2. Archivo YAML (`config.yml`)
3. Valores por defecto

### Configuración desde YAML

Crea un archivo `config.yml` basado en el ejemplo:

```bash
cp config.example.yml config.yml
```

Ejemplo de `config.yml`:

```yaml
database:
  username: sa
  password: Meteorology2025!
  connection_timeout: 300

paths:
  backup_dir: backup
  logs_dir: .data/logs

loader:
  batch_size: 100
  max_workers: 5
```

### Configuración desde variables de entorno

Crea un archivo `.env` basado en el ejemplo:

```bash
cp .env.example .env
```

Las variables usan prefijos según la sección:

- `DB_*` para configuración de base de datos
- `PATH_*` para rutas
- `LOADER_*` para opciones del cargador

Ejemplo `.env`:

```bash
DB_PASSWORD=MiPasswordSeguro123!
LOADER_BATCH_SIZE=150
LOADER_MAX_WORKERS=10
```

### Descubrimiento automático de servicios

Los servicios se descubren automáticamente desde `docker-compose.yml`. No necesitas configurarlos manualmente. La herramienta detecta:

- Servicios SQL Server con puertos expuestos
- Archivos `.sql` asociados en el directorio `backup/`
- Volúmenes de datos en `.data/<servicio>`

Para añadir una nueva base de datos:

1. Añade el servicio en `docker-compose.yml`
2. Coloca el archivo `.sql` en `backup/`
3. La herramienta lo detectará automáticamente

## Uso

Ubícate en el directorio `db-tool/` y ejecuta:

```fish
python main.py
```

El flujo es:

1. Crear directorios `.data/` y `.data/logs/` si no existen.
2. Ejecutar `docker-compose up -d` para levantar los servicios definidos en tu `docker-compose.yml`.
3. Esperar que cada puerto declarado esté accesible.
4. Leer el `.sql` de cada servicio, dividirlo en sentencias (respeta separadores `GO`) y ejecutar por lotes con `sqlcmd`.
5. Mostrar progreso en vivo y escribir logs por servicio en `.data/logs/<servicio>.log`.

Salida esperada al finalizar:

``` txt
✅ Todas las DBs cargadas en .data/*
```

Si hubo errores, se listarán por servicio y el detalle completo quedará en su archivo de log.

## Logs y datos

- Logs: `.data/logs/<servicio>.log` (stdout/stderr de cada lote ejecutado por `sqlcmd`).
- Datos: `.data/<servicio>` (directorio de datos persistentes del contenedor/servicio).

## Formato de scripts SQL

- Se soporta el separador de lotes `GO` (insensible a mayúsculas/minúsculas y espacios) y la división por sentencias con `sqlparse`.
- Los `;` finales se toleran y se recortan al dividir.

## Solución de problemas

- `Timeout healthy check`:
  - Verifica que el contenedor esté arriba: `docker ps`.
  - Revisa puertos y mapeos en `docker-compose.yml`.
  - Asegúrate de no tener otro proceso ocupando el puerto.
- `sqlcmd: command not found`:
  - Instala las herramientas de SQL Server (mssql-tools) y añade `sqlcmd` al PATH.
- `No statements found`:
  - Valida que el `.sql` no esté vacío y su codificación sea UTF-8 (el loader abre con `errors="replace"`).
- Errores en ejecución SQL:
  - Revisa `.data/logs/<servicio>.log` para el lote y mensaje exacto (`STDERR` se registra).

## Personalización

- Tamaño de lote (`batch_size`): por defecto 50 sentencias por ejecución de `sqlcmd`. Modifica en `database_loader.py` si necesitas lotes más grandes/pequeños.
- Concurrencia: `max_workers=5` en `main.py`. Ajusta según tu máquina y número de servicios.
- UI de progreso: `src/progress/renderer.py` controla colores y ancho de barra.

## Licencia

Este proyecto se publica bajo la licencia incluida en `LICENSE` en la raíz del repositorio.
