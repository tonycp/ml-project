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

Si usas un entorno virtual con sh:

```sh
uv main .venv
source .venv/bin/activate.sh
pip install sqlparse
```

Si tu `pyproject.toml` ya lista dependencias, también puedes:

```sh
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

### Uso básico

Ubícate en el directorio `db-tool/` y ejecuta:

```sh
uv main load
```

O simplemente:

```sh
uv main
```

### Comandos disponibles

- `load` (default): Carga todas las bases de datos
- `setup`: Solo inicializa directorios y Docker Compose
- `list`: Lista servicios disponibles

### Opciones de línea de comandos

```sh
uv main load [opciones]
```

**Opciones disponibles:**

- `--service NOMBRE`: Cargar solo un servicio específico
  - Ejemplo: `--service varadero` o `--service casablanca_2020`
  - Si no se especifica, carga todas las bases de datos
  - Si el servicio no existe, muestra error y lista disponibles

- `--start-percent PERCENT`: Reanudar desde un porcentaje específico (0-100)
  - Ejemplo: `--start-percent 25.5` continúa desde el 25.5%
  - Útil para reanudar cargas interrumpidas
  
- `--batch-size N`: Número de sentencias SQL por lote (override config)
  - Ejemplo: `--batch-size 150`
  - Default: valor del config (normalmente 100)
  
- `--max-workers N`: Número máximo de workers paralelos (override config)
  - Ejemplo: `--max-workers 3`
  - Default: valor del config (normalmente 5)

- `-c, --config PATH`: Ruta a archivo de configuración YAML alternativo
  - Ejemplo: `-c custom-config.yml`

### Ejemplos

**Carga normal:**

```sh
uv main load
```

**Cargar solo una base de datos:**

```sh
uv main load --service varadero
```

**Reanudar una base específica desde el 50%:**

```sh
uv main load --service casablanca_2020 --start-percent 50
```

**Reanudar desde el 50%:**

```sh
uv main load --start-percent 50
```

**Lotes más grandes con menos workers:**

```sh
uv main load --batch-size 200 --max-workers 3
```

**Usar configuración personalizada:**

```sh
uv main load -c production.yml
```

### Flujo de ejecución

1. Crear directorios `.data/` y `.data/logs/` si no existen
2. Ejecutar `docker-compose up -d` para levantar servicios
3. Esperar que cada puerto esté accesible
4. Leer y parsear archivos `.sql` (respeta separadores `GO`)
5. Ejecutar sentencias por lotes con `sqlcmd`
6. Mostrar progreso en vivo con barras Rich
7. Escribir logs detallados en `.data/logs/<servicio>.log`

**Salida esperada:**

``` out
✅ Todas las DBs cargadas exitosamente
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
