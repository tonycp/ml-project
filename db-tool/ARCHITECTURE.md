# Estructura del Proyecto

## 📁 Organización de Archivos

``` sh
db-tool/
├── main.py                  # Punto de entrada
├── docker-compose.yml       # Definición de servicios
├── pyproject.toml          # Dependencias
├── README.md               # Documentación principal
├── ARCHITECTURE.md         # Este archivo
│
├── backup/                 # Archivos SQL
│   ├── start/             # Scripts de inicialización (CREATE DATABASE)
│   └── *.sql              # Backups de datos
│
├── .data/                 # Generado en runtime
│   ├── logs/              # Logs por servicio
│   └── */                 # Datos de contenedores
│
└── src/
    ├── config.py          # Configuración centralizada
    ├── interfaces.py      # Contratos (ABCs)
    ├── implementations.py # Implementaciones concretas
    ├── services.py        # Lógica de negocio
    ├── models.py          # Modelos de dominio
    │
    ├── cli/              # Interfaz de línea de comandos
    │   ├── __init__.py
    │   ├── main.py       # CLI principal con comandos
    │   └── progress.py   # UI de progreso
    │
    ├── loader/           # Cargador de datos
    │   ├── __init__.py
    │   ├── database_loader_v2.py  # Loader refactorizado
    │   └── _utils.py              # Utilidades
    │
    └── progress/         # Tracking de progreso
        ├── __init__.py
        └── tracker.py    # Estado de progreso
```

## 🏗️ Principios SOLID Aplicados

### Single Responsibility Principle (SRP)

Cada clase tiene una única responsabilidad:

- **`SQLCmdExecutor`**: Solo ejecuta SQL
- **`StreamingSQLParser`**: Solo parsea SQL
- **`SocketHealthChecker`**: Solo verifica salud
- **`FileLogger`**: Solo registra logs
- **`IdentityInsertService`**: Solo maneja IDENTITY_INSERT
- **`SetupExecutor`**: Solo ejecuta setups
- **`DatabaseLoader`**: Solo orquesta (Facade)

### Open/Closed Principle (OCP)

Extensible sin modificar código existente:

```python
# Agregar nuevo ejecutor sin modificar DatabaseLoader
class AzureExecutor(ISQLExecutor):
    def execute(self, sql: str, database: str = "master"):
        # Implementación para Azure SQL
        pass

# Agregar nuevo parser sin modificar DatabaseLoader  
class RegexParser(IStatementParser):
    def parse(self, sql_path: Path):
        # Implementación con regex puro
        pass
```

### Liskov Substitution Principle (LSP)

Todas las implementaciones son intercambiables:

```python
# Cualquier ISQLExecutor funciona
executor: ISQLExecutor = SQLCmdExecutor(...)
executor: ISQLExecutor = AzureExecutor(...)

# Cualquier ILogger funciona
logger: ILogger = FileLogger(...)
logger: ILogger = ConsoleLogger(...)
```

### Interface Segregation Principle (ISP)

Interfaces pequeñas y específicas:

- `ISQLExecutor`: Solo ejecutar
- `IStatementParser`: Solo parsear
- `IHealthChecker`: Solo verificar salud
- `ILogger`: Solo registrar

### Dependency Inversion Principle (DIP)

Dependemos de abstracciones, no de implementaciones:

```python
class DatabaseLoader:
    def load_database(self, ...):
        # Depende de interfaces, no de clases concretas
        executor: ISQLExecutor = SQLCmdExecutor(...)
        parser: IStatementParser = StreamingSQLParser(...)
        logger: ILogger = FileLogger(...)
```

## 🔧 Patrones de Diseño

### Facade Pattern

`DatabaseLoader` actúa como fachada que simplifica la interacción con múltiples subsistemas:

```python
loader = DatabaseLoader()  # Fachada
loader.setup()             # Orquesta múltiples operaciones
loader.load_database(...)  # Coordina parser, executor, logger, etc.
```

### Strategy Pattern

Intercambia implementaciones en runtime:

```python
# Strategy para ejecutar SQL
executor = SQLCmdExecutor(...)  # Estrategia A
executor = AzureExecutor(...)   # Estrategia B

# Strategy para parsear
parser = StreamingSQLParser(...)  # Estrategia A
parser = RegexParser(...)         # Estrategia B
```

### Template Method (implícito en IStatementParser)

Define esqueleto de algoritmo, subclases implementan pasos:

```python
class IStatementParser(ABC):
    @abstractmethod
    def parse(self, sql_path: Path):
        """Template method: define cómo parsear"""
        pass
```

### Dependency Injection

Inyectamos dependencias en constructores:

```python
class SQLCmdExecutor:
    def __init__(self, host: str, port: int, config: DatabaseConfig):
        # Inyección de config
        self.config = config

class SetupExecutor:
    def __init__(self, executor: ISQLExecutor, logger: ILogger):
        # Inyección de dependencias
        self.executor = executor
        self.logger = logger
```

## 📦 Configuración Centralizada

Toda configuración en `src/config.py`:

```python
@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig   # Credenciales, timeouts
    paths: PathConfig          # Rutas de archivos
    loader: LoaderConfig       # Batch size, workers
```

**Beneficios:**

- Single Source of Truth
- Fácil testing (mocks)
- Configuración desde ENV (futuro)

## 🧪 Testabilidad

Gracias a DI e interfaces, testing es trivial:

```python
# Mock de executor
class MockExecutor(ISQLExecutor):
    def execute(self, sql, database="master"):
        return True, "OK", ""

# Test
def test_load_database():
    config = AppConfig()
    loader = DatabaseLoader(config)
    
    # Inyecta mock
    executor = MockExecutor()
    parser = MockParser()
    
    result = loader.load_database("test_service")
    assert result == True
```

## 🚀 Comandos CLI

```bash
# Cargar todas las bases (default)
python main.py load

# Solo setup (directorios + Docker)
python main.py setup

# Listar servicios disponibles
python main.py list
```

## 🔄 Flujo de Ejecución

``` sh
main.py
  └─> CLI.run("load")
      └─> DatabaseLoader.__init__()
          └─> _discover_services()  # Lee docker-compose.yml
      └─> DatabaseLoader.setup()
          ├─> _create_directories()
          ├─> _start_docker_compose()
          └─> _initialize_trackers()
      └─> ThreadPoolExecutor
          └─> DatabaseLoader.load_database(service_name)
              ├─> SocketHealthChecker.is_healthy()
              ├─> SetupExecutor.execute_setup()  # CREATE DATABASE
              └─> _load_data()
                  ├─> StreamingSQLParser.parse()
                  ├─> IdentityInsertService.wrap_batch()
                  ├─> SQLCmdExecutor.execute()
                  └─> FileLogger.log_batch()
```

## 📝 Extensibilidad

### Agregar nuevo comando CLI

```python
# src/cli/main.py
class CLI:
    def cmd_backup(self):
        """Nuevo comando: hace backup de DBs"""
        # Implementación
        pass
```

### Agregar nuevo executor

```python
# src/implementations.py
class PostgreSQLExecutor(ISQLExecutor):
    def execute(self, sql, database="postgres"):
        # Implementación con psycopg2
        pass
```

### Configuración desde .env

```python
# src/config.py
@classmethod
def from_env(cls) -> "AppConfig":
    import os
    return cls(
        database=DatabaseConfig(
            username=os.getenv("DB_USER", "sa"),
            password=os.getenv("DB_PASS", "default"),
        )
    )
```

## 🎯 Beneficios de la Refactorización

1. **Mantenibilidad**: Código modular y desacoplado
2. **Testabilidad**: Interfaces facilitan mocking
3. **Extensibilidad**: Agregar features sin romper existente
4. **Claridad**: Responsabilidades claras
5. **Reutilización**: Componentes independientes
6. **Configuración**: Centralizada y tipada
7. **CLI**: Comandos separados y organizados
