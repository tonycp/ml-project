# ETL Tool - Multi-DB Data Extraction

Herramienta ETL modular para extraer datos relevantes de **SQL Server** y **Postgres** → **SQLite limpio**. Usa solo DTOs de campos específicos, no esquemas completos.

**Tech Stack:**

``` yaml
pydantic Settings + pydantic-settings-logging
pyodbc (SQL Server) + psycopg (Postgres)
SQLAlchemy + advanced-alchemy
Dependency Injector (DI)
pytest + uv (packaging)
```

## 📁 **Project Structure**

``` bash
📁 etl-tool/
├── main.py                 # CLI entrypoint
├── pyproject.toml          # uv dependencies
├── src/
│   ├── config/             # Pydantic Settings
│   ├── connection/         # DB engines/factories
│   ├── container/          # DI wiring
│   ├── interface/          # Protocols/ABCs
│   ├── model/              # DTOs source/target
│   ├── schema/             # SQLAlchemy SQLite models
│   └── service/            # Extract/Transform/Load
└── test/                   # pytest mirror structure
```

## 🔧 **Key Dependencies**

``` toml
pydantic-settings>=2.0,<3.0
pydantic-settings-logging>=0.1,<1.0
dependency-injector>=4.0,<5.0
sqlalchemy>=2.0,<3.0
advanced-alchemy>=0.19,<0.20
pyodbc>=5.0,<6.0
psycopg[binary,pool]>=3.1,<4.0
pytest>=8.0,<9.0
```

## 📋 **Layer Responsibilities**

| Capa              | Responsabilidad       | Ejemplo                               |
|-------------------|-----------------------|---------------------------------------|
| `config/`         | Settings tipados      | `AppSettings.from_env()`              |
| `interface/`      | Contratos gateways    | `SqlServerSourceInterface`            |
| `model/source/`   | DTOs crudos origen    | `SqlServerOrderDTO`                   |
| `model/target/`   | DTOs limpios destino  | `CleanOrderRecord`                    |
| `service/`        | Lógica ETL            | `ExtractService`, `TransformService`  |
| `connection/`     | Engines/conexiones    | `create_sqlserver_engine()`           |
