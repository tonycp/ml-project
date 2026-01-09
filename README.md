# ML Project - Sistema Integral de Procesamiento de Datos y Forecasting Aéreo

Proyecto integral de Machine Learning que combina procesamiento de noticias, extracción de eventos, gestión de bases de datos y forecasting de tráfico aéreo para el espacio aéreo cubano.

## 🎯 Visión General

Este proyecto implementa un ecosistema completo para el análisis predictivo del tráfico aéreo cubano, integrando múltiples fuentes de datos incluyendo:

- **Datos ATC/ATFM**: Información de control de tráfico aéreo y gestión de flujo
- **Noticias**: Extracción de eventos relevantes que afectan el tráfico aéreo
- **Datos Meteorológicos**: Condiciones climáticas que impactan las operaciones
- **Datos Externos**: Eventos sociales, políticos y económicos

## 🏗️ Arquitectura del Proyecto

```
ml-project/
├── models/                    # 🚀 Sistema de Forecasting de Aeronaves
│   ├── aircraft_forecasting_optuna.py
│   ├── data_loader.py         # Carga ATC/ATFM
│   ├── preprocessing.py       # Limpieza y feature engineering
│   ├── model.py              # ARIMA, Prophet, LSTM, Ensemble
│   ├── train.py              # Entrenamiento automatizado
│   └── evaluate.py           # Evaluación con métricas
├── event-tool/                # 📰 Extracción de Eventos de Noticias
│   ├── Event_extractor/      # Pipeline NLP en español
│   ├── process_all_news.py   # Procesamiento masivo
│   └── examples/             # Casos de uso
├── etl-tool/                  # 🔧 ETL Multi-Base de Datos
│   ├── src/config/           # Configuración Pydantic
│   ├── src/connection/       # SQL Server + Postgres
│   └── src/service/          # Extract/Transform/Load
├── db-tool/                   # 🗄️ Carga Automatizada de BD
│   ├── src/loader/           # Docker + SQL Server
│   └── src/progress/         # UI de progreso en vivo
├── docs/                      # 📚 Documentación Técnica
│   ├── informe_tecnico.pdf   # Análisis completo
│   └── referencias.bib       # Bibliografía
└── examples/                  # 💡 Ejemplos y Comparaciones
    ├── training/             # Scripts de entrenamiento
    └── comparisons/          # Análisis comparativos
```

## 🚀 Características Principales

### Sistema de Forecasting de Aeronaves (`models/`)

- **Modelos ML**: ARIMA, Prophet, Random Forest, LSTM y Ensemble
- **Target**: Predicción del número total de aeronaves por día/hora
- **Features**: Temporales, lags, estadísticas móviles, estacionalidad
- **Datos**: Resúmenes ATC diarios, ATFM horarios, rutas mensuales
- **Optimización**: Hyperparameter tuning con Optuna
- **Evaluación**: MAE, RMSE, MAPE, R² con validación cruzada

### Extracción de Eventos (`event-tool/`)

- **NLP en Español**: Procesamiento de noticias con spaCy
- **Tipos de Eventos**: Cultural, Deportivo, Meteorológico, Político, Económico, Social, Incidente
- **Clasificación de Sentimiento**: Positivo, Negativo, Neutral
- **Extracción de Fechas**: Explícitas, relativas, rangos
- **Pipeline Completo**: Desde texto hasta eventos estructurados

### ETL Multi-Base de Datos (`etl-tool/`)

- **Fuentes Múltiples**: SQL Server y PostgreSQL
- **DTOs Específicos**: Solo campos relevantes, no esquemas completos
- **Inyección de Dependencias**: Arquitectura modular con DI
- **Transformación**: Limpieza y normalización automática

### Carga Automatizada de BD (`db-tool/`)

- **Docker Compose**: Levanta contenedores SQL Server automáticamente
- **Progreso en Vivo**: UI terminal con barras de progreso
- **Ejecución Paralela**: Múltiples bases simultáneamente
- **Logs Detallados**: Registro completo por servicio

## 🛠️ Instalación y Configuración

### Prerrequisitos

- Python 3.8+
- Docker y Docker Compose
- sqlcmd (mssql-tools)
- Git

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tonycp/ml-project.git
cd ml-project
```

### 2. Instalar Dependencias Principales

```bash
pip install -r requirements.txt
```

### 3. Descargar Modelo de spaCy

```bash
python -m spacy download es_core_news_lg
```

### 4. Configurar Subproyectos

#### ETL Tool
```bash
cd etl-tool
pip install -e .
```

#### DB Tool
```bash
cd db-tool
cp .env.example .env
# Editar .env con tus credenciales
pip install -e .
```

## 🚀 Inicio Rápido

### 1. Forecasting de Aeronaves

```bash
# Ejecutar tests básicos
python models/test_basic.py

# Ejemplo completo de uso
python models/example_usage.py

# Entrenar modelos
python models/train.py --data-type daily_atc --models arima prophet lstm --save-models

# Evaluar rendimiento
python models/evaluate.py --horizons 1 7 14 --output-dir evaluation_results
```

### 2. Extracción de Eventos

```python
from Event_extractor import EventExtractionPipeline, NewsContent
from datetime import datetime

# Crear pipeline
pipeline = EventExtractionPipeline()

# Procesar noticia
news = NewsContent(
    text="El festival de música se realizará del 10 al 15 de enero de 2025",
    id="noticia_001",
    date=datetime(2024, 12, 1)
)

# Extraer eventos
events = pipeline.extract_events(news)
for event in events:
    print(f"Fecha: {event.date}, Tipo: {event.event_type}, Sentimiento: {event.sentiment}")
```

### 3. Procesamiento Masivo de Noticias

```bash
cd event-tool
python process_all_news.py
```

### 4. Carga de Bases de Datos

```bash
cd db-tool
uv main load
# O para una base específica
uv main load --service varadero
```

### 5. ETL de Datos

```bash
cd etl-tool
uv main
```

## 📊 Modelos y Algoritmos

### Forecasting de Aeronaves

| Modelo | Ventajas | Casos de Uso | Performance Típica |
|--------|----------|--------------|-------------------|
| **ARIMA** | Simple, interpretable | Tendencias lineales, horizontes cortos | MAE: 8-12, R²: 0.85-0.90 |
| **Prophet** | Maneja estacionalidad automática | Datos con patrones estacionales | MAE: 7-10, R²: 0.87-0.92 |
| **Random Forest** | Robusto, interpretable | Features complejas, no lineales | MAE: 6-10, R²: 0.88-0.93 |
| **LSTM** | Captura patrones complejos | Horizontes largos, dependencias temporales | MAE: 6-9, R²: 0.89-0.94 |
| **Ensemble** | Combina fortalezas | Mayor robustez general | MAE: 5-8, R²: 0.91-0.95 |

### Clasificación de Eventos

- **Tipos Principales**: CULTURAL, DEPORTIVO, METEOROLOGICO, POLITICO, ECONOMICO, SOCIAL, INCIDENTE, REGULACION
- **Sentimiento**: POSITIVO (✅), NEGATIVO (❌), NEUTRAL (⚪)
- **Confianza**: Umbral configurable (default: 0.5)

## 📈 Datos y Features

### Fuentes de Datos Principales

1. **Datos ATC Diarios**: Resúmenes de operaciones de control de tráfico aéreo
2. **Datos ATFM Horarios**: Vuelos agrupados por hora y área
3. **Datos Mensuales por Ruta**: Vuelos mensuales por ruta específica
4. **Noticias**: Eventos que pueden afectar el tráfico aéreo
5. **Datos Meteorológicos**: Condiciones climáticas

### Ingeniería de Features

- **Temporales**: Día de semana, mes, trimestre, fin de semana
- **Estacionales**: Codificación sinusoidal de patrones cíclicos
- **Lags**: Valores anteriores (1, 7, 14, 30 días)
- **Móviles**: Estadísticas móviles (media, std, min, max)
- **Festivos**: Indicadores de días festivos cubanos
- **Eventos**: Conteo de eventos por tipo y sentimiento
- **Meteorológicos**: Temperatura, humedad, viento, precipitación

## 🔧 Configuración Avanzada

### Configuración de Modelos

```python
from models import ModelConfig

config = ModelConfig()
config.models['lstm'] = {
    'sequence_length': 30,
    'hidden_units': 128,
    'dropout_rate': 0.3,
    'epochs': 200
}
```

### Configuración de Features

```python
config.feature_config = {
    'temporal_features': True,
    'lag_features': [1, 7, 14, 30],
    'rolling_features': [7, 14, 30],
    'seasonal_features': True,
    'holiday_features': True,
    'covid_adjustment': True
}
```

### Configuración de Base de Datos

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

## 📋 Métricas de Evaluación

### Forecasting

- **MAE** (Mean Absolute Error): Error absoluto medio
- **RMSE** (Root Mean Square Error): Raíz del error cuadrático medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio
- **R²**: Coeficiente de determinación

### Clasificación de Eventos

- **Precision**: Proporción de eventos clasificados correctamente
- **Recall**: Proporción de eventos reales detectados
- **F1-Score**: Media armónica de precision y recall
- **Accuracy**: Proporción total de clasificaciones correctas

## 📚 Documentación Técnica

- **Informe Técnico**: `docs/informe_tecnico.pdf` - Análisis completo del sistema
- **Referencias**: `docs/referencias.bib` - Bibliografía académica
- **Arquitectura**: Documentación detallada en cada subproyecto

## 🧪 Ejemplos y Casos de Uso

### Scripts de Entrenamiento

```bash
# Entrenamiento básico
python examples/training/train_sklearn_model.py

# Comparación de modelos
python examples/comparisons/compare_sklearn_models.py

# Análisis de sentimiento
python examples/comparisons/compare_sentiment_classifiers.py
```

### Visualización de Resultados

```python
# Visualizar estudio Optuna
python models/visualize_study.py

# Resultados de entrenamiento
python models/visualize_results.py
```

## 🐛 Troubleshooting

### Problemas Comunes

1. **ImportError**: Verificar instalación de dependencias
2. **MemoryError**: Reducir sequence_length en LSTM
3. **Docker Issues**: Verificar instalación de Docker y Docker Compose
4. **sqlcmd no encontrado**: Instalar mssql-tools
5. **Modelos spaCy**: Descargar es_core_news_lg

### Logs y Debugging

```bash
# Logging detallado en forecasting
python models/train.py --log-level DEBUG --log-file training.log

# Logs de base de datos
cat .data/logs/<servicio>.log

# Logs de ETL
tail -f logs/etl.log
```

## 🛣️ Roadmap

### Implementado ✅
- [x] Sistema de forecasting de aeronaves con ML
- [x] Extracción de eventos de noticias en español
- [x] Pipeline ETL multi-base de datos
- [x] Carga automatizada de bases de datos SQL Server
- [x] Optimización de hiperparámetros con Optuna

### Planeado 📋
- [ ] API REST para procesamiento en línea
- [ ] Dashboard interactivo con Streamlit/Dash
- [ ] Soporte para más idiomas en extracción de eventos
- [ ] Extracción de ubicaciones geográficas
- [ ] Identificación de actores/entidades involucradas
- [ ] Integración con datos en tiempo real
- [ ] Modelo de deep learning más avanzado (Transformers)

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Estándares de Código

- Usar **Black** para formateo (line-length: 100)
- Seguir **PEP 8** para estilo
- Añadir **type hints** donde sea posible
- Escribir **tests** para nuevas funcionalidades
- Documentar con **docstrings** siguiendo Google Style

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

- **Proyecto**: [https://github.com/tonycp/ml-project](https://github.com/tonycp/ml-project)
- **Issues**: [https://github.com/tonycp/ml-project/issues](https://github.com/tonycp/ml-project/issues)

## 🙏 Agradecimientos

- **spaCy** por el procesamiento de lenguaje natural en español
- **Prophet** de Facebook por el forecasting con estacionalidad
- **Optuna** por la optimización de hiperparámetros
- **SQLAlchemy** por el ORM multi-base de datos
- **Docker** por la contenerización de servicios de base de datos

---

**Nota**: Este proyecto es parte de un trabajo académico para la Universidad y está diseñado para ser un caso de estudio completo de un sistema de Machine Learning en producción.
