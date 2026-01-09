# Arquitectura del Event Extractor

Este documento describe la arquitectura y organización del paquete Event Extractor.

## Visión General

Event Extractor es una librería Python modular diseñada para extraer eventos estructurados de noticias en español. El paquete utiliza procesamiento de lenguaje natural (NLP) y clasificación basada en reglas para identificar fechas, tipos de eventos y otra información relevante.

## Estructura del Proyecto

```
Event_extractor/
├── __init__.py                    # Punto de entrada principal del paquete
├── models/                        # Modelos de datos
│   ├── __init__.py
│   └── event.py                   # Event, EventType
├── extractors/                    # Extractores de información
│   ├── __init__.py
│   └── date_extractor.py          # DateExtractor
├── classifiers/                   # Clasificadores de eventos
│   ├── __init__.py
│   └── event_type_classifier.py   # EventTypeClassifier
├── pipeline/                      # Pipeline de procesamiento
│   ├── __init__.py
│   └── event_pipeline.py          # EventExtractionPipeline, EventAggregator
├── utils/                         # Utilidades
│   ├── __init__.py
│   └── text_preprocessor.py       # Funciones de preprocesamiento
└── event_extractor.py             # Archivo heredado (compatibilidad)
```

## Componentes Principales

### 1. Modelos de Datos (`models/`)

#### NewsContent
Representa el contenido completo de una noticia:
- Texto de la noticia
- Metadata asociada
- Datos originales sin procesar

#### Event
Representa un evento extraído:
- Fecha del evento
- Tipo de evento (EventType)
- Título y descripción
- Nivel de confianza

#### EventType (Enum)
Tipos de eventos soportados:
- CULTURAL, DEPORTIVO, METEOROLOGICO
- POLITICO, ECONOMICO, SOCIAL
- INCIDENTE, REGULACION, OTRO

### 2. Extractores (`extractors/`)

#### DateExtractor
Extrae fechas del texto usando múltiples estrategias:
- **Fechas explícitas**: "25 de diciembre de 2024"
- **Fechas numéricas**: "25/12/2024"
- **Fechas relativas**: "mañana", "próxima semana"
- **Rangos de fechas**: "del 10 al 15 de enero"
- **NER con spaCy**: Detección de entidades de fecha

**Característica importante**: Los rangos de fechas generan eventos separados para la fecha de inicio y fin.

### 3. Clasificadores (`classifiers/`)

#### EventTypeClassifier
Clasifica el tipo de evento usando palabras clave:
- Diccionario extensible de palabras clave por tipo
- Puntuación basada en coincidencias
- Soporte para clasificación múltiple (top-k)
- Posibilidad de añadir palabras clave personalizadas

### 4. Pipeline (`pipeline/`)

#### EventExtractionPipeline
Pipeline principal que orquesta todo el proceso:
1. Recibe NewsContent
2. Extrae fechas del texto
3. Clasifica el tipo de evento
4. Crea objetos Event para cada fecha
5. Devuelve lista de eventos extraídos

Características:
- Procesamiento en batch
- Configuración de confianza mínima
- Personalización de palabras clave

#### EventAggregator
Utilidades para procesar listas de eventos:
- Eliminar duplicados
- Filtrar por rango de fechas
- Filtrar por tipo de evento
- Ordenar eventos

### 5. Utilidades (`utils/`)

#### text_preprocessor
Funciones para preprocesamiento de texto:
- Tokenización
- Lematización
- Lectura de archivos de texto

## Flujo de Datos

```
┌─────────────────┐
│   NewsContent   │  (Entrada: texto de noticia + metadata)
└────────┬────────┘
         │
         v
┌─────────────────────────────┐
│ EventExtractionPipeline     │
├─────────────────────────────┤
│ 1. DateExtractor            │  → Extrae fechas
│ 2. EventTypeClassifier      │  → Clasifica tipo
│ 3. Genera Events            │  → Crea eventos
└────────┬────────────────────┘
         │
         v
┌─────────────────┐
│  List[Event]    │  (Salida: lista de eventos)
└────────┬────────┘
         │
         v
┌─────────────────────────────┐
│   EventAggregator           │  (Opcional: filtrado/ordenamiento)
├─────────────────────────────┤
│ - remove_duplicates()       │
│ - filter_by_date_range()    │
│ - filter_by_type()          │
│ - sort_by_date()            │
└─────────────────────────────┘
```

## Decisiones de Diseño

### 1. Eventos Separados para Rangos de Fechas

**Decisión**: Cuando se detecta un rango de fechas (ej: "del 10 al 15 de enero"), se crean eventos separados para cada fecha límite.

**Razón**: 
- Permite modelar el inicio y fin de eventos como entidades distintas
- Facilita análisis temporal (ej: "eventos que comienzan en enero")
- Mejor granularidad para agregaciones y filtros

### 2. Clasificación Basada en Palabras Clave

**Decisión**: Usar clasificación basada en reglas con palabras clave en lugar de ML.

**Razón**:
- Más interpretable y controlable
- No requiere datos de entrenamiento
- Fácilmente extensible por el usuario
- Rendimiento adecuado para casos comunes
- Se puede mejorar con ML en el futuro

### 3. Modularidad

**Decisión**: Separar componentes en módulos independientes.

**Razón**:
- Permite usar componentes individualmente
- Facilita testing y mantenimiento
- Posibilita reemplazar componentes (ej: cambiar clasificador)
- Extensible para nuevas funcionalidades

### 4. Uso de spaCy

**Decisión**: Usar spaCy para NLP en español.

**Razón**:
- Modelo preentrenado robusto para español
- Buena detección de entidades (DATE)
- Rápido y eficiente
- Comunidad activa y bien documentado

## Extensibilidad

El paquete está diseñado para ser extensible:

### Añadir Nuevos Tipos de Eventos
```python
# En models/event.py, añadir a EventType enum
NUEVO_TIPO = "nuevo_tipo"

# En classifiers/event_type_classifier.py, añadir keywords
EventType.NUEVO_TIPO: [
    'palabra1', 'palabra2', ...
]
```

### Añadir Nuevos Extractores
```python
# Crear nuevo archivo en extractors/
class LocationExtractor:
    def extract_locations(self, text: str) -> List[str]:
        # Implementación
        pass
```

### Personalizar el Pipeline
```python
class CustomPipeline(EventExtractionPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.location_extractor = LocationExtractor()
    
    def extract_events(self, news_content):
        events = super().extract_events(news_content)
        # Añadir lógica personalizada
        return events
```

## Rendimiento

### Complejidad
- **DateExtractor**: O(n) donde n es longitud del texto
- **EventTypeClassifier**: O(k*m) donde k es número de keywords, m es longitud del texto
- **Pipeline completo**: O(n + k*m) - lineal con el tamaño del texto

### Optimizaciones Posibles
1. Cachear resultados de spaCy para textos repetidos
2. Paralelizar procesamiento en batch
3. Usar modelos más pequeños de spaCy para mayor velocidad
4. Implementar clasificador ML para mejor precisión

## Testing

El paquete está preparado para testing con pytest:

```bash
pytest tests/
```

Áreas clave para testing:
- Extracción de fechas (múltiples formatos)
- Clasificación de tipos (diferentes textos)
- Pipeline completo (end-to-end)
- Manejo de edge cases (textos vacíos, sin fechas, etc.)

## Dependencias

### Principales
- **spacy**: NLP y detección de entidades
- **python-dateutil**: Parsing flexible de fechas
- **tqdm**: Barras de progreso para batch processing

### Desarrollo
- **pytest**: Framework de testing
- **black**: Formateo de código
- **flake8**: Linting
- **mypy**: Type checking

## Futuras Mejoras

1. **Extracción de ubicaciones geográficas**
2. **Identificación de entidades (personas, organizaciones)**
3. **Modelo ML para clasificación** (más preciso que keywords)
4. **Soporte multiidioma**
5. **API REST** para uso en producción
6. **Detección de sentimiento** del evento
7. **Resolución de correferencias** (múltiples menciones del mismo evento)
8. **Integración con bases de datos** para almacenamiento

## Licencia

MIT License - Ver LICENSE para detalles.
