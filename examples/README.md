# Event Extractor - Ejemplos

Este directorio contiene ejemplos de uso de la librería Event Extractor.

## Ejemplos Disponibles

### 1. basic_usage.py

Ejemplo básico que muestra cómo usar el pipeline completo para extraer eventos de noticias.

```bash
python examples/basic_usage.py
```

Características:
- Extracción de eventos de múltiples noticias
- Diferentes tipos de eventos (cultural, deportivo, meteorológico)
- Procesamiento en batch
- Filtrado y ordenamiento de eventos

### 2. component_usage.py

Ejemplo de uso de componentes individuales del paquete.

```bash
python examples/component_usage.py
```

Características:
- Uso del DateExtractor
- Uso del EventTypeClassifier
- Añadir palabras clave personalizadas
- Clasificación múltiple

### 3. reference_date_demo.py

**NUEVO**: Demostración del manejo de fechas con y sin `reference_date`.

```bash
python examples/reference_date_demo.py
```

Características:
- Muestra qué fechas se extraen sin `reference_date` (solo explícitas)
- Muestra qué fechas se extraen con `reference_date` (todas)
- Demuestra cómo el pipeline usa la fecha de metadata automáticamente
- **Importante para entender cómo evitar fechas erróneas**

### 4. sentiment_classification_demo.py

**NUEVO**: Demostración de la clasificación de sentimiento de eventos.

```bash
python examples/sentiment_classification_demo.py
```

Características:
- Clasifica eventos como positivos, negativos o neutrales
- Muestra ejemplos de cada categoría de sentimiento
- Integración con el pipeline completo
- Personalización de palabras clave para sentimientos

### 5. data_loader_template.py

Plantillas para cargar datos desde diferentes fuentes.

```bash
python examples/data_loader_template.py
```

Características:
- Plantillas para JSON, CSV, API, Base de datos, RSS
- Ejemplo completo de pipeline desde carga hasta extracción
- Adaptable a tu formato de datos específico

## Requisitos

Antes de ejecutar los ejemplos, asegúrate de:

1. Instalar el paquete:
```bash
pip install -e .
```

2. Instalar el modelo de spaCy:
```bash
python -m spacy download es_core_news_sm
```

## Crear tus propios ejemplos

Puedes crear tus propios scripts de ejemplo siguiendo esta estructura básica:

```python
from Event_extractor import EventExtractionPipeline, NewsContent, NewsMetadata
from datetime import datetime

# Inicializar pipeline
pipeline = EventExtractionPipeline()

# Crear contenido de noticia
metadata = NewsMetadata(
    title="Tu título",
    date=datetime.now(),
    source="Tu fuente"
)

news = NewsContent(
    text="Tu texto aquí...",
    metadata=metadata
)

# Extraer eventos
events = pipeline.extract_events(news)

# Procesar resultados
for event in events:
    print(event)
```
