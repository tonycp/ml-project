# Event Extractor

Sistema de extracción de eventos (con fecha y tipo) de contenido de noticias en español.

## Características

- 🗓️ **Extracción de fechas**: Detecta fechas explícitas, relativas y rangos de fechas
- 🏷️ **Clasificación de eventos**: Identifica tipos de eventos (cultural, deportivo, meteorológico, etc.)
- 😢 **Clasificación de sentimiento**: Determina si los eventos son positivos, negativos o neutrales
- 📊 **Pipeline completo**: Procesa noticias de extremo a extremo
- 🔄 **Manejo de rangos**: Trata fechas de inicio y fin como eventos separados
- 🇪🇸 **Optimizado para español**: Procesamiento de lenguaje natural en español

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tonycp/ml-project.git
cd ml-project
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar modelo de spaCy

```bash
python -m spacy download es_core_news_lg
```

## Uso Básico

```python
from Event_extractor import EventExtractionPipeline, NewsContent
from datetime import datetime

# Crear pipeline
pipeline = EventExtractionPipeline()

# Crear contenido de noticia
news = NewsContent(
    text="El festival de música se realizará del 10 al 15 de enero de 2025 "
         "en el parque central. Habrá conciertos de diferentes géneros.",
    id="noticia_001",
    date=datetime(2024, 12, 1)
)

# Extraer eventos
events = pipeline.extract_events(news)

# Mostrar resultados
for event in events:
    print(f"Fecha: {event.date}")
    print(f"Tipo: {event.event_type}")
    print(f"Sentimiento: {event.sentiment}")
    print(f"Confianza: {event.confidence}")
    print("---")
```
```python
from Event_extractor import EventExtractionPipeline, NewsContent
from datetime import datetime

# Crear pipeline
pipeline = EventExtractionPipeline()

# Crear contenido de noticia

news = NewsContent(
    text="El festival de música se realizará del 10 al 15 de enero de 2025 "
         "en el parque central. Habrá conciertos de diferentes géneros.",
    metadata=metadata,
    title="Festival de Música en la Ciudad"
)

# Extraer eventos
events = pipeline.extract_events(news)

# Mostrar resultados
for event in events:
    print(f"Fecha: {event.date}")
    print(f"Tipo: {event.event_type}")
    print(f"Sentimiento: {event.sentiment}")  # ✅ Positivo, ❌ Negativo, ⚪ Neutral
    print(f"Título: {event.title}")
    print(f"Confianza: {event.confidence}")
    print("---")
```

## Tipos de Eventos Soportados

La librería clasifica eventos en las siguientes categorías:

- **CULTURAL**: Festivales, conciertos, exposiciones, teatro, cine
- **DEPORTIVO**: Partidos, campeonatos, torneos, competiciones
- **METEOROLOGICO**: Tormentas, huracanes, alertas climáticas
- **POLITICO**: Elecciones, leyes, decretos, referéndums
- **ECONOMICO**: Mercados, empresas, bolsa, comercio
- **SOCIAL**: Manifestaciones, protestas, movimientos sociales
- **INCIDENTE**: Accidentes, emergencias, desastres
- **REGULACION**: Normativas, reglamentos, restricciones

## Clasificación de Sentimiento

Además del tipo, cada evento se clasifica según su sentimiento:

- **✅ POSITIVO**: Celebraciones, logros, inauguraciones, victorias, festivales
- **❌ NEGATIVO**: Cancelaciones, protestas, accidentes, desastres, cierres
- **⚪ NEUTRAL**: Reuniones, anuncios, conferencias, trámites administrativos

### Ejemplos de Clasificación

```python
"Gran festival de música" → POSITIVO
"Cancelación de vuelos" → NEGATIVO  
"Reunión del comité" → NEUTRAL
"Victoria en campeonato" → POSITIVO
"Grave accidente" → NEGATIVO
```
- **OTRO**: Eventos que no encajan en las categorías anteriores

## Extracción de Fechas

El extractor de fechas reconoce múltiples formatos:

- **Fechas explícitas completas**: "25 de diciembre de 2024"
- **Fechas numéricas**: "25/12/2024", "25-12-2024"
- **Fechas sin año**: "15 de enero" *(requiere reference_date)*
- **Rangos**: "del 1 al 5 de enero" *(año explícito o requiere reference_date)*
- **Fechas relativas**: "hoy", "mañana", "la próxima semana" *(requiere reference_date)*

### ⚠️ Importante: Manejo de reference_date

Para **evitar fechas erróneas**, el extractor tiene un comportamiento seguro:

- **Sin `reference_date`**: Solo extrae fechas **explícitas y completas** (con año)
- **Con `reference_date`**: Extrae todos los tipos de fechas, usando la referencia para resolver ambigüedades


### Importante: Manejo de Rangos

Cuando se detecta un rango de fechas (ej: "del 10 al 15 de enero"), el sistema crea **eventos separados** para la fecha de inicio y la fecha de fin. Esto permite modelar correctamente el comienzo y el final de eventos prolongados.

## API Avanzada

### Procesar múltiples noticias

```python
news_list = [news1, news2, news3]
all_events = pipeline.extract_events_batch(news_list)
```

### Filtrar y ordenar eventos

```python
from Event_extractor import EventAggregator, EventType

# Eliminar duplicados
unique_events = EventAggregator.remove_duplicates(all_events)

# Filtrar por tipo
cultural_events = EventAggregator.filter_by_type(
    unique_events, 
    [EventType.CULTURAL]
)

# Filtrar por rango de fechas
from datetime import datetime

filtered = EventAggregator.filter_by_date_range(
    unique_events,
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31)
)

# Ordenar por fecha
sorted_events = EventAggregator.sort_by_date(filtered)
```

### Personalizar clasificación

```python
from Event_extractor import EventType

# Añadir palabras clave personalizadas
pipeline.add_custom_keywords(
    EventType.CULTURAL,
    ["bienal", "muestra", "vernissage"]
)

# Ajustar confianza mínima
pipeline.set_min_confidence(0.5)
```

## Uso de Componentes Individuales

### DateExtractor


### EventSentimentClassifier

```python
from Event_extractor import EventSentimentClassifier

classifier = EventSentimentClassifier()
sentiment, confidence = classifier.classify(
    "Cancelación del festival por mal tiempo"
)
# sentiment = EventSentiment.NEGATIVE
```
```python
from Event_extractor import DateExtractor
from datetime import datetime

extractor = DateExtractor(reference_date=datetime(2025, 1, 1))
dates = extractor.extract_dates("El evento será el 15 de enero")
```

### EventTypeClassifier

```python
from Event_extractor import EventTypeClassifier

classifier = EventTypeClassifier()
event_type, confidence = classifier.classify(
    "Gran concierto de rock este sábado"
)
```

## Estructura del Proyecto

```
Event_extractor/
├── __init__.py          # Punto de entrada principal
├── models/              # Modelos de datos
│   └── event.py        # Event, EventType
├── extractors/         # Extractores de información
│   └── date_extractor.py
├── classifiers/        # Clasificadores
│   └── event_type_classifier.py
├── pipeline/           # Pipeline principal
│   └── event_pipeline.py
└── utils/              # Utilidades
    └── text_preprocessor.py
```

## Roadmap

- [ ] Soporte para más formatos de entrada de noticias
- [ ] Extracción de ubicaciones geográficas
- [ ] Identificación de actores/entidades involucradas
- [ ] Modelo de machine learning para clasificación
- [ ] API REST para procesamiento en línea
- [ ] Soporte para más idiomas

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Proyecto: [https://github.com/tonycp/ml-project](https://github.com/tonycp/ml-project)

## Agradecimientos

- spaCy por el procesamiento de lenguaje natural
- dateutil por el parsing de fechas
