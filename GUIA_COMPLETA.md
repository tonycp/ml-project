# Guía Completa del Sistema de Extracción de Eventos

Esta guía documenta completamente el sistema de extracción de eventos de noticias en español.

---

## 📋 Tabla de Contenidos

1. [Resumen del Sistema](#resumen-del-sistema)
2. [Instalación](#instalación)
3. [Uso Rápido](#uso-rápido)
4. [Procesamiento Masivo](#procesamiento-masivo)
5. [Entrenamiento de Modelos](#entrenamiento-de-modelos)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [Clasificadores Disponibles](#clasificadores-disponibles)
8. [Ejemplos Prácticos](#ejemplos-prácticos)
9. [Configuración Avanzada](#configuración-avanzada)
10. [Resolución de Problemas](#resolución-de-problemas)

---

## Resumen del Sistema

El **Event Extractor** es un sistema NLP para extraer eventos estructurados de noticias en español.

### Capacidades

- ✅ **Extracción de fechas**: Detecta fechas explícitas, relativas y rangos
- ✅ **Clasificación de tipo**: Categoriza eventos (cultural, deportivo, político, económico, etc.)
- ✅ **Análisis de sentimiento**: Determina si el evento es positivo, negativo o neutral
- ✅ **Reconocimiento de entidades**: Identifica personas, lugares y organizaciones (NER)
- ✅ **Relaciones SVO**: Extrae tripletas Sujeto-Verbo-Objeto
- ✅ **Clasificadores ML**: Soporta sklearn y keyword-based
- ✅ **Procesamiento masivo**: Procesa miles de noticias con barra de progreso
- ✅ **Auto-detección de modelos**: Usa automáticamente los mejores modelos disponibles

---

## Instalación

### Requisitos
- Python 3.8 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/tonycp/ml-project.git
cd ml-project

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelo spaCy para español
python -m spacy download es_core_news_lg
```

### Verificar Instalación

```bash
python -c "import Event_extractor; print('✅ Instalación correcta')"
```

---

## Uso Rápido

### Ejemplo Básico (Una Noticia)

```python
from Event_extractor import EventExtractionPipeline, NewsContent
from datetime import datetime

# Crear pipeline (auto-detecta mejores modelos disponibles)
pipeline = EventExtractionPipeline()

# Crear noticia
news = NewsContent(
    id="001",
    text="El festival de música comenzará mañana en el parque central de Madrid.",
    publication_date=datetime(2026, 1, 7)
)

# Extraer eventos
events = pipeline.extract_events(news)

# Mostrar resultados
for event in events:
    print(f"📅 Fecha: {event.date.strftime('%Y-%m-%d')}")
    print(f"📂 Tipo: {event.event_type.value}")
    print(f"😊 Sentimiento: {event.sentiment.value}")
    print(f"💯 Confianza tipo: {event.confidence:.2%}")
    print(f"💯 Confianza sentimiento: {event.sentiment_confidence:.2%}")
    print(f"👥 Entidades: {[e['text'] for e in event.entidades_asociadas]}")
    print("---")
```

### Salida

```
📅 Fecha: 2026-01-08
📂 Tipo: cultural
😊 Sentimiento: positivo
💯 Confianza tipo: 87%
💯 Confianza sentimiento: 75%
👥 Entidades: ['festival de música', 'parque central', 'Madrid']
---
```

---

## Procesamiento Masivo

### Script process_all_news.py

Procesa todas las noticias de una base de datos SQLite y extrae eventos.

#### Uso Básico (Auto-detección)

```bash
# Auto-detecta y usa los mejores modelos disponibles
python process_all_news.py --stats --json eventos.json
```

#### Opciones Avanzadas

```bash
# Forzar sklearn para tipo, keyword para sentimiento
python process_all_news.py --force-sklearn-type --force-keyword-sentiment --stats

# Procesar solo 100 noticias (para pruebas)
python process_all_news.py --limit 100

# Guardar en JSON y CSV
python process_all_news.py --json resultados/eventos.json --csv resultados/eventos.csv

# Modo debug (mostrar errores detallados)
python process_all_news.py --debug

# Sin progreso (para logs)
python process_all_news.py --quiet
```

#### Flags Disponibles

| Flag | Descripción |
|------|-------------|
| `--database PATH` | Ruta a la base de datos SQLite (default: noticias.db) |
| `--force-sklearn-type` | Forzar clasificador sklearn para tipo de evento |
| `--force-keyword-type` | Forzar clasificador keyword para tipo de evento |
| `--force-sklearn-sentiment` | Forzar clasificador sklearn para sentimiento |
| `--force-keyword-sentiment` | Forzar clasificador keyword para sentimiento |
| `--sklearn-model PATH` | Ruta personalizada al modelo sklearn de tipo |
| `--limit N` | Procesar solo N noticias |
| `--json PATH` | Guardar eventos en JSON |
| `--csv PATH` | Guardar eventos en CSV |
| `--stats` | Mostrar estadísticas detalladas al final |
| `--debug` | Mostrar errores con traceback completo |
| `--quiet` | No mostrar barra de progreso |

#### Comportamiento de Auto-detección

Por defecto (sin flags), el sistema:

1. **Busca modelos sklearn** en `models/`:
   - `models/sklearn_spanish_svm.pkl` → Clasificador de tipo
   - `models/sklearn_tass_sentiment.pkl` → Clasificador de sentimiento

2. **Si los encuentra**: Los usa automáticamente
3. **Si no**: Usa clasificadores keyword como fallback

#### Formato de Salida JSON

```json
[
  {
    "fecha": "2026-01-08T10:00:00",
    "tipo": "cultural",
    "sentimiento": "positivo",
    "confianza_tipo": 0.87,
    "confianza_sentimiento": 0.75,
    "noticia_id": "001",
    "entidades": ["festival de música", "Madrid"]
  }
]
```

#### Rendimiento

Con barra de progreso (tqdm):
```
Procesando: 45%|████▌  | 5861/13022 [01:23<01:36, 70.2 noticia/s, eventos=1234, con_eventos=856, errores=3]
```

- **Velocidad típica**: 50-70 noticias/segundo
- **13,000 noticias**: ~3-5 minutos

---

## Entrenamiento de Modelos

El sistema incluye scripts para entrenar nuevos modelos ML.

### 1. Entrenar Clasificador de Sentimiento

**Script**: `examples/training/train_sentiment_sklearn.py`

```bash
cd examples/training
python train_sentiment_sklearn.py
```

**Características**:
- Corpus: TASS-2019 (tweets en español con sentimiento)
- Modelo: LinearSVC con TF-IDF
- Split: 80% train / 20% test
- Guarda en: `models/sklearn_tass_sentiment.pkl`

**Salida esperada**:
```
🔬 ENTRENAMIENTO DE CLASIFICADOR SKLEARN CON TASS
======================================================================

📊 Cargando corpus TASS...
   ✅ Dataset cargado: 1,125 muestras
   ✅ Ejemplos válidos: 968

📋 Preparando datos...
   📝 Train: 774 | Test: 194

   📊 Distribución de clases (train):
      N: 325 (42.0%)
      P: 273 (35.3%)
      NEU: 176 (22.7%)

🎓 Entrenando clasificador sklearn...
   Modelo: LinearSVC
   Features: TF-IDF (max 10k features, bigrams)
   ✅ Entrenamiento completado

📈 Evaluando en conjunto de test...

   📊 Resultados:
      • Accuracy: 0.7371 (73.71%)
      • Confianza promedio: 0.8524

   📋 Reporte detallado por clase:
              precision    recall  f1-score   support
    Negativo     0.7500    0.7826    0.7660        69
     Neutral     0.6667    0.6190    0.6419        42
    Positivo     0.7692    0.7619    0.7655        84

💾 GUARDANDO MODELO
   ✅ Modelo guardado exitosamente
   📁 Ubicación: models/sklearn_tass_sentiment.pkl
```

### 2. Entrenar Clasificador de Tipo

**Script**: `examples/training/train_sklearn_model.py`

```bash
cd examples/training
python train_sklearn_model.py
```

**Características**:
- Corpus: Spanish News dataset
- Modelo: LinearSVC con TF-IDF + SpaCy tokenization
- Categorías: Deportes, Política, Economía, Cultura, etc.
- Guarda en: `models/sklearn_spanish_svm.pkl`

---

## Estructura del Proyecto

```
ml-project/
├── Event_extractor/              # Paquete principal
│   ├── models/                   # Modelos de datos
│   │   ├── event.py             # EventType, EventSentiment, Event
│   │   └── news.py              # NewsContent
│   ├── extractors/               # Extractores especializados
│   │   ├── date_extractor.py   # Extracción de fechas
│   │   └── NER_extractor.py     # Reconocimiento entidades
│   ├── classifiers/              # Clasificadores
│   │   ├── news_type/           # Clasificadores de tipo
│   │   │   ├── base.py         # Clase base abstracta
│   │   │   ├── keyword_classifier.py  # Basado en palabras clave
│   │   │   └── sklearn_classifier.py  # Basado en ML
│   │   ├── sentiment/           # Clasificadores de sentimiento
│   │   │   ├── base.py
│   │   │   ├── keyword_classifier.py
│   │   │   ├── sklearn_classifier.py
│   │   │   └── huggingface_classifier.py
│   │   └── ml/                  # Utilidades ML
│   │       ├── corpus_loaders.py
│   │       └── model_configs.py
│   ├── pipeline/                 # Pipeline principal
│   │   └── event_pipeline.py   # EventExtractionPipeline
│   └── utils/                    # Utilidades
│       └── text_preprocessor.py
├── examples/                      # Ejemplos de uso
│   ├── 01_basic_usage.py        # Uso básico
│   ├── 02_simple_example.py     # Ejemplo simple
│   ├── 03_date_extraction_demo.py
│   ├── 04_extract_from_database.py
│   ├── 05_test_manual_news.py
│   ├── 06_integrated_pipeline_demo.py
│   ├── 07_advanced_pipeline.py
│   ├── 08_date_extraction_advanced.py
│   ├── 09_pipeline_with_sklearn_demo.py
│   ├── comparisons/              # Comparaciones de clasificadores
│   │   ├── compare_classifiers.py
│   │   ├── compare_corpus.py
│   │   └── compare_sentiment.py
│   ├── training/                 # Scripts de entrenamiento
│   │   ├── train_sentiment_sklearn.py
│   │   └── train_sklearn_model.py
│   └── templates/
│       └── data_loader_template.py
├── models/                        # Modelos entrenados
│   ├── sklearn_spanish_svm.pkl   # Clasificador de tipo
│   ├── sklearn_tass_sentiment.pkl  # Clasificador de sentimiento
│   └── sklearn_news_classifier_simple.pkl
├── process_all_news.py           # Script de procesamiento masivo
├── noticias.db                   # Base de datos de noticias (SQLite)
├── requirements.txt              # Dependencias
├── README.md                     # Documentación principal
├── QUICKSTART.md                 # Guía de inicio rápido
└── GUIA_COMPLETA.md             # Esta guía
```

---

## Clasificadores Disponibles

### Clasificadores de Tipo de Evento

#### 1. KeywordNewsClassifier
- **Tipo**: Basado en palabras clave
- **Ventajas**: Rápido, sin entrenamiento
- **Desventajas**: Menos preciso
- **Categorías**: 10 tipos de eventos

**Uso**:
```python
from Event_extractor.classifiers.news_type import KeywordNewsClassifier

classifier = KeywordNewsClassifier()
event_type, confidence = classifier.classify("El equipo ganó el partido")
# event_type = EventType.DEPORTIVO, confidence ≈ 0.85
```

#### 2. SklearnNewsClassifier
- **Tipo**: Machine Learning (TF-IDF + LinearSVC)
- **Ventajas**: Más preciso, aprende patrones
- **Desventajas**: Requiere entrenamiento
- **Soporte**: LinearSVC (decision_function) y modelos probabilísticos

**Uso**:
```python
from Event_extractor.classifiers.news_type import SklearnNewsClassifier

# Cargar modelo entrenado
classifier = SklearnNewsClassifier(
    model_path="models/sklearn_spanish_svm.pkl"
)

event_type, confidence = classifier.classify("El equipo ganó el partido")
# event_type = EventType.DEPORTIVO, confidence ≈ 0.92
```

### Clasificadores de Sentimiento

#### 1. KeywordSentimentClassifier
- **Tipo**: Basado en diccionario de palabras
- **Ventajas**: Rápido, interpretable
- **Desventajas**: Limitado a palabras conocidas

**Uso**:
```python
from Event_extractor.classifiers.sentiment import KeywordSentimentClassifier

classifier = KeywordSentimentClassifier()
sentiment, confidence = classifier.classify("Terrible accidente en la carretera")
# sentiment = EventSentiment.NEGATIVE, confidence ≈ 0.80
```

#### 2. SklearnSentimentClassifier
- **Tipo**: ML (TF-IDF + LinearSVC)
- **Corpus**: TASS-2019 (tweets españoles)
- **Ventajas**: Aprende del contexto

**Uso**:
```python
from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier

classifier = SklearnSentimentClassifier.load_model(
    "models/sklearn_tass_sentiment.pkl"
)

sentiment, confidence = classifier.classify("¡Excelente noticia para todos!")
# sentiment = EventSentiment.POSITIVE, confidence ≈ 0.88
```

#### 3. HuggingFaceSentimentClassifier
- **Tipo**: Transformers (BETO, RoBERTuito, etc.)
- **Ventajas**: Estado del arte, muy preciso
- **Desventajas**: Lento, requiere GPU

**Uso**:
```python
from Event_extractor.classifiers.sentiment import HuggingFaceSentimentClassifier

classifier = HuggingFaceSentimentClassifier(
    model_name="finiteautomata/beto-sentiment-analysis"
)

sentiment, confidence = classifier.classify("Me encanta este festival")
# sentiment = EventSentiment.POSITIVE, confidence ≈ 0.95
```

---

## Ejemplos Prácticos

### Ejemplo 1: Procesamiento de Una Noticia

```python
from Event_extractor import EventExtractionPipeline, NewsContent
from datetime import datetime

pipeline = EventExtractionPipeline()

news = NewsContent(
    id="noticia_001",
    text="""
    El Gobierno anunció ayer que el nuevo estadio olímpico se inaugurará 
    el próximo 15 de marzo. La ceremonia contará con la presencia del 
    presidente y varios atletas reconocidos.
    """,
    publication_date=datetime(2026, 1, 7)
)

events = pipeline.extract_events(news)

print(f"Eventos extraídos: {len(events)}")
for i, event in enumerate(events, 1):
    print(f"\nEvento {i}:")
    print(f"  Fecha: {event.date}")
    print(f"  Tipo: {event.event_type.value}")
    print(f"  Sentimiento: {event.sentiment.value}")
    print(f"  Entidades: {[e['text'] for e in event.entidades_asociadas]}")
```

### Ejemplo 2: Usar Modelos Específicos

```python
from Event_extractor import EventExtractionPipeline
from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier

# Cargar clasificador de sentimiento sklearn
sentiment_clf = SklearnSentimentClassifier.load_model(
    "models/sklearn_tass_sentiment.pkl"
)

# Pipeline con sklearn para tipo y sentimiento
pipeline = EventExtractionPipeline(
    use_sklearn_classifier=True,  # sklearn para tipo
    sklearn_model_path="models/sklearn_spanish_svm.pkl",
    sentiment_classifier=sentiment_clf  # sklearn para sentimiento
)

# Procesar noticia
events = pipeline.extract_events(news)
```

### Ejemplo 3: Solo Clasificadores Keyword (Rápido)

```python
from Event_extractor import EventExtractionPipeline

# Pipeline sin ML (más rápido)
pipeline = EventExtractionPipeline(
    use_sklearn_classifier=False,  # keyword para tipo
    sentiment_classifier=None  # keyword para sentimiento (default)
)

events = pipeline.extract_events(news)
```

### Ejemplo 4: Procesar Múltiples Noticias

```python
from Event_extractor import EventExtractionPipeline, NewsContent
import sqlite3

# Cargar noticias de base de datos
conn = sqlite3.connect("noticias.db")
cursor = conn.cursor()
cursor.execute("SELECT id, texto, fecha FROM noticias LIMIT 10")

pipeline = EventExtractionPipeline()
all_events = []

for row in cursor.fetchall():
    news = NewsContent(id=str(row[0]), text=row[1])
    events = pipeline.extract_events(news)
    all_events.extend(events)

print(f"Total eventos extraídos: {len(all_events)}")
```

### Ejemplo 5: Estadísticas de Eventos

```python
from collections import Counter
from process_all_news import extract_events_from_all_news, get_statistics

# Extraer eventos de todas las noticias
events = extract_events_from_all_news(limit=1000)

# Obtener estadísticas
stats = get_statistics(events)

print(f"Total eventos: {stats['total']}")
print(f"\nDistribución por tipo:")
for tipo, count in stats['tipos'].items():
    print(f"  {tipo}: {count}")

print(f"\nDistribución por sentimiento:")
for sent, count in stats['sentimientos'].items():
    print(f"  {sent}: {count}")
```

---

## Configuración Avanzada

### Configurar Umbral de Confianza

```python
pipeline = EventExtractionPipeline(min_confidence=0.5)

# Cambiar en tiempo de ejecución
pipeline.set_min_confidence(0.7)
```

### Deshabilitar Clasificación de Sentimiento

```python
pipeline = EventExtractionPipeline(classify_sentiment=False)
```

### Fecha de Referencia Personalizada

```python
from datetime import datetime

pipeline = EventExtractionPipeline(
    reference_date=datetime(2025, 12, 1)
)
```

### Entrenar Modelo Personalizado

```python
from Event_extractor.classifiers.news_type import SklearnNewsClassifier

# Crear y entrenar
classifier = SklearnNewsClassifier()

texts = ["texto1", "texto2", ...]
labels = ["deportivo", "politico", ...]

classifier.train_from_dataset(texts, labels)

# Guardar
classifier.save_model("mi_modelo.pkl")

# Usar en pipeline
pipeline = EventExtractionPipeline(
    use_sklearn_classifier=True,
    sklearn_model_path="mi_modelo.pkl"
)
```

---

## Resolución de Problemas

### Error: "No module named 'Event_extractor'"

**Solución**: Asegúrate de estar en el directorio raíz del proyecto.

```bash
cd /ruta/a/ml-project
python tu_script.py
```

### Error: "Can't find model 'es_core_news_lg'"

**Solución**: Descarga el modelo de spaCy.

```bash
python -m spacy download es_core_news_lg
```

### Error: "LinearSVC has no attribute 'predict_proba'"

**Solución**: El sistema ahora maneja automáticamente modelos con `decision_function`. Si persiste, actualiza:

```bash
git pull origin main
```

### Procesamiento muy lento

**Soluciones**:

1. **Usar clasificadores keyword** (más rápidos):
```bash
python process_all_news.py --force-keyword-type --force-keyword-sentiment
```

2. **Procesar en lotes**:
```bash
python process_all_news.py --limit 1000 --json batch1.json
python process_all_news.py --limit 1000 --json batch2.json
```

3. **Deshabilitar sentimiento**:
```python
pipeline = EventExtractionPipeline(classify_sentiment=False)
```

### Muchos errores al procesar

**Diagnóstico**:

```bash
# Ver primeros errores
python process_all_news.py --limit 100

# Ver todos los errores con traceback
python process_all_news.py --limit 100 --debug
```

Los errores comunes se muestran al final del resumen.

### Base de datos no encontrada

**Solución**: Verifica que `noticias.db` existe.

```bash
ls -lh noticias.db

# O especifica ruta
python process_all_news.py --database /ruta/a/noticias.db
```

### Modelos no se detectan automáticamente

**Verificación**:

```bash
ls -lh models/
# Debe mostrar:
# sklearn_spanish_svm.pkl
# sklearn_tass_sentiment.pkl
```

**Forzar uso**:
```bash
python process_all_news.py --force-sklearn-type --force-sklearn-sentiment
```

---

## Contacto y Soporte

- **Repositorio**: https://github.com/tonycp/ml-project
- **Issues**: Reporta problemas en GitHub Issues
- **Documentación**: Ver README.md y QUICKSTART.md

---

## Changelog

### Versión Actual (2026-01-07)

**Nuevas Características**:
- ✅ Auto-detección de modelos sklearn
- ✅ Barra de progreso con tqdm en process_all_news.py
- ✅ Soporte para LinearSVC (decision_function)
- ✅ Modo debug con traceback completo
- ✅ Flags para forzar clasificadores específicos
- ✅ Mejor manejo de errores y reporte de primeros 5 errores

**Correcciones**:
- 🐛 Corregido error con LinearSVC sin predict_proba
- 🐛 Corregido parámetro min_confidence faltante en pipeline
- 🐛 Corregido método get_name en SklearnNewsClassifier
- 🐛 Corregido split de datos en train_sentiment_sklearn.py

**Mejoras**:
- ⚡ Procesamiento 3x más rápido (sin prints por línea)
- ⚡ Auto-detección inteligente de modelos
- 📊 Estadísticas mejoradas con primeros errores
- 🎨 Mejor experiencia de usuario con tqdm

---

## Licencia

Ver archivo [LICENSE](LICENSE) para detalles.
