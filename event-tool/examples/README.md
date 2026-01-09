# Event Extractor - Ejemplos de Uso

Este directorio contiene ejemplos organizados de uso del sistema de extracción de eventos, ordenados desde lo más básico hasta lo más avanzado.

## 📁 Estructura Reorganizada

```
examples/
├── 01_basic_usage.py                    # Uso básico del sistema
├── 02_component_usage.py                # Uso de componentes individuales
├── 03_reference_date_demo.py            # Demo de fechas de referencia
├── 04_extract_from_database.py          # Extracción desde base de datos
├── 05_test_manual_news.py               # Testing interactivo manual
├── 06_integrated_pipeline_demo.py       # Demo completo del pipeline
├── 07_sentiment_classification_demo.py  # Demo de clasificación de sentimiento
├── 08_pipeline_with_sentiment.py        # Pipeline con clasificadores pluggables
├── 09_pipeline_with_sklearn_demo.py     # Pipeline con sklearn
├── comparisons/                          # Comparaciones entre modelos
│   ├── compare_sentiment_classifiers.py
│   ├── compare_sklearn_models.py
│   ├── model_corpus_comparison.py
│   └── tass_sklearn_demo.py
├── training/                             # Entrenamiento de modelos
│   ├── train_sentiment_sklearn.py
│   └── train_sklearn_model.py
└── templates/                            # Templates reutilizables
    └── data_loader_template.py
```

## 📋 Guía de Ejemplos

### 🚀 Nivel 1: Conceptos Básicos (01-03)

#### 1. Uso Básico del Sistema
**Archivo:** `01_basic_usage.py`

Ejemplo básico de extracción de eventos desde un texto de noticia.
```bash
python examples/01_basic_usage.py
```

**Qué aprenderás:**
- Crear un pipeline de extracción de eventos
- Procesar noticias simples
- Ver los eventos extraídos

#### 2. Uso de Componentes Individuales
**Archivo:** `02_component_usage.py`

Demuestra el uso individual de cada componente del sistema.
```bash
python examples/02_component_usage.py
```

**Componentes demostrados:**
- DateExtractor: Extracción de fechas
- EventTypeClassifier: Clasificación de tipos de evento
- SentimentClassifier: Clasificación de sentimiento
- NER: Extracción de entidades nombradas

#### 3. Fechas de Referencia
**Archivo:** `03_reference_date_demo.py`

Demuestra el uso de fechas de referencia para normalizar fechas relativas.
```bash
python examples/03_reference_date_demo.py
```

**Qué aprenderás:**
- Usar fechas de referencia (ej: "hoy", "ayer")
- Normalizar fechas relativas
- Manejar diferentes formatos de fecha

---

### 🗄️ Nivel 2: Integración con Datos Reales (04-05)

#### 4. Extracción desde Base de Datos
**Archivo:** `04_extract_from_database.py`

Carga noticias desde una base de datos SQLite y extrae eventos.
```bash
python examples/04_extract_from_database.py
```

**Características:**
- Conexión a SQLite (noticias.db)
- Procesamiento batch de noticias
- Estadísticas de eventos extraídos
- Análisis temporal y de entidades

**Requisitos:**
- Base de datos `noticias.db` en el directorio raíz

#### 5. Testing Interactivo Manual
**Archivo:** `05_test_manual_news.py`

Herramienta interactiva para validar el pipeline con noticias propias.
```bash
python examples/05_test_manual_news.py
```

**Características:**
- Entrada de texto manual (terminar con doble Enter)
- Fecha opcional (DD/MM/YYYY)
- Visualización paso a paso del procesamiento:
  1. Tokenización
  2. Análisis spaCy + NER
  3. Extracción de fechas
  4. Clasificación de tipo
  5. Análisis de sentimiento
  6. Extracción de triples SVO
- Detalles completos del evento extraído

---

### 🚀 Nivel 3: Pipeline Completo (06-09)

#### 6. Demo Completo del Pipeline
**Archivo:** `06_integrated_pipeline_demo.py`

Demostración completa del pipeline con noticias sintéticas.
```bash
python examples/06_integrated_pipeline_demo.py
```

**Incluye:**
- Noticias de ejemplo variadas
- Pipeline completo: fechas + tipo + sentimiento + entidades
- Estadísticas completas
- Visualización temporal

#### 7. Demo de Clasificación de Sentimiento
**Archivo:** `07_sentiment_classification_demo.py`

Clasificación de sentimiento en noticias de ejemplo.
```bash
python examples/07_sentiment_classification_demo.py
```

**Clasificadores disponibles:**
- Keyword-based (reglas)
- HuggingFace (modelos transformers)
- Sklearn (TF-IDF + ML)

#### 8. Pipeline con Clasificadores Pluggables
**Archivo:** `08_pipeline_with_sentiment.py`

Pipeline con diferentes clasificadores de sentimiento intercambiables.
```bash
python examples/08_pipeline_with_sentiment.py
```

#### 9. Pipeline con Sklearn
**Archivo:** `09_pipeline_with_sklearn_demo.py`

Pipeline usando clasificación basada en sklearn.
```bash
python examples/09_pipeline_with_sklearn_demo.py
```

---

### 📊 Comparaciones entre Modelos

#### Comparar Clasificadores de Sentimiento
**Archivo:** `comparisons/compare_sentiment_classifiers.py`

Compara los 3 tipos de clasificadores: Keyword, HuggingFace, Sklearn.
```bash
python examples/comparisons/compare_sentiment_classifiers.py
```

**Métricas evaluadas:**
- Accuracy
- Precision/Recall/F1-score por clase
- Velocidad de inferencia
- Ranking por rendimiento

#### Comparar Algoritmos de Sklearn
**Archivo:** `comparisons/compare_sklearn_models.py`

Compara diferentes algoritmos de sklearn: SVM, Naive Bayes, Logistic Regression, Random Forest.
```bash
python examples/comparisons/compare_sklearn_models.py
```

**Incluye:**
- Entrenamiento en TASS-2019
- Evaluación en conjunto de test
- Classification report por modelo
- Comparación de velocidad

#### Comparar Modelos y Corpus
**Archivo:** `comparisons/model_corpus_comparison.py`

Compara diferentes combinaciones de modelos y corpus.
```bash
python examples/comparisons/model_corpus_comparison.py
```

#### Demo TASS con Sklearn
**Archivo:** `comparisons/tass_sklearn_demo.py`

Demo específico del corpus TASS-2019 con sklearn.
```bash
python examples/comparisons/tass_sklearn_demo.py
```

**Características:**
- Usa corpus TASS-2019 (español)
- 1,125 tweets de entrenamiento
- 1,706 tweets de test
- Labels: P (positivo), N (negativo), NEU (neutral)

---

### 🎓 Entrenamiento de Modelos

#### Entrenar Clasificador de Sentimiento
**Archivo:** `training/train_sentiment_sklearn.py`

Entrena un clasificador de sentimiento sklearn en el corpus TASS.
```bash
python examples/training/train_sentiment_sklearn.py
```

**Características:**
- Descarga automática de TASS-2019
- TF-IDF vectorization
- Multiple algoritmos sklearn
- Guarda modelo entrenado
- Evaluación en test set

#### Entrenar Modelo Sklearn Genérico
**Archivo:** `training/train_sklearn_model.py`

Script genérico para entrenar modelos sklearn.
```bash
python examples/training/train_sklearn_model.py
```

---

### 📝 Templates

#### Template de Cargador de Datos
**Archivo:** `templates/data_loader_template.py`

Template reutilizable para cargar datos de diferentes fuentes.

**Soporta:**
- SQLite
- CSV/JSON
- APIs
- Archivos de texto

---

## 🎯 Recomendaciones de Uso

### Para Empezar
1. Comienza con `01_basic_usage.py` para entender el flujo básico
2. Explora componentes individuales con `02_component_usage.py`
3. Prueba con tus propias noticias usando `05_test_manual_news.py`

### Para Integrar en tu Proyecto
1. Usa `04_extract_from_database.py` como referencia para cargar datos
2. Adapta el pipeline de `06_integrated_pipeline_demo.py` a tus necesidades
3. Revisa `templates/data_loader_template.py` para diferentes fuentes de datos

### Para Mejorar el Modelo
1. Compara diferentes modelos con los scripts de `comparisons/`
2. Entrena tu propio modelo con scripts de `training/`
3. Evalúa resultados con métricas de sklearn

---

## 📦 Dependencias

```bash
# Instalar todas las dependencias (desde el directorio raíz del proyecto)
pip install -r requirements.txt

# Descargar modelo de spaCy para español
python -m spacy download es_core_news_lg
```

---

## 🐛 Troubleshooting

### Error: No module named 'Event_extractor'
Asegúrate de ejecutar los ejemplos desde el directorio raíz del proyecto:
```bash
cd /ruta/a/ml-project
python examples/01_basic_usage.py
```

### Error con spaCy
Si falta el modelo de spaCy:
```bash
python -m spacy download es_core_news_lg
```

### Error con TASS corpus
El corpus TASS-2019 se descarga automáticamente con `datasets` de HuggingFace. Si hay problemas de conexión, los scripts de comparación usan datos sintéticos como fallback.

---

## 📚 Documentación Adicional

- **QUICKSTART.md**: Guía rápida de inicio
- **ARCHITECTURE.md**: Arquitectura del sistema
- **README.md**: Documentación principal del proyecto
