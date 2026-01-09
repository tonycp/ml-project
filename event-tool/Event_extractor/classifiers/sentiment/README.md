# Sentiment Classifiers

Este módulo proporciona una arquitectura modular para clasificación de sentimiento de eventos, siguiendo el mismo patrón de diseño que los clasificadores de tipos de noticias.

## 📐 Arquitectura

### Clase Abstracta Base

```python
class SentimentClassifier(ABC):
    """Interfaz común para todos los clasificadores de sentimiento"""
    
    @abstractmethod
    def classify(text: str) -> Tuple[EventSentiment, float]
        """Clasifica texto y retorna (sentimiento, confianza)"""
    
    @abstractmethod
    def get_name() -> str
        """Retorna el nombre del clasificador"""
```

### Implementaciones Disponibles

#### 1. KeywordSentimentClassifier

Clasificador basado en palabras clave y reglas heurísticas.

**Características:**
- ✅ Sin dependencias externas
- ✅ Rápido y ligero
- ✅ No requiere entrenamiento
- ❌ Menor precisión que modelos ML

**Uso:**
```python
from Event_extractor.classifiers.sentiment import KeywordSentimentClassifier

clf = KeywordSentimentClassifier()
sentiment, confidence = clf.classify("El festival fue un éxito increíble")
# → (EventSentiment.POSITIVE, 1.0)
```

#### 2. HuggingFaceSentimentClassifier

Clasificador usando modelos transformers pre-entrenados de HuggingFace.

**Modelos Soportados:**
- `MarIASentimentClassifier`: MarIA/RoBERTa español (UMUTeam/roberta-spanish-sentiment-analysis)
- `BETOSentimentClassifier`: BETO español (finiteautomata/beto-sentiment-analysis)
- `MultilingualSentimentClassifier`: XLM-RoBERTa multilingüe

**Características:**
- ✅ Alta precisión
- ✅ Modelos específicos para español
- ✅ Pre-entrenados (no requiere entrenamiento)
- ❌ Requiere `transformers` library
- ❌ Más lento que keyword o sklearn

**Instalación:**
```bash
pip install transformers torch
```

**Uso:**
```python
from Event_extractor.classifiers.sentiment import MarIASentimentClassifier

# MarIA/RoBERTa español
clf = MarIASentimentClassifier()
sentiment, confidence = clf.classify("Cancelan el concierto por mal tiempo")
# → (EventSentiment.NEGATIVE, 0.95)

# O usar directamente con cualquier modelo de HuggingFace
from Event_extractor.classifiers.sentiment import HuggingFaceSentimentClassifier
clf = HuggingFaceSentimentClassifier(
    model_name="finiteautomata/beto-sentiment-analysis"
)
```

#### 3. SklearnSentimentClassifier

Clasificador usando TF-IDF + modelos sklearn (SVM, Naive Bayes, etc.).

**Características:**
- ✅ Balance entre precisión y velocidad
- ✅ Soporta entrenamiento con corpus personalizados
- ✅ Exportable y reutilizable
- ✅ Análisis de features importantes
- ❌ Requiere entrenamiento previo

**Entrenamiento con TASS:**
```python
from datasets import load_dataset
from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier

# Cargar corpus TASS
ds_train = load_dataset("mrm8488/tass-2019", split="train")
train_texts = ds_train['sentence']
train_labels = ds_train['sentiments']  # 'N', 'NEU', 'P'

# Entrenar
clf = SklearnSentimentClassifier(model_name="SVM Linear TASS")
clf.train(texts=train_texts, labels=train_labels)

# Guardar modelo
clf.save_model("models/sklearn_tass_sentiment.pkl")
```

**Uso de modelo entrenado:**
```python
from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier

# Cargar modelo
clf = SklearnSentimentClassifier.load_model("models/sklearn_tass_sentiment.pkl")

# Clasificar
sentiment, confidence = clf.classify("Terrible accidente en la autopista")
# → (EventSentiment.NEGATIVE, 0.87)

# Ver features importantes
importance = clf.get_feature_importance(top_n=10)
for sentiment, features in importance.items():
    print(f"{sentiment.value}:")
    for word, score in features:
        print(f"  - {word}: {score:.4f}")
```

## 🔄 Integración con Pipeline

El pipeline principal acepta cualquier implementación de `SentimentClassifier`:

```python
from Event_extractor.pipeline.event_pipeline import EventExtractionPipeline
from Event_extractor.classifiers.sentiment import (
    KeywordSentimentClassifier,
    MarIASentimentClassifier,
    SklearnSentimentClassifier
)

# Opción 1: Keyword (por defecto)
pipeline = EventExtractionPipeline(classify_sentiment=True)

# Opción 2: MarIA/RoBERTa
pipeline = EventExtractionPipeline(
    sentiment_classifier=MarIASentimentClassifier()
)

# Opción 3: Sklearn
clf = SklearnSentimentClassifier.load_model("models/sklearn_tass_sentiment.pkl")
pipeline = EventExtractionPipeline(sentiment_classifier=clf)

# Extraer eventos
eventos = pipeline.extract_events(news_content)
```

## 📊 Corpus Disponibles

### TASS (Twitter Analytics in Spanish)

Corpus de tweets en español con anotaciones de sentimiento.

**Dataset:** `mrm8488/tass-2019`
- **Train:** 1,125 tweets
- **Test:** 1,706 tweets
- **Labels:** `N` (negativo), `NEU` (neutral), `P` (positivo)

**Carga:**
```python
from datasets import load_dataset

ds_train = load_dataset("mrm8488/tass-2019", split="train")
ds_test = load_dataset("mrm8488/tass-2019", split="test")
```

## 📚 Ejemplos

Ver `examples/ml_classification/` para ejemplos completos:

- **`train_sentiment_sklearn.py`**: Entrenar clasificador sklearn con TASS
- **`compare_sentiment_classifiers.py`**: Comparar diferentes clasificadores
- **`pipeline_with_sentiment.py`**: Usar en pipeline principal

## 🔄 Retrocompatibilidad

El alias `EventSentimentClassifier` apunta a `KeywordSentimentClassifier` para mantener compatibilidad con código existente:

```python
# Código antiguo sigue funcionando
from Event_extractor.classifiers import EventSentimentClassifier
clf = EventSentimentClassifier()
```

## 💡 Recomendaciones

| Clasificador | Velocidad | Precisión | Dependencias | Uso Recomendado |
|--------------|-----------|-----------|--------------|-----------------|
| Keyword | ⚡⚡⚡ Muy rápido | ⭐⭐ Básica | Ninguna | Prototipado rápido, baseline |
| Sklearn | ⚡⚡ Rápido | ⭐⭐⭐ Buena | sklearn | Producción, balance |
| Transformers | ⚡ Lento | ⭐⭐⭐⭐ Excelente | transformers + torch | Máxima precisión |

## 🔧 Extensibilidad

Para agregar un nuevo clasificador, simplemente hereda de `SentimentClassifier`:

```python
from Event_extractor.classifiers.sentiment import SentimentClassifier
from Event_extractor.models.event import EventSentiment
from typing import Tuple

class MiClasificador(SentimentClassifier):
    def classify(self, text: str) -> Tuple[EventSentiment, float]:
        # Tu implementación
        return EventSentiment.POSITIVE, 0.85
    
    def get_name(self) -> str:
        return "Mi Clasificador Custom"
```

Y úsalo en el pipeline:

```python
pipeline = EventExtractionPipeline(
    sentiment_classifier=MiClasificador()
)
```
