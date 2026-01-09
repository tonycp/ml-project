"""
Clasificadores de sentimiento para eventos.

Este módulo proporciona diferentes implementaciones de clasificadores de sentimiento
que heredan de la clase abstracta SentimentClassifier:

- KeywordSentimentClassifier: Basado en palabras clave y reglas
- HuggingFaceSentimentClassifier: Usando modelos transformers pre-entrenados
  - MarIASentimentClassifier: Alias para MarIA/RoBERTa español
  - BETOSentimentClassifier: Alias para BETO español
  - MultilingualSentimentClassifier: Alias para XLM-RoBERTa multilingüe
- SklearnSentimentClassifier: Usando TF-IDF + sklearn

Todos los clasificadores implementan la misma interfaz y pueden usarse
de manera intercambiable en el pipeline.
"""

from .base import SentimentClassifier
from .keyword_classifier import KeywordSentimentClassifier
from .huggingface_classifier import (
    HuggingFaceSentimentClassifier,
    MarIASentimentClassifier,
    BETOSentimentClassifier,
    MultilingualSentimentClassifier
)
from .sklearn_classifier import SklearnSentimentClassifier

# Alias para retrocompatibilidad con el nombre anterior
EventSentimentClassifier = KeywordSentimentClassifier

__all__ = [
    # Clase base
    'SentimentClassifier',
    
    # Implementaciones
    'KeywordSentimentClassifier',
    'HuggingFaceSentimentClassifier',
    'SklearnSentimentClassifier',
    
    # Aliases de modelos específicos
    'MarIASentimentClassifier',
    'BETOSentimentClassifier',
    'MultilingualSentimentClassifier',
    
    # Retrocompatibilidad
    'EventSentimentClassifier',
]
