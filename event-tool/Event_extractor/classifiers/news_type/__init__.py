"""
Clasificadores de tipos de noticias/eventos.

Este módulo contiene todos los clasificadores que determinan el tipo
de evento/noticia (deportivo, político, cultural, etc.).

Todos los clasificadores heredan de NewsTypeClassifier para garantizar
una interfaz común compatible con el pipeline.
"""

from .base import NewsTypeClassifier
from .keyword_classifier import KeywordNewsClassifier
from .sklearn_classifier import SklearnNewsClassifier

# Alias para retrocompatibilidad
EventTypeClassifier = KeywordNewsClassifier

__all__ = [
    'NewsTypeClassifier',
    'KeywordNewsClassifier',
    'SklearnNewsClassifier',
    'EventTypeClassifier',  # Retrocompatibilidad
]
