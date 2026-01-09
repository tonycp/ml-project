"""
Clasificadores de Machine Learning para noticias.

Este módulo contiene utilidades para entrenamiento de clasificadores ML
(corpus loaders y configuraciones de modelos).
"""

from .corpus_loaders import (
    CorpusLoader,
    AGNewsCorpus,
    SpanishNewsCorpus,
    CorpusStats
)
from .model_configs import (
    ModelConfig,
    get_model_config,
    get_model_configs,
    list_available_models,
    list_model_sets
)

__all__ = [
    'CorpusLoader',
    'AGNewsCorpus',
    'SpanishNewsCorpus',
    'CorpusStats',
    'ModelConfig',
    'get_model_config',
    'get_model_configs',
    'list_available_models',
    'list_model_sets',
]
