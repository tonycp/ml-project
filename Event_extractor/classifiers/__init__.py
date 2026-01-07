"""
Módulo de clasificadores de eventos.

Contiene:
- news_type/: Clasificadores de tipos de noticias (keywords, ML)
- sentiment/: Clasificadores de sentimiento (keywords, transformers, ML)
- ml/: Utilidades ML (corpus_loaders, model_configs)
"""

from .news_type import (
    NewsTypeClassifier,
    KeywordNewsClassifier,
    SklearnNewsClassifier,
    EventTypeClassifier,  # Alias para retrocompatibilidad
)

from .sentiment import (
    SentimentClassifier,
    KeywordSentimentClassifier,
    HuggingFaceSentimentClassifier,
    SklearnSentimentClassifier,
    MarIASentimentClassifier,
    BETOSentimentClassifier,
    MultilingualSentimentClassifier,
    EventSentimentClassifier,  # Alias para retrocompatibilidad
)

__all__ = [
    # News type classifiers
    'NewsTypeClassifier',
    'KeywordNewsClassifier',
    'SklearnNewsClassifier',
    'EventTypeClassifier',
    
    # Sentiment classifiers
    'SentimentClassifier',
    'KeywordSentimentClassifier',
    'HuggingFaceSentimentClassifier',
    'SklearnSentimentClassifier',
    'MarIASentimentClassifier',
    'BETOSentimentClassifier',
    'MultilingualSentimentClassifier',
    'EventSentimentClassifier',
]
