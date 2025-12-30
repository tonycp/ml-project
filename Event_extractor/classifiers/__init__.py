"""
Módulo de clasificadores de eventos.
"""

from .event_type_classifier import EventTypeClassifier
from .event_sentiment_classifier import EventSentimentClassifier

__all__ = ['EventTypeClassifier', 'EventSentimentClassifier']
