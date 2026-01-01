"""
Modelos de datos para el paquete de extracción de eventos.
"""

from .news import NewsMetadata, NewsContent
from .event import Event, EventType, EventSentiment

__all__ = [
    'NewsMetadata',
    'NewsContent', 
    'Event',
    'EventType',
    'EventSentiment'
]
