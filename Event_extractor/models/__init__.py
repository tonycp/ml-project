"""
Modelos de datos para el paquete de extracción de eventos.
"""

from .news import NewsContent
from .event import Event, EventType, EventSentiment

__all__ = [
    'NewsContent', 
    'Event',
    'EventType',
    'EventSentiment'
]
