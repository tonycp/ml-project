"""
Event Extractor - Librería para extracción de eventos de noticias

Este paquete proporciona herramientas para extraer eventos (con fecha y tipo)
de contenido de noticias en español.

Uso básico:
    from src.Event_extractor import EventExtractionPipeline, NewsContent
    from datetime import datetime
    
    # Crear pipeline
    pipeline = EventExtractionPipeline()
    
    # Crear contenido de noticia
    news = NewsContent(
        id="news_001",
        text="El festival se realizará del 10 al 15 de enero de 2025...",
        publication_date=datetime.now()
    )
    
    # Extraer eventos
    events = pipeline.extract_events(news)
    
    for event in events:
        print(f"{event.date}: {event.event_type} - {event.title}")
"""

__version__ = "0.1.0"

# Importar clases principales
from .models import NewsContent, Event, EventType, EventSentiment
from .pipeline import EventExtractionPipeline, EventAggregator
from .extractors import DateExtractor
from .classifiers import EventTypeClassifier, EventSentimentClassifier

__all__ = [
    # Modelos de datos
    'NewsContent',
    'Event',
    'EventType',
    'EventSentiment',
    
    # Pipeline principal
    'EventExtractionPipeline',
    'EventAggregator',
    
    # Componentes individuales
    'DateExtractor',
    'EventTypeClassifier',
    'EventSentimentClassifier',
]
