"""
event_extractor.py - Archivo heredado para compatibilidad hacia atrás

Este archivo mantiene las clases originales para compatibilidad.
Para nuevos desarrollos, usa los módulos específicos en:
- models/ para estructuras de datos
- extractors/ para extractores
- classifiers/ para clasificadores
- pipeline/ para el pipeline completo
"""

from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime

# Importar desde los nuevos módulos
from .models.news import NewsMetadata, NewsContent
from .models.event import Event, EventType


@dataclass
class NewsProcessor:
    """
    Clase base para procesar noticias en formato JSON.
    Proporciona métodos para validar y estructurar datos de noticias.
    """
    
    def process_news(self, raw_input: dict) -> NewsContent:
        """
        Procesa una noticia en formato JSON y devuelve un objeto NewsContent.
        
        Args:
            raw_input: Diccionario con los datos de la noticia.
        
        Returns:
            NewsContent: Objeto que contiene el texto y metadata de la noticia.
        """
        # Esta es una implementación placeholder
        # El formato exacto dependerá de cómo lleguen las noticias
        
        metadata = NewsMetadata(
            title=raw_input.get('title', ''),
            date=raw_input.get('date', datetime.now()),
            source=raw_input.get('source', ''),
            author=raw_input.get('author'),
            category=raw_input.get('category'),
            url=raw_input.get('url'),
            tags=raw_input.get('tags')
        )
        
        news_content = NewsContent(
            text=raw_input.get('text', ''),
            metadata=metadata,
            title=raw_input.get('title'),
            raw_data=raw_input
        )
        
        return news_content


# Exportar todo para compatibilidad
__all__ = [
    'NewsMetadata',
    'NewsContent',
    'Event',
    'EventType',
    'NewsProcessor'
]
