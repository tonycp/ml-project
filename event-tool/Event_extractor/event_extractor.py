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
from .models.news import NewsContent
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
        
        news_content = NewsContent(
            id=raw_input.get('id', 'unknown'),
            text=raw_input.get('text', ''),
            publication_date=raw_input.get('date', datetime.now())
        )
        
        return news_content


# Exportar todo para compatibilidad
__all__ = [
    'NewsContent',
    'Event',
    'EventType',
    'NewsProcessor'
]
