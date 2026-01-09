"""
Modelos de datos para representar noticias.
"""

from dataclasses import dataclass
from datetime import datetime

@dataclass
class NewsContent:
    """
    Estructura de datos para almacenar el contenido procesado de una noticia.
    
    Attributes:
        id: Identificador único de la noticia.
        text: Texto completo de la noticia.
        publication_date: Fecha de publicación de la noticia.
    """
    id: str
    text: str
    publication_date: datetime
