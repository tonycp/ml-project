"""
Modelos de datos para representar noticias.
"""

from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime


@dataclass
class NewsMetadata:
    """
    Estructura de datos para almacenar la metadata de una noticia.
    
    Attributes:
        title: Título de la noticia
        date: Fecha de publicación
        source: Fuente de la noticia
        author: Autor de la noticia (opcional)
        category: Categoría o sector (ej: aviación, incidentes, regulaciones)
        url: URL de la noticia original (opcional)
        tags: Etiquetas adicionales (opcional)
    """
    title: str
    date: datetime
    source: str
    author: Optional[str] = None
    category: Optional[str] = None
    url: Optional[str] = None
    tags: Optional[list[str]] = None


@dataclass
class NewsContent:
    """
    Estructura de datos para almacenar el contenido procesado de una noticia.
    
    Attributes:
        text: Texto completo de la noticia
        metadata: Metadata asociada a la noticia
        title: Título de la noticia (opcional)
        raw_data: Datos originales sin procesar (opcional)
    """
    text: str
    metadata: NewsMetadata
    title: Optional[str] = None
    raw_data: Optional[Any] = None
