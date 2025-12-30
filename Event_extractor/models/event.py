"""
Modelos de datos para representar eventos extraídos.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum


class EventSentiment(str, Enum):
    """
    Sentimiento o polaridad de un evento.
    """
    POSITIVE = "positive"  # Eventos positivos: festivales, inauguraciones, celebraciones
    NEGATIVE = "negative"  # Eventos negativos: cancelaciones, protestas, incidentes
    NEUTRAL = "neutral"    # Eventos neutros: reuniones, anuncios informativos
    
    @classmethod
    def from_string(cls, value: str) -> 'EventSentiment':
        """
        Convierte un string a EventSentiment.
        
        Args:
            value: String que representa el sentimiento
            
        Returns:
            EventSentiment correspondiente, o NEUTRAL si no se reconoce
        """
        try:
            return cls(value.lower())
        except ValueError:
            return cls.NEUTRAL


class EventType(str, Enum):
    """
    Tipos de eventos que pueden ser extraídos de las noticias.
    """
    CULTURAL = "cultural"
    DEPORTIVO = "deportivo"
    METEOROLOGICO = "meteorologico"
    POLITICO = "politico"
    ECONOMICO = "economico"
    SOCIAL = "social"
    INCIDENTE = "incidente"
    REGULACION = "regulacion"
    OTRO = "otro"
    
    @classmethod
    def from_string(cls, value: str) -> 'EventType':
        """
        Convierte un string a EventType.
        
        Args:
            value: String que representa el tipo de evento
            
        Returns:
            EventType correspondiente, o OTRO si no se reconoce
        """
        try:
            return cls(value.lower())
        except ValueError:
            return cls.OTRO


@dataclass
class Event:
    """
    Estructura de datos para almacenar un evento extraído de una noticia.
    
    Cada fecha de un evento se modela como un evento separado. Por ejemplo,
    si un festival tiene lugar del 1 al 5 de enero, se crearán eventos para
    la fecha de inicio (1 de enero) y la fecha de fin (5 de enero).
    
    Attributes:
        date: Fecha del evento
        event_type: Tipo de evento
        sentiment: Sentimiento del evento (positivo, negativo, neutral)
        title: Título o descripción del evento (opcional)
        description: Descripción detallada del evento (opcional)
        source_news_id: ID de la noticia de origen (opcional)
        confidence: Nivel de confianza en la extracción (0.0 a 1.0)
        sentiment_confidence: Nivel de confianza en la clasificación de sentimiento (0.0 a 1.0)
    """
    date: datetime
    event_type: EventType
    sentiment: EventSentiment = EventSentiment.NEUTRAL
    title: Optional[str] = None # Quizas innecesario
    description: Optional[str] = None # Quizas innecesario
    source_news_id: Optional[str] = None # Quizas innecesario
    entidades_asociadas: Optional[list] = None  # Lista de entidades nombradas asociadas al evento
    confidence: float = 1.0
    sentiment_confidence: float = 1.0
    
    def __post_init__(self):
        """Valida los valores después de la inicialización."""
        if not isinstance(self.event_type, EventType):
            self.event_type = EventType.from_string(str(self.event_type))
        
        if not isinstance(self.sentiment, EventSentiment):
            self.sentiment = EventSentiment.from_string(str(self.sentiment))
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence debe estar entre 0.0 y 1.0")
        
        if not 0.0 <= self.sentiment_confidence <= 1.0:
            raise ValueError("sentiment_confidence debe estar entre 0.0 y 1.0")
