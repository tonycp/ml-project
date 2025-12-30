"""
Pipeline principal para la extracción de eventos de noticias.
"""

from typing import List, Optional
from datetime import datetime

from ..models.news import NewsContent
from ..models.event import Event, EventType, EventSentiment
from ..extractors.date_extractor import DateExtractor
from ..classifiers.event_type_classifier import EventTypeClassifier
from ..classifiers.event_sentiment_classifier import EventSentimentClassifier
from ..utils.text_preprocessor import _tokenize_text


class EventExtractionPipeline:
    """
    Pipeline completo para extraer eventos de noticias.
    
    Este pipeline:
    1. Recibe contenido de noticias (NewsContent)
    2. Extrae todas las fechas mencionadas en el texto
    3. Clasifica el tipo de evento
    4. Crea un objeto Event para cada fecha encontrada
    
    Cada fecha se trata como un evento separado, incluso si provienen
    del mismo rango de fechas (ej: inicio y fin de un festival).
    """
    
    def __init__(
        self,
        reference_date: Optional[datetime] = None,
        min_confidence: float = 0.3,
        classify_sentiment: bool = True
    ):
        """
        Inicializa el pipeline de extracción.
        
        Args:
            reference_date: Fecha de referencia para resolver fechas relativas.
                          Si no se proporciona, se usará la fecha de metadata de cada noticia.
            min_confidence: Confianza mínima para aceptar una clasificación
            classify_sentiment: Si True, clasifica el sentimiento (positivo/negativo) de eventos
        """
        self.default_reference_date = reference_date
        self.type_classifier = EventTypeClassifier()
        self.sentiment_classifier = EventSentimentClassifier()
        self.min_confidence = min_confidence
        self.classify_sentiment = classify_sentiment
    
    def extract_events(self, news_content: NewsContent) -> List[Event]:
        """
        Extrae eventos del contenido de una noticia.
        
        Args:
            news_content: Contenido de la noticia a procesar
            
        Returns:
            Lista de eventos extraídos, uno por cada fecha encontrada
        """
        # Determinar la fecha de referencia: primero del constructor, luego de la metadata
        reference_date = self.default_reference_date or news_content.metadata.date
        
        # Preprocesar el texto (tokenización y limpieza)
        processed_text = _tokenize_text(news_content.text)
        processed_title = _tokenize_text(news_content.title) if news_content.title else None
        
        # Crear extractor de fechas con la fecha de referencia apropiada
        date_extractor = DateExtractor(reference_date=reference_date)
        
        # Extraer fechas del texto preprocesado
        dates = date_extractor.extract_dates(processed_text)
        
        # Si no se encontraron fechas, intentar con el título preprocesado
        if not dates and processed_title:
            dates = date_extractor.extract_dates(processed_title)
        
        # TODO: Revisar esto, puede q la fecha de publicacion no sea confiable
        # Si aún no hay fechas, usar la fecha de publicación de la noticia
        if not dates and news_content.metadata.date:
            dates = [news_content.metadata.date]
        
        # Clasificar el tipo de evento usando el texto preprocesado (como string)
        text_to_classify = ' '.join(processed_text)
        if processed_title:
            text_to_classify = f"{' '.join(processed_title)}. {text_to_classify}"
        event_type, confidence = self.type_classifier.classify(
            text_to_classify,
            threshold=self.min_confidence
        )
        
        # Clasificar el sentimiento del evento
        sentiment = EventSentiment.NEUTRAL
        sentiment_confidence = 1.0
        
        if self.classify_sentiment:
            sentiment, sentiment_confidence = self.sentiment_classifier.classify(
                text_to_classify,
                threshold=self.min_confidence
            )
        
        # Crear un evento para cada fecha encontrada
        events = []
        for date in dates:
            event = Event(
                date=date,
                event_type=event_type,
                sentiment=sentiment,
                title=news_content.title or news_content.metadata.title,
                description=self._extract_description(news_content.text),
                source_news_id=news_content.metadata.url,
                confidence=confidence,
                sentiment_confidence=sentiment_confidence
            )
            events.append(event)
        
        return events
    
    def extract_events_batch(
        self,
        news_contents: List[NewsContent]
    ) -> List[Event]:
        """
        Extrae eventos de múltiples noticias.
        
        Args:
            news_contents: Lista de contenidos de noticias
            
        Returns:
            Lista consolidada de todos los eventos extraídos
        """
        all_events = []
        
        for news_content in news_contents:
            events = self.extract_events(news_content)
            all_events.extend(events)
        
        return all_events
    
    def _extract_description(self, text: str, max_length: int = 200) -> str:
        """
        Extrae una descripción resumida del texto.
        
        Args:
            text: Texto completo
            max_length: Longitud máxima de la descripción
            
        Returns:
            Descripción resumida
        """
        if len(text) <= max_length:
            return text
        
        # Buscar el final de la primera oración
        for i, char in enumerate(text[:max_length + 50]):
            if char in '.!?' and i > max_length // 2:
                return text[:i + 1].strip()
        
        # Si no se encuentra un punto, cortar en la última palabra completa
        truncated = text[:max_length].rsplit(' ', 1)[0]
        return truncated + '...'
    
    def set_min_confidence(self, confidence: float):
        """
        Establece la confianza mínima para aceptar clasificaciones.
        
        Args:
            confidence: Valor entre 0.0 y 1.0
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence debe estar entre 0.0 y 1.0")
        self.min_confidence = confidence
    
    def add_custom_keywords(self, event_type: EventType, keywords: List[str]):
        """
        Añade palabras clave personalizadas para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            keywords: Lista de palabras clave a añadir
        """
        self.type_classifier.add_keywords(event_type, keywords)


class EventAggregator:
    """
    Agregador de eventos para consolidar y eliminar duplicados.
    """
    
    @staticmethod
    def remove_duplicates(
        events: List[Event],
        date_tolerance_days: int = 0
    ) -> List[Event]:
        """
        Elimina eventos duplicados basándose en fecha y tipo.
        
        Args:
            events: Lista de eventos
            date_tolerance_days: Tolerancia en días para considerar fechas iguales
            
        Returns:
            Lista de eventos sin duplicados
        """
        if not events:
            return []
        
        unique_events = []
        seen = set()
        
        for event in events:
            # Crear una clave única para el evento
            date_key = event.date.date() if date_tolerance_days == 0 else \
                       (event.date.year, event.date.month, event.date.day // (date_tolerance_days + 1))
            
            key = (date_key, event.event_type, event.title)
            
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        return unique_events
    
    @staticmethod
    def filter_by_date_range(
        events: List[Event],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Event]:
        """
        Filtra eventos por rango de fechas.
        
        Args:
            events: Lista de eventos
            start_date: Fecha de inicio (inclusive)
            end_date: Fecha de fin (inclusive)
            
        Returns:
            Lista de eventos filtrados
        """
        filtered = events
        
        if start_date:
            filtered = [e for e in filtered if e.date >= start_date]
        
        if end_date:
            filtered = [e for e in filtered if e.date <= end_date]
        
        return filtered
    
    @staticmethod
    def filter_by_type(
        events: List[Event],
        event_types: List[EventType]
    ) -> List[Event]:
        """
        Filtra eventos por tipo.
        
        Args:
            events: Lista de eventos
            event_types: Lista de tipos de eventos a incluir
            
        Returns:
            Lista de eventos filtrados
        """
        return [e for e in events if e.event_type in event_types]
    
    @staticmethod
    def sort_by_date(events: List[Event], reverse: bool = False) -> List[Event]:
        """
        Ordena eventos por fecha.
        
        Args:
            events: Lista de eventos
            reverse: Si True, ordena de más reciente a más antiguo
            
        Returns:
            Lista de eventos ordenados
        """
        return sorted(events, key=lambda e: e.date, reverse=reverse)
