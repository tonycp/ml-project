"""
Pipeline principal para la extracción de eventos de noticias.
"""

from typing import List, Optional
from datetime import datetime

from ..models.news import NewsContent
from ..models.event import Event, EventType, EventSentiment
from ..extractors.date_extractor import DateExtractor
from ..classifiers.news_type import KeywordNewsClassifier, SklearnNewsClassifier, NewsTypeClassifier
from ..classifiers.sentiment import (
    KeywordSentimentClassifier,
    HuggingFaceSentimentClassifier,
    SklearnSentimentClassifier,
    SentimentClassifier
)
from ..utils.text_preprocessor import _tokenize_text, get_processed_text, extract_svo

class EventExtractionPipeline:
    """
    Pipeline completo para extraer eventos de noticias.
    
    Este pipeline:
    1. Recibe contenido de noticias (NewsContent)
    2. Extrae todas las fechas mencionadas en el texto
    3. Clasifica el tipo de evento
    4. Clasifica el sentimiento del evento
    5. Crea un objeto Event para cada fecha encontrada
    
    Cada fecha se trata como un evento separado, incluso si provienen
    del mismo rango de fechas (ej: inicio y fin de un festival).
    """
    
    def __init__(
        self,
        reference_date: Optional[datetime] = None,
        min_confidence: float = 0.3,
        classify_sentiment: bool = True,
        use_sklearn_classifier: bool = False,
        sklearn_model_path: Optional[str] = None,
        sentiment_classifier: Optional[SentimentClassifier] = None
    ):
        """
        Inicializa el pipeline de extracción.
        
        Args:
            reference_date: Fecha de referencia para resolver fechas relativas.
                          Si no se proporciona, se usará la fecha de metadata de cada noticia.
            min_confidence: Confianza mínima para aceptar una clasificación (default: 0.3)
            classify_sentiment: Si True, clasifica el sentimiento (positivo/negativo) de eventos
            use_sklearn_classifier: Si True, usa el clasificador sklearn (TF-IDF + SVM)
                                   en lugar del clasificador basado en keywords
            sklearn_model_path: Ruta al modelo sklearn entrenado. Si None, usa el modelo
                               por defecto (models/sklearn_spanish_svm.pkl)
            sentiment_classifier: Clasificador de sentimiento personalizado. Si None, usa
                                KeywordSentimentClassifier (por defecto)
        """
        self.default_reference_date = reference_date
        self.min_confidence = min_confidence
        self.classify_sentiment = classify_sentiment
        self.use_sklearn_classifier = use_sklearn_classifier
        
        # Inicializar clasificador de tipos
        if use_sklearn_classifier:
            # Usar clasificador sklearn (TF-IDF + SVM)
            if sklearn_model_path is None:
                # Usar modelo por defecto
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent
                sklearn_model_path = str(project_root / "models" / "sklearn_spanish_svm.pkl")
            
            self.type_classifier = SklearnNewsClassifier(
                model_path=sklearn_model_path,
                use_spacy_tokenizer=True
            )
        else:
            # Usar clasificador basado en keywords (original)
            self.type_classifier = KeywordNewsClassifier()
        
        # Inicializar clasificador de sentimiento
        if sentiment_classifier is not None:
            self.sentiment_classifier = sentiment_classifier
        else:
            # Usar clasificador basado en keywords (por defecto)
            self.sentiment_classifier = KeywordSentimentClassifier()
    
    def extract_events(self, news_content: NewsContent) -> List[Event]:
        """
        Extrae eventos del contenido de una noticia.
        
        Este método realiza:
        1. Extracción de fechas
        2. Clasificación del tipo de evento
        3. Clasificación del sentimiento
        4. Extracción de entidades relacionadas (SVO)
        5. Creación de eventos con toda la información
        
        Args:
            news_content: Contenido de la noticia a procesar
            
        Returns:
            Lista de eventos extraídos, uno por cada fecha encontrada
        """
        # Determinar la fecha de referencia: primero del constructor, luego de la metadata
        reference_date = self.default_reference_date or news_content.publication_date
        
        # Obtener documento procesado con spaCy
        doc = get_processed_text(news_content.text, force=True)

        # Preprocess the text (tokenization and cleaning)
        processed_text = _tokenize_text(news_content.text)

        # Create a date extractor with the appropriate reference date
        date_extractor = DateExtractor(reference_date=reference_date)

        # Extract dates from the preprocessed text
        dates = date_extractor.extract_dates(processed_text)
        
        # Si no hay fechas, la noticia no tiene eventos temporales
        if not dates:
            return []

        # If no dates are found, use the publication date of the news
        if not dates and news_content.publication_date:
            dates = [news_content.publication_date]

        # Classify the type of event using the preprocessed text (as a string)
        text_to_classify = ' '.join(processed_text)
        
        if self.use_sklearn_classifier:
            # Usar clasificador sklearn (devuelve EventType directamente)
            event_type, confidence = self.type_classifier.predict(text_to_classify)
        else:
            # Usar clasificador basado en keywords (original)
            event_type, confidence = self.type_classifier.classify(
                text_to_classify,
                threshold=self.min_confidence
            )

        # Classify the sentiment of the event
        sentiment = EventSentiment.NEUTRAL
        sentiment_confidence = 1.0

        if self.classify_sentiment:
            sentiment, sentiment_confidence = self.sentiment_classifier.classify(
                text_to_classify,
                threshold=self.min_confidence
            )
        
        # Extraer entidades relacionadas usando SVO (Subject-Verb-Object)
        svo_triples = extract_svo(doc)
        
        # Consolidar entidades únicas del texto y SVO
        entidades = self._extract_entities_from_doc(doc, svo_triples)
        
        # Crear un evento para cada fecha encontrada
        events = []
        for date in dates:
            event = Event(
                date=date,
                event_type=event_type,
                sentiment=sentiment,
                source_news_id=news_content.id,
                entidades_asociadas=entidades,
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
    
    def _extract_entities_from_doc(self, doc, svo_triples: List[tuple]) -> List[dict]:
        """
        Extrae entidades nombradas y sus roles desde el documento spaCy y triples SVO.
        
        Args:
            doc: Documento spaCy procesado
            svo_triples: Lista de tuplas (sujeto_texto, verbo_texto, objeto_texto)
            
        Returns:
            Lista de diccionarios con información de entidades:
            [
                {
                    'text': 'El Gobierno español',
                    'role': 'subject',
                    'action': 'anunció',
                    'ent_type': 'ORG'
                },
                ...
            ]
        """
        entities = []
        seen_entities = set()  # Para evitar duplicados
        
        # 1. Extraer entidades nombradas del documento (personas, organizaciones, lugares, etc.)
        for ent in doc.ents:
            entity_key = (ent.text.lower(), ent.label_)
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                entities.append({
                    'text': ent.text,
                    'lemma': ent.lemma_,
                    'role': 'named_entity',
                    'action': None,
                    'ent_type': ent.label_  # PER, ORG, LOC, MISC, etc.
                })
        
        # 2. Extraer información de los triples SVO (sujeto-verbo-objeto)
        for subject_text, verb_text, object_text in svo_triples:
            # Procesar sujeto
            if subject_text and subject_text.strip():
                entity_key = (subject_text.lower(), 'subject')
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities.append({
                        'text': subject_text,
                        'lemma': subject_text.lower(),  # Simplificado
                        'role': 'subject',
                        'action': verb_text,
                        'ent_type': self._detect_entity_type(subject_text, doc)
                    })
            
            # Procesar objeto
            if object_text and object_text.strip():
                entity_key = (object_text.lower(), 'object')
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    entities.append({
                        'text': object_text,
                        'lemma': object_text.lower(),  # Simplificado
                        'role': 'object',
                        'action': verb_text,
                        'ent_type': self._detect_entity_type(object_text, doc)
                    })
            
            # Procesar verbo como acción principal
            if verb_text and verb_text.strip():
                verb_key = (verb_text.lower(), 'action')
                if verb_key not in seen_entities:
                    seen_entities.add(verb_key)
                    entities.append({
                        'text': verb_text,
                        'lemma': verb_text.lower(),
                        'role': 'action',
                        'action': None,
                        'ent_type': None
                    })
        
        return entities
    
    def _detect_entity_type(self, text: str, doc) -> Optional[str]:
        """
        Detecta el tipo de entidad buscando en el documento spaCy.
        
        Args:
            text: Texto de la entidad
            doc: Documento spaCy
            
        Returns:
            Tipo de entidad (PER, ORG, LOC, etc.) o None
        """
        text_lower = text.lower()
        for ent in doc.ents:
            if text_lower in ent.text.lower() or ent.text.lower() in text_lower:
                return ent.label_
        return None
    
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
