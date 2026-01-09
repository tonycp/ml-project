"""
Clase abstracta base para clasificadores de sentimiento.

Define la interfaz común que todos los clasificadores de sentimiento deben implementar
para ser compatibles con el pipeline de extracción de eventos.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from ...models.event import EventSentiment


class SentimentClassifier(ABC):
    """
    Clase abstracta base para clasificadores de sentimiento.
    
    Todos los clasificadores de sentimiento (basados en keywords, ML, transformers, etc.)
    deben heredar de esta clase e implementar sus métodos abstractos.
    """
    
    @abstractmethod
    def classify(self, text: str, threshold: Optional[float] = None) -> Tuple[EventSentiment, float]:
        """
        Clasifica un texto y devuelve el sentimiento más probable.
        
        Args:
            text: Texto a clasificar (noticia, descripción, evento)
            threshold: Umbral de confianza mínimo (opcional)
            
        Returns:
            Tupla (EventSentiment, confidence) donde:
            - EventSentiment: Sentimiento clasificado (POSITIVE, NEGATIVE, NEUTRAL)
            - confidence: Confianza de la clasificación (0.0 a 1.0)
            
        Raises:
            NotImplementedError: Si el método no está implementado
        """
        pass
    
    def predict(self, text: str) -> Tuple[EventSentiment, float]:
        """
        Alias de classify() para compatibilidad con sklearn.
        
        Args:
            text: Texto a clasificar
            
        Returns:
            Tupla (EventSentiment, confidence)
        """
        return self.classify(text)
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[EventSentiment, float]]:
        """
        Clasifica múltiples textos de manera eficiente.
        
        Implementación por defecto que llama a classify() para cada texto.
        Los clasificadores pueden sobrescribir esto para optimizar el procesamiento por lotes.
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de tuplas (EventSentiment, confidence)
        """
        return [self.classify(text) for text in texts]
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Devuelve el nombre del clasificador.
        
        Returns:
            Nombre descriptivo del clasificador (ej: "Keyword-based", "MarIA RoBERTa", "SVM Linear")
        """
        pass
    
    def get_description(self) -> str:
        """
        Devuelve una descripción del clasificador.
        
        Returns:
            Descripción del funcionamiento del clasificador
        """
        return f"{self.get_name()} sentiment classifier"
    
    def supports_training(self) -> bool:
        """
        Indica si el clasificador soporta entrenamiento.
        
        Returns:
            True si puede entrenarse con datos, False si es pre-entrenado o basado en reglas
        """
        return False
    
    def get_sentiment_labels(self) -> List[EventSentiment]:
        """
        Devuelve la lista de sentimientos que puede clasificar.
        
        Returns:
            Lista de EventSentiment soportados
        """
        return [EventSentiment.POSITIVE, EventSentiment.NEGATIVE, EventSentiment.NEUTRAL]
