"""
Clase abstracta base para clasificadores de tipos de noticias/eventos.

Define la interfaz común que todos los clasificadores de tipos deben implementar
para ser compatibles con el pipeline de extracción de eventos.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from ...models.event import EventType


class NewsTypeClassifier(ABC):
    """
    Clase abstracta base para clasificadores de tipos de noticias.
    
    Todos los clasificadores de tipos (basados en keywords, ML, etc.)
    deben heredar de esta clase e implementar sus métodos abstractos.
    """
    
    @abstractmethod
    def classify(self, text: str, threshold: Optional[float] = None) -> Tuple[EventType, float]:
        """
        Clasifica un texto y devuelve el tipo de evento más probable.
        
        Args:
            text: Texto a clasificar (noticia, descripción, etc.)
            threshold: Umbral de confianza mínimo (opcional)
            
        Returns:
            Tupla (EventType, confidence) donde:
            - EventType: Tipo de evento clasificado
            - confidence: Confianza de la clasificación (0.0 a 1.0)
            
        Raises:
            NotImplementedError: Si el método no está implementado
        """
        pass
    
    def predict(self, text: str) -> Tuple[EventType, float]:
        """
        Alias de classify() para compatibilidad con sklearn.
        
        Args:
            text: Texto a clasificar
            
        Returns:
            Tupla (EventType, confidence)
        """
        return self.classify(text)
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[EventType, float]]:
        """
        Clasifica múltiples textos de manera eficiente.
        
        Implementación por defecto que llama a classify() para cada texto.
        Los clasificadores pueden sobrescribir esto para optimizar el procesamiento por lotes.
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de tuplas (EventType, confidence)
        """
        return [self.classify(text) for text in texts]
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Devuelve el nombre del clasificador.
        
        Returns:
            Nombre descriptivo del clasificador (ej: "Keyword-based", "SVM Linear")
        """
        pass
    
    def get_description(self) -> str:
        """
        Devuelve una descripción del clasificador.
        
        Returns:
            Descripción opcional del clasificador
        """
        return f"{self.get_name()} classifier"
    
    def supports_training(self) -> bool:
        """
        Indica si el clasificador soporta entrenamiento.
        
        Returns:
            True si el clasificador puede ser entrenado, False si es estático
        """
        return False
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {self.get_name()}>"
    
    def __repr__(self) -> str:
        return self.__str__()
