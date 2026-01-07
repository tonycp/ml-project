"""
Clasificador de sentimiento usando modelos pre-entrenados de HuggingFace.
"""

from typing import Tuple, List, Optional, Dict
from .base import SentimentClassifier
from ...models.event import EventSentiment


class HuggingFaceSentimentClassifier(SentimentClassifier):
    """
    Clasificador de sentimiento usando modelos transformers de HuggingFace.
    
    Soporta modelos pre-entrenados como:
    - UMUTeam/roberta-spanish-sentiment-analysis (MarIA/RoBERTa español)
    - finiteautomata/beto-sentiment-analysis (BETO español)
    - cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingüe)
    - tabularisai/multilingual-sentiment-analysis (multilingüe)
    """
    
    def __init__(self, model_name: str = "UMUTeam/roberta-spanish-sentiment-analysis", 
                 device: Optional[str] = None):
        """
        Inicializa el clasificador con un modelo de HuggingFace.
        
        Args:
            model_name: Nombre del modelo en HuggingFace Hub
            device: Dispositivo a usar ('cpu', 'cuda', None para auto-detectar)
        """
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self._label_mapping = None
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo y pipeline de HuggingFace."""
        try:
            from transformers import pipeline
            
            # Cargar pipeline de clasificación
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device
            )
            
            # Detectar mapeo de labels según el modelo
            self._detect_label_mapping()
            
        except ImportError:
            raise ImportError(
                "transformers no está instalado. "
                "Instálalo con: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error al cargar el modelo {self.model_name}: {str(e)}"
            )
    
    def _detect_label_mapping(self):
        """
        Detecta el mapeo de labels del modelo a nuestros EventSentiment.
        
        Los modelos pueden usar diferentes nombres para las clases:
        - POS/NEG/NEU
        - POSITIVE/NEGATIVE/NEUTRAL
        - 1/0/-1
        - positive/negative/neutral
        """
        # Mapeos comunes de modelos a nuestros EventSentiment
        self._label_mapping = {
            # Formato estándar
            'POS': EventSentiment.POSITIVE,
            'NEG': EventSentiment.NEGATIVE,
            'NEU': EventSentiment.NEUTRAL,
            'POSITIVE': EventSentiment.POSITIVE,
            'NEGATIVE': EventSentiment.NEGATIVE,
            'NEUTRAL': EventSentiment.NEUTRAL,
            'positive': EventSentiment.POSITIVE,
            'negative': EventSentiment.NEGATIVE,
            'neutral': EventSentiment.NEUTRAL,
            # Formato numérico
            'LABEL_0': EventSentiment.NEGATIVE,
            'LABEL_1': EventSentiment.NEUTRAL,
            'LABEL_2': EventSentiment.POSITIVE,
            # Otros formatos
            '0': EventSentiment.NEGATIVE,
            '1': EventSentiment.NEUTRAL,
            '2': EventSentiment.POSITIVE,
        }
    
    def classify(self, text: str, threshold: Optional[float] = None) -> Tuple[EventSentiment, float]:
        """
        Clasifica el sentimiento del texto usando el modelo de HuggingFace.
        
        Args:
            text: Texto a clasificar
            threshold: Umbral mínimo de confianza (no usado, mantenido por compatibilidad)
            
        Returns:
            Tupla (EventSentiment, confidence)
        """
        if not text or not text.strip():
            return EventSentiment.NEUTRAL, 0.5
        
        try:
            # Truncar texto si es muy largo (máximo 512 tokens para la mayoría de modelos)
            text = text[:2000]
            
            # Clasificar con el modelo
            result = self._pipeline(text)[0]
            
            # Extraer label y score
            label = result['label']
            confidence = result['score']
            
            # Mapear label del modelo a nuestro EventSentiment
            sentiment = self._label_mapping.get(label, EventSentiment.NEUTRAL)
            
            return sentiment, float(confidence)
            
        except Exception as e:
            # En caso de error, devolver neutral con baja confianza
            print(f"Error en clasificación: {str(e)}")
            return EventSentiment.NEUTRAL, 0.5
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[EventSentiment, float]]:
        """
        Clasifica múltiples textos de manera eficiente usando batch processing.
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de tuplas (EventSentiment, confidence)
        """
        if not texts:
            return []
        
        try:
            # Truncar textos
            texts = [text[:2000] if text else "" for text in texts]
            
            # Clasificar en batch
            results = self._pipeline(texts)
            
            # Convertir resultados
            classified = []
            for result in results:
                label = result['label']
                confidence = result['score']
                sentiment = self._label_mapping.get(label, EventSentiment.NEUTRAL)
                classified.append((sentiment, float(confidence)))
            
            return classified
            
        except Exception as e:
            print(f"Error en clasificación batch: {str(e)}")
            # Fallback: clasificar uno por uno
            return [self.classify(text) for text in texts]
    
    def get_name(self) -> str:
        """Devuelve el nombre del clasificador."""
        # Extraer nombre corto del modelo
        model_short = self.model_name.split('/')[-1]
        return f"HuggingFace: {model_short}"
    
    def get_description(self) -> str:
        """Devuelve una descripción del clasificador."""
        return f"Clasificador usando modelo transformer pre-entrenado: {self.model_name}"
    
    def supports_training(self) -> bool:
        """Indica que este modelo está pre-entrenado y no requiere entrenamiento adicional."""
        return False
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Devuelve información sobre el modelo cargado.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'task': 'sentiment-analysis',
            'supports_batch': 'yes'
        }


# Aliases para modelos específicos comunes
class MarIASentimentClassifier(HuggingFaceSentimentClassifier):
    """
    Clasificador usando MarIA/RoBERTa español para sentiment analysis.
    
    Basado en UMUTeam/roberta-spanish-sentiment-analysis, que es una versión
    fine-tuned de PlanTL-GOB-ES/roberta-base-bne (MarIA) del BSC.
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(
            model_name="UMUTeam/roberta-spanish-sentiment-analysis",
            device=device
        )
    
    def get_name(self) -> str:
        return "MarIA RoBERTa (Spanish)"


class BETOSentimentClassifier(HuggingFaceSentimentClassifier):
    """
    Clasificador usando BETO (BERT español) para sentiment analysis.
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(
            model_name="finiteautomata/beto-sentiment-analysis",
            device=device
        )
    
    def get_name(self) -> str:
        return "BETO (Spanish)"


class MultilingualSentimentClassifier(HuggingFaceSentimentClassifier):
    """
    Clasificador multilingüe para sentiment analysis.
    Funciona en múltiples idiomas incluyendo español.
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(
            model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=device
        )
    
    def get_name(self) -> str:
        return "XLM-RoBERTa (Multilingual)"
