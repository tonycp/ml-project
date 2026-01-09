"""
Clasificador de sentimiento basado en palabras clave y reglas.
"""

from typing import Dict, List, Tuple, Optional
from .base import SentimentClassifier
from ...models.event import EventSentiment


class KeywordSentimentClassifier(SentimentClassifier):
    """
    Clasificador de sentimiento basado en palabras clave.
    
    Determina si un evento es positivo (celebración, logro, inauguración),
    negativo (cancelación, protesta, incidente) o neutral usando análisis
    de palabras clave y reglas heurísticas.
    """
    
    def __init__(self):
        """Inicializa el clasificador con las palabras clave para cada sentimiento."""
        self._initialize_keywords()
    
    def _initialize_keywords(self):
        """Define las palabras clave para cada tipo de sentimiento."""
        self.positive_keywords = [
            # Celebraciones y festividades
            'festival', 'celebración', 'fiesta', 'concierto', 'inauguración',
            'apertura', 'estreno', 'lanzamiento', 'debut', 'premio',
            
            # Logros y éxitos
            'victoria', 'triunfo', 'éxito', 'logro', 'récord', 'ganador',
            'campeón', 'medalla', 'título', 'galardón',
            
            # Eventos positivos
            'boda', 'graduación', 'aniversario', 'cumpleaños', 'homenaje',
            'reconocimiento', 'condecoración', 'ovación',
            
            # Mejoras y avances
            'mejora', 'avance', 'progreso', 'desarrollo', 'crecimiento',
            'recuperación', 'beneficio', 'ganancia', 'aumento',
            
            # Entretenimiento positivo
            'espectáculo', 'gala', 'show', 'presentación', 'actuación',
            'exhibición', 'exposición', 'muestra',
            
            # Palabras positivas generales
            'feliz', 'alegre', 'exitoso', 'favorable', 'positivo',
            'prometedor', 'esperanzador', 'alentador', 'optimista'
        ]
        
        self.negative_keywords = [
            # Cancelaciones y suspensiones
            'cancelación', 'cancelado', 'suspensión', 'suspendido', 
            'aplazado', 'pospuesto', 'diferido', 'anulado',
            
            # Protestas y conflictos
            'protesta', 'manifestación', 'huelga', 'paro', 'bloqueo',
            'conflicto', 'enfrentamiento', 'disputa', 'polémica',
            
            # Incidentes y accidentes
            'accidente', 'incidente', 'choque', 'colisión', 'caída',
            'explosión', 'incendio', 'derrumbe', 'colapso',
            
            # Desastres y emergencias
            'desastre', 'catástrofe', 'tragedia', 'crisis', 'emergencia',
            'calamidad', 'devastación', 'destrucción',
            
            # Pérdidas y derrotas
            'derrota', 'pérdida', 'fracaso', 'declive', 'caída',
            'descenso', 'reducción', 'disminución',
            
            # Problemas y daños
            'problema', 'daño', 'perjuicio', 'afectación', 'deterioro',
            'falla', 'avería', 'rotura',
            
            # Víctimas y heridos
            'víctima', 'herido', 'fallecido', 'muerto', 'lesionado',
            'afectado', 'damnificado',
            
            # Situaciones negativas
            'cierre', 'clausura', 'quiebra', 'bancarrota', 'despido',
            'recorte', 'sanción', 'multa', 'penalización',
            
            # Palabras negativas generales
            'grave', 'crítico', 'severo', 'preocupante', 'alarmante',
            'peligroso', 'riesgoso', 'amenaza', 'temor', 'pánico'
        ]
        
        self.neutral_keywords = [
            # Reuniones y encuentros
            'reunión', 'encuentro', 'conferencia', 'congreso', 'seminario',
            'taller', 'simposio', 'foro', 'mesa redonda',
            
            # Anuncios y comunicados
            'anuncio', 'comunicado', 'declaración', 'informe', 'reporte',
            'presentación', 'publicación',
            
            # Procesos administrativos
            'elección', 'votación', 'consulta', 'censo', 'registro',
            'trámite', 'proceso', 'procedimiento',
            
            # Eventos informativos
            'conferencia de prensa', 'rueda de prensa', 'entrevista',
            'comparecencia', 'audiencia'
        ]
        
        # Normalizar a minúsculas
        self.positive_keywords = [kw.lower() for kw in self.positive_keywords]
        self.negative_keywords = [kw.lower() for kw in self.negative_keywords]
        self.neutral_keywords = [kw.lower() for kw in self.neutral_keywords]
    
    def classify(self, text: str, threshold: Optional[float] = 0.3) -> Tuple[EventSentiment, float]:
        """
        Clasifica el sentimiento del texto.
        
        Args:
            text: Texto a clasificar
            threshold: Umbral mínimo de diferencia para asignar sentimiento no neutral
            
        Returns:
            Tupla (EventSentiment, confidence) donde confidence es un valor entre 0 y 1
        """
        if threshold is None:
            threshold = 0.3
            
        text_lower = text.lower()
        
        # Contar coincidencias para cada sentimiento
        positive_score = self._count_matches(text_lower, self.positive_keywords)
        negative_score = self._count_matches(text_lower, self.negative_keywords)
        neutral_score = self._count_matches(text_lower, self.neutral_keywords)
        
        if positive_score == 0 and negative_score == 0 and neutral_score == 0:
            # Sin coincidencias, asumir neutral con baja confianza
            return EventSentiment.NEUTRAL, 0.5
        
        # Calcular puntuaciones normalizadas
        total_score = positive_score + negative_score + neutral_score
        
        # if total_score == 0:
        #     # Sin coincidencias, asumir neutral con baja confianza
        #     return EventSentiment.NEUTRAL, 0.5
        
        # Normalizar puntuaciones
        pos_norm = positive_score / total_score
        neg_norm = negative_score / total_score
        neu_norm = neutral_score / total_score
        
        # Determinar el sentimiento dominante
        max_score = max(pos_norm, neg_norm, neu_norm)
        
        # Si la diferencia no es significativa, devolver neutral
        if max_score < threshold:
            return EventSentiment.NEUTRAL, 0.5
        
        if pos_norm == max_score:
            sentiment = EventSentiment.POSITIVE
            confidence = min(pos_norm * 2, 1.0)  # Escalar confianza
        elif neg_norm == max_score:
            sentiment = EventSentiment.NEGATIVE
            confidence = min(neg_norm * 2, 1.0)
        else:
            sentiment = EventSentiment.NEUTRAL
            confidence = min(neu_norm * 2, 1.0)
        
        return sentiment, confidence
    
    def get_name(self) -> str:
        """Devuelve el nombre del clasificador."""
        return "Keyword-based Sentiment Classifier"
    
    def get_description(self) -> str:
        """Devuelve una descripción del clasificador."""
        return "Clasificador basado en palabras clave y reglas manuales para análisis de sentimiento"
    
    def supports_training(self) -> bool:
        """Indica que este clasificador no requiere entrenamiento."""
        return False
    
    def classify_detailed(self, text: str) -> Dict[EventSentiment, float]:
        """
        Clasifica el texto y devuelve las puntuaciones para todos los sentimientos.
        
        Args:
            text: Texto a clasificar
            
        Returns:
            Diccionario con las puntuaciones para cada sentimiento
        """
        text_lower = text.lower()
        
        positive_score = self._count_matches(text_lower, self.positive_keywords)
        negative_score = self._count_matches(text_lower, self.negative_keywords)
        neutral_score = self._count_matches(text_lower, self.neutral_keywords)
        
        if positive_score == 0 and negative_score == 0 and neutral_score == 0:
            return {
                EventSentiment.POSITIVE: 0.0,
                EventSentiment.NEGATIVE: 0.0,
                EventSentiment.NEUTRAL: 1.0
            }
            
        total = positive_score + negative_score + neutral_score
        
        return {
            EventSentiment.POSITIVE: positive_score / total,
            EventSentiment.NEGATIVE: negative_score / total,
            EventSentiment.NEUTRAL: neutral_score / total
        }
    
    def _count_matches(self, text: str, keywords: List[str]) -> float:
        """
        Cuenta las coincidencias de palabras clave en el texto.
        
        Args:
            text: Texto en minúsculas
            keywords: Lista de palabras clave
            
        Returns:
            Número de coincidencias
        """
        count = sum(1 for keyword in keywords if keyword in text)
        return count
    
    def add_positive_keywords(self, keywords: List[str]):
        """
        Añade palabras clave positivas.
        
        Args:
            keywords: Lista de nuevas palabras clave
        """
        self.positive_keywords.extend([kw.lower() for kw in keywords])
    
    def add_negative_keywords(self, keywords: List[str]):
        """
        Añade palabras clave negativas.
        
        Args:
            keywords: Lista de nuevas palabras clave
        """
        self.negative_keywords.extend([kw.lower() for kw in keywords])
    
    def add_neutral_keywords(self, keywords: List[str]):
        """
        Añade palabras clave neutrales.
        
        Args:
            keywords: Lista de nuevas palabras clave
        """
        self.neutral_keywords.extend([kw.lower() for kw in keywords])
