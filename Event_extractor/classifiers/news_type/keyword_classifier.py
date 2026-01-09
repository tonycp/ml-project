"""
Clasificador de tipos de eventos usando reglas y keywords.
Implementa la interfaz NewsTypeClassifier.
"""

from typing import Dict, List, Tuple, Optional
from ...models.event import EventType
from .base import NewsTypeClassifier


class KeywordNewsClassifier(NewsTypeClassifier):
    """
    Clasificador de tipos de noticias basado en palabras clave y reglas.
    
    Analiza el texto de una noticia para determinar el tipo de evento
    (cultural, deportivo, meteorológico, etc.) basándose en la presencia
    de palabras clave específicas.
    
    Hereda de NewsTypeClassifier para ser compatible con el pipeline.
    """
    
    def __init__(self):
        """Inicializa el clasificador con las palabras clave para cada tipo."""
        self._initialize_keywords()
    
    def get_name(self) -> str:
        """Nombre del clasificador."""
        return "Keyword-based Classifier"
    
    def get_description(self) -> str:
        """Descripción del clasificador."""
        return "Clasificador basado en palabras clave y reglas manuales"
    
    def _initialize_keywords(self):
        """Define las palabras clave para cada tipo de evento."""
        self.keywords = {
            EventType.CULTURAL: [
                'festival', 'concierto', 'exposición', 'teatro', 'museo',
                'obra', 'artista', 'cultural', 'música', 'danza', 'ballet',
                'ópera', 'cine', 'película', 'documental', 'arte', 'galería',
                'espectáculo', 'actuación', 'performance'
            ],
            
            EventType.DEPORTIVO: [
                'fútbol', 'baloncesto', 'tenis', 'atletismo', 'natación',
                'partido', 'campeonato', 'torneo', 'liga', 'copa', 'mundial',
                'olímpico', 'deporte', 'equipo', 'jugador', 'entrenamiento',
                'competición', 'carrera', 'maratón', 'ciclismo', 'boxeo',
                'estadio', 'cancha', 'gol', 'victoria', 'derrota'
            ],
            
            EventType.METEOROLOGICO: [
                'tormenta', 'huracán', 'tornado', 'lluvia', 'nieve',
                'granizo', 'viento', 'temperatura', 'clima', 'meteorológico',
                'ciclón', 'frente', 'precipitaciones', 'sequía', 'inundación',
                'ola de calor', 'ola de frío', 'temporal', 'ventisca',
                'pronóstico', 'alerta meteorológica', 'tifón'
            ],
            
            EventType.POLITICO: [
                'elecciones', 'votación', 'parlamento', 'congreso', 'senado',
                'presidente', 'ministro', 'alcalde', 'gobierno', 'política',
                'ley', 'legislación', 'decreto', 'partido político',
                'campaña', 'debate', 'reforma', 'constitución', 'diputado',
                'referéndum', 'plebiscito', 'comicios'
            ],
            
            EventType.ECONOMICO: [
                'bolsa', 'mercado', 'economía', 'empresa', 'inversión',
                'financiero', 'banco', 'crédito', 'deuda', 'pib', 'inflación',
                'desempleo', 'comercio', 'exportación', 'importación',
                'industria', 'producción', 'ventas', 'precio', 'tarifa',
                'fusión', 'adquisición', 'quiebra', 'bursátil'
            ],
            
            EventType.SOCIAL: [
                'manifestación', 'protesta', 'huelga', 'marcha', 'movimiento social',
                'comunidad', 'sociedad', 'ciudadanos', 'vecinos', 'población',
                'organización', 'asociación', 'fundación', 'ong', 'voluntariado',
                'beneficencia', 'solidaridad', 'ayuda humanitaria', 'refugiados',
                'migrantes', 'derechos humanos'
            ],
            
            EventType.INCIDENTE: [
                'accidente', 'incidente', 'emergencia', 'desastre', 'catástrofe',
                'explosión', 'incendio', 'colapso', 'choque', 'colisión',
                'ataque', 'atentado', 'víctima', 'herido', 'fallecido',
                'rescate', 'evacuación', 'daños', 'destrucción', 'caos',
                'crisis', 'tragedia', 'siniestro'
            ],
            
            EventType.REGULACION: [
                'regulación', 'normativa', 'reglamento', 'disposición',
                'circular', 'resolución', 'ordenanza', 'estatuto',
                'restricción', 'prohibición', 'autorización', 'permiso',
                'licencia', 'certificación', 'homologación', 'acreditación',
                'inspección', 'control', 'fiscalización', 'sanción',
                'multa', 'cumplimiento', 'obligatorio'
            ]
        }
        
        # Normalizar todas las palabras clave a minúsculas
        for event_type in self.keywords:
            self.keywords[event_type] = [kw.lower() for kw in self.keywords[event_type]]
    
    def classify(self, text: str, threshold: float = 0.3) -> Tuple[EventType, float]:
        """
        Clasifica el texto en un tipo de evento.
        
        Args:
            text: Texto a clasificar
            threshold: Umbral mínimo de confianza para asignar un tipo específico
            
        Returns:
            Tupla (EventType, confidence) donde confidence es un valor entre 0 y 1
        """
        text_lower = text.lower()
        
        # Contar coincidencias para cada tipo de evento
        scores = self._calculate_scores(text_lower)
        
        # Si no hay coincidencias suficientes, devolver OTRO
        if not scores or max(scores.values()) < threshold:
            return EventType.OTRO, 0.5
        
        # Obtener el tipo con mayor puntuación
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = min(scores[best_type], 1.0)
        
        return best_type, confidence
    
    def classify_multiple(self, text: str, top_k: int = 3) -> List[Tuple[EventType, float]]:
        """
        Clasifica el texto y devuelve los top K tipos más probables.
        
        Args:
            text: Texto a clasificar
            top_k: Número de tipos a devolver
            
        Returns:
            Lista de tuplas (EventType, confidence) ordenadas por confianza
        """
        text_lower = text.lower()
        scores = self._calculate_scores(text_lower)
        
        if not scores:
            return [(EventType.OTRO, 0.5)]
        
        # Ordenar por puntuación y devolver top K
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Normalizar las puntuaciones para que sumen 1
        total = sum(score for _, score in sorted_types[:top_k])
        if total > 0:
            normalized = [(event_type, score / total) for event_type, score in sorted_types[:top_k]]
        else:
            normalized = [(event_type, score) for event_type, score in sorted_types[:top_k]]
        
        return normalized
    
    def _calculate_scores(self, text_lower: str) -> Dict[EventType, float]:
        """
        Calcula las puntuaciones para cada tipo de evento basándose en palabras clave.
        
        Args:
            text_lower: Texto en minúsculas
            
        Returns:
            Diccionario con las puntuaciones para cada tipo
        """
        scores = {}
        
        for event_type, keywords in self.keywords.items():
            # Contar cuántas palabras clave aparecen en el texto
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if matches > 0:
                # Normalizar por el número total de palabras clave
                score = matches / len(keywords)
                scores[event_type] = score
        
        return scores
    
    def add_keywords(self, event_type: EventType, keywords: List[str]):
        """
        Añade palabras clave adicionales para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            keywords: Lista de nuevas palabras clave
        """
        normalized_keywords = [kw.lower() for kw in keywords]
        self.keywords[event_type].extend(normalized_keywords)
    
    def get_keywords(self, event_type: EventType) -> List[str]:
        """
        Obtiene las palabras clave para un tipo de evento.
        
        Args:
            event_type: Tipo de evento
            
        Returns:
            Lista de palabras clave
        """
        return self.keywords.get(event_type, [])
