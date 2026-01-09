"""
Clasificador de sentimiento usando sklearn con TF-IDF.
"""

from typing import Tuple, List, Optional, Dict, Any
import pickle
from pathlib import Path

from .base import SentimentClassifier
from ...models.event import EventSentiment


class SklearnSentimentClassifier(SentimentClassifier):
    """
    Clasificador de sentimiento usando TF-IDF + sklearn.
    
    Utiliza vectorización TF-IDF y modelos de sklearn (SVM, Naive Bayes, etc.)
    entrenados con corpus de sentiment como TASS.
    """
    
    def __init__(self, vectorizer=None, model=None, model_name: str = "SVM Linear"):
        """
        Inicializa el clasificador sklearn.
        
        Args:
            vectorizer: Vectorizador TF-IDF entrenado (sklearn TfidfVectorizer)
            model: Modelo sklearn entrenado
            model_name: Nombre descriptivo del modelo
        """
        self.vectorizer = vectorizer
        self.model = model
        self.model_name = model_name
        self._label_to_sentiment = self._create_label_mapping()
    
    def _create_label_mapping(self) -> Dict[Any, EventSentiment]:
        """
        Crea el mapeo de labels del modelo a EventSentiment.
        
        TASS usa las siguientes labels:
        - 'N' o 0: Negativo
        - 'NEU' o 1: Neutral
        - 'P' o 2: Positivo
        """
        return {
            # Labels de TASS (string)
            'N': EventSentiment.NEGATIVE,
            'NEU': EventSentiment.NEUTRAL,
            'P': EventSentiment.POSITIVE,
            'NONE': EventSentiment.NEUTRAL,
            # Labels numéricas
            0: EventSentiment.NEGATIVE,
            1: EventSentiment.NEUTRAL,
            2: EventSentiment.POSITIVE,
        }
    
    def classify(self, text: str, threshold: Optional[float] = None) -> Tuple[EventSentiment, float]:
        """
        Clasifica el sentimiento del texto.
        
        Args:
            text: Texto a clasificar
            threshold: Umbral mínimo de confianza (no usado, mantenido por compatibilidad)
            
        Returns:
            Tupla (EventSentiment, confidence)
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError(
                "El clasificador no ha sido entrenado. "
                "Llama a train() o carga un modelo pre-entrenado con load_model()"
            )
        
        if not text or not text.strip():
            return EventSentiment.NEUTRAL, 0.5
        
        try:
            # Vectorizar el texto
            X = self.vectorizer.transform([text])
            
            # Predecir clase
            y_pred = self.model.predict(X)[0]
            
            # Obtener probabilidades si el modelo las soporta
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)[0]
                confidence = float(max(probas))
            elif hasattr(self.model, 'decision_function'):
                # Para SVM usar decision function
                decision = self.model.decision_function(X)[0]
                # Si es array (multiclase), tomar el máximo
                if hasattr(decision, '__len__'):
                    confidence = float(max(abs(d) for d in decision) / (1 + max(abs(d) for d in decision)))
                else:
                    # Normalizar a [0, 1]
                    confidence = float(1 / (1 + abs(decision)))
            else:
                # Sin probabilidades, usar confianza fija
                confidence = 0.8
            
            # Mapear predicción a EventSentiment
            sentiment = self._label_to_sentiment.get(y_pred, EventSentiment.NEUTRAL)
            
            return sentiment, confidence
            
        except Exception as e:
            print(f"Error en clasificación: {str(e)}")
            return EventSentiment.NEUTRAL, 0.5
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[EventSentiment, float]]:
        """
        Clasifica múltiples textos de manera eficiente.
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de tuplas (EventSentiment, confidence)
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("El clasificador no ha sido entrenado")
        
        if not texts:
            return []
        
        try:
            # Vectorizar todos los textos
            X = self.vectorizer.transform(texts)
            
            # Predecir todas las clases
            y_pred = self.model.predict(X)
            
            # Obtener probabilidades si es posible
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)
                confidences = [float(max(proba)) for proba in probas]
            elif hasattr(self.model, 'decision_function'):
                decisions = self.model.decision_function(X)
                confidences = []
                for d in decisions:
                    if hasattr(d, '__len__'):
                        # Multiclase: tomar max
                        conf = float(max(abs(val) for val in d) / (1 + max(abs(val) for val in d)))
                    else:
                        # Binario
                        conf = float(1 / (1 + abs(d)))
                    confidences.append(conf)
            else:
                confidences = [0.8] * len(texts)
            
            # Mapear predicciones
            results = []
            for pred, conf in zip(y_pred, confidences):
                sentiment = self._label_to_sentiment.get(pred, EventSentiment.NEUTRAL)
                results.append((sentiment, conf))
            
            return results
            
        except Exception as e:
            print(f"Error en clasificación batch: {str(e)}")
            return [self.classify(text) for text in texts]
    
    def train(self, texts: List[str], labels: List[Any], 
              vectorizer_params: Optional[Dict] = None,
              model_params: Optional[Dict] = None):
        """
        Entrena el clasificador con un corpus.
        
        Args:
            texts: Lista de textos de entrenamiento
            labels: Lista de labels (puede ser 'N', 'NEU', 'P' o 0, 1, 2)
            vectorizer_params: Parámetros para TfidfVectorizer
            model_params: Parámetros para el modelo sklearn
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        
        # Configuración por defecto del vectorizador
        if vectorizer_params is None:
            vectorizer_params = {
                'max_features': 10000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95
            }
        
        # Crear y entrenar vectorizador
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        X = self.vectorizer.fit_transform(texts)
        
        # Configuración por defecto del modelo
        if model_params is None:
            model_params = {'max_iter': 1000, 'random_state': 42}
        
        # Crear y entrenar modelo
        self.model = LinearSVC(**model_params)
        self.model.fit(X, labels)
    
    def save_model(self, path: str):
        """
        Guarda el modelo entrenado en disco.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("No hay modelo entrenado para guardar")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'model_name': self.model_name,
            'label_mapping': self._label_to_sentiment
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, path: str) -> 'SklearnSentimentClassifier':
        """
        Carga un modelo entrenado desde disco.
        
        Args:
            path: Ruta del modelo guardado
            
        Returns:
            Instancia del clasificador con el modelo cargado
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(
            vectorizer=model_data['vectorizer'],
            model=model_data['model'],
            model_name=model_data.get('model_name', 'SVM Linear')
        )
        
        if 'label_mapping' in model_data:
            classifier._label_to_sentiment = model_data['label_mapping']
        
        return classifier
    
    def get_name(self) -> str:
        """Devuelve el nombre del clasificador."""
        return f"Sklearn: {self.model_name}"
    
    def get_description(self) -> str:
        """Devuelve una descripción del clasificador."""
        return f"Clasificador sklearn usando TF-IDF + {self.model_name}"
    
    def supports_training(self) -> bool:
        """Indica que este clasificador soporta entrenamiento."""
        return True
    
    def is_trained(self) -> bool:
        """
        Verifica si el clasificador ha sido entrenado.
        
        Returns:
            True si el modelo está entrenado y listo para usar
        """
        return self.vectorizer is not None and self.model is not None
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[EventSentiment, List[Tuple[str, float]]]:
        """
        Obtiene las características más importantes para cada clase.
        
        Args:
            top_n: Número de características a retornar por clase
            
        Returns:
            Diccionario con las top features por sentimiento
        """
        if not self.is_trained():
            return {}
        
        if not hasattr(self.model, 'coef_'):
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Obtener coeficientes por clase
        importance = {}
        for idx, coef in enumerate(self.model.coef_):
            # Obtener top features positivas
            top_indices = coef.argsort()[-top_n:][::-1]
            top_features = [(feature_names[i], float(coef[i])) for i in top_indices]
            
            # Mapear índice de clase a sentimiento
            sentiment = self._label_to_sentiment.get(idx, EventSentiment.NEUTRAL)
            importance[sentiment] = top_features
        
        return importance
