"""
Clasificador de tipos de noticias usando TF-IDF + clasificadores sklearn.
Entrenado con diferentes corpus de noticias (AG News, Spanish News, etc.).
Implementa la interfaz NewsTypeClassifier.
"""

import os
import pickle
from typing import Tuple, List, Optional
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from ...models.event import EventType
from ...utils.text_preprocessor import _tokenize_text, tokenize_texts_batch
from .base import NewsTypeClassifier

# Importar ModelConfig desde ml/
try:
    from ..ml.model_configs import ModelConfig
except ImportError:
    ModelConfig = None


class SklearnNewsClassifier(NewsTypeClassifier):
    """
    Clasificador de noticias basado en TF-IDF y clasificadores sklearn.
    
    Utiliza:
    - TF-IDF para vectorización de texto
    - Clasificador configurable (Naive Bayes, SVM, Logistic, RF, etc.)
    - Preprocesamiento con SpaCy para lematización y filtrado
    - Corpus: AG News, Spanish News, MLSUM, etc.
    
    Hereda de NewsTypeClassifier para ser compatible con el pipeline.
    """
    
    # Mapeo de categorías del dataset a EventType
    CATEGORY_MAPPING = {
        'World': EventType.POLITICO,
        'Sports': EventType.DEPORTIVO,
        'Business': EventType.ECONOMICO,
        'Sci/Tech': EventType.OTRO,
        'Entertainment': EventType.CULTURAL,
    }
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        model_config: Optional['ModelConfig'] = None,
        use_spacy_tokenizer: bool = True
    ):
        """
        Inicializa el clasificador.
        
        Args:
            model_path: Ruta al modelo guardado. Si None, crea uno nuevo sin entrenar.
            model_config: Configuración del modelo (classifier_class y params).
                         Si None, usa Naive Bayes por defecto.
            use_spacy_tokenizer: Si True, usa tokenizador de SpaCy (más lento pero mejor).
                                Si False, usa tokenizador estándar de sklearn (más rápido).
        """
        self.model_config = model_config
        self.use_spacy_tokenizer = use_spacy_tokenizer
        self.pipeline = None
        self.label_to_event_type = {}
        self.event_type_to_label = {}
        self._tokenized_cache = None  # Cache para tokens pre-procesados
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Inicializa el pipeline de TF-IDF + clasificador."""
        # Determinar clasificador a usar
        if self.model_config:
            classifier = self.model_config.classifier_class(**self.model_config.params)
        else:
            # Por defecto: Naive Bayes
            classifier = MultinomialNB(alpha=0.1)
        
        # Configurar TF-IDF
        # Cuando usamos SpaCy, pre-tokenizamos y pasamos texto ya procesado
        # Por lo tanto, TF-IDF solo necesita dividir por espacios
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=False if self.use_spacy_tokenizer else True,
            strip_accents=None if self.use_spacy_tokenizer else 'unicode'
        )
        
        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', classifier)
        ])
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesa el texto usando text_preprocessor.
        
        Args:
            text: Texto a preprocesar
            
        Returns:
            Lista de tokens lematizados y limpios
        """
        if isinstance(text, list):
            return text
        return _tokenize_text(text)
    
    def train_from_dataset(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """
        Entrena el clasificador con un dataset.
        
        Args:
            texts: Lista de textos de noticias
            labels: Lista de etiquetas/categorías
            test_size: Proporción del conjunto de test
            random_state: Semilla para reproducibilidad
            
        Returns:
            Diccionario con métricas de evaluación
        """
        # Crear mapeos entre labels y EventTypes
        unique_labels = sorted(set(labels))
        self.label_to_event_type = {
            label: self.CATEGORY_MAPPING.get(label, EventType.OTRO)
            for label in unique_labels
        }
        self.event_type_to_label = {
            v: k for k, v in self.label_to_event_type.items()
        }
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print(f"Entrenando con {len(X_train)} noticias...")
        print(f"Categorías: {unique_labels}")
        
        # Si usamos SpaCy, pre-tokenizar en lote para mayor eficiencia
        if self.use_spacy_tokenizer:
            print("Pre-tokenizando textos con SpaCy (esto puede tardar)...")
            from tqdm import tqdm
            
            # Pre-tokenizar train y test en lotes
            train_tokens = tokenize_texts_batch(X_train, batch_size=100)
            test_tokens = tokenize_texts_batch(X_test, batch_size=100)
            
            # Convertir tokens de vuelta a strings para TF-IDF
            X_train_processed = [' '.join(tokens) for tokens in train_tokens]
            X_test_processed = [' '.join(tokens) for tokens in test_tokens]
            
            print(f"✅ Tokenización completada")
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Entrenar el pipeline
        self.pipeline.fit(X_train_processed, y_train)
        
        # Evaluar
        y_pred = self.pipeline.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=unique_labels)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': report,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def classify(self, text: str, threshold: float = 0.3) -> Tuple[EventType, float]:
        """
        Clasifica un texto en un tipo de evento.
        
        Args:
            text: Texto a clasificar
            threshold: Umbral mínimo de confianza (no usado en este modelo)
            
        Returns:
            Tupla (EventType, confidence)
        """
        if self.pipeline is None:
            raise ValueError("El modelo no está entrenado. Usa train_from_dataset() primero.")
        
        # Obtener predicción
        predicted_label = self.pipeline.predict([text])[0]
        classes = self.pipeline.classes_
        
        # Intentar obtener confianza con probabilidades o decision_function
        if hasattr(self.pipeline, 'predict_proba'):
            # Modelos con predict_proba (LogisticRegression, MultinomialNB, etc.)
            probs = self.pipeline.predict_proba([text])[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx]
        elif hasattr(self.pipeline, 'decision_function'):
            # Modelos con decision_function (LinearSVC, SVC sin probability=True)
            decision_scores = self.pipeline.decision_function([text])[0]
            
            # Convertir decision scores a algo similar a confianza (0-1)
            # Para multiclase, usar softmax-like transformation
            if len(decision_scores.shape) == 0 or len(classes) == 2:
                # Binario: convertir a probabilidad con sigmoid aproximado
                confidence = 1.0 / (1.0 + np.exp(-abs(decision_scores)))
            else:
                # Multiclase: usar softmax para normalizar
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probs = exp_scores / exp_scores.sum()
                max_idx = np.argmax(probs)
                confidence = probs[max_idx]
        else:
            # Fallback: usar confianza fija moderada
            confidence = 0.6
        
        # Mapear a EventType
        event_type = self.label_to_event_type.get(predicted_label, EventType.OTRO)
        
        return event_type, float(confidence)
    
    def classify_multiple(self, text: str, top_k: int = 3) -> List[Tuple[EventType, float]]:
        """
        Clasifica el texto y devuelve los top K tipos más probables.
        
        Args:
            text: Texto a clasificar
            top_k: Número de tipos a devolver
            
        Returns:
            Lista de tuplas (EventType, confidence) ordenadas por confianza
        """
        if self.pipeline is None:
            raise ValueError("El modelo no está entrenado.")
        
        classes = self.pipeline.classes_
        
        # Obtener scores/probabilidades según el modelo
        if hasattr(self.pipeline, 'predict_proba'):
            probs = self.pipeline.predict_proba([text])[0]
        elif hasattr(self.pipeline, 'decision_function'):
            decision_scores = self.pipeline.decision_function([text])[0]
            # Convertir a pseudo-probabilidades con softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probs = exp_scores / exp_scores.sum()
        else:
            # Fallback: solo devolver la predicción principal
            predicted_label = self.pipeline.predict([text])[0]
            event_type = self.label_to_event_type.get(predicted_label, EventType.OTRO)
            return [(event_type, 0.6)]
        
        # Ordenar por probabilidad
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            label = classes[idx]
            confidence = probs[idx]
            event_type = self.label_to_event_type.get(label, EventType.OTRO)
            results.append((event_type, float(confidence)))
        
        return results
    
    def save_model(self, path: str):
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'label_to_event_type': self.label_to_event_type,
            'event_type_to_label': self.event_type_to_label
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en: {path}")
    
    def load_model(self, path: str):
        """
        Carga un modelo entrenado.
        
        Args:
            path: Ruta del modelo guardado
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.label_to_event_type = model_data['label_to_event_type']
        self.event_type_to_label = model_data['event_type_to_label']
        
        print(f"Modelo cargado desde: {path}")
    
    # ========================================================================
    # Métodos de la interfaz NewsTypeClassifier
    # ========================================================================
    
    def get_name(self) -> str:
        """Nombre del clasificador."""
        if self.model_config:
            return f"Sklearn: {self.model_config.name}"
        return "Sklearn Classifier (default)"
    
    def get_description(self) -> str:
        """Descripción del clasificador."""
        desc = "Clasificador basado en TF-IDF + sklearn"
        if self.model_config:
            desc += f" ({self.model_config.description})"
        if self.use_spacy_tokenizer:
            desc += " con tokenización SpaCy"
        return desc
    
    def supports_training(self) -> bool:
        """Este clasificador soporta entrenamiento."""
        return True


def download_and_prepare_agnews_spanish():
    """
    Descarga y prepara el corpus AG News en español.
    
    AG News es el corpus más grande públicamente disponible de noticias 
    clasificadas en español (~127k noticias en 4 categorías).
    
    Returns:
        Tupla (texts, labels) con las noticias y sus categorías
    """
    try:
        from datasets import load_dataset
        
        print("Descargando AG News en español...")
        # Usamos el dataset traducido al español
        dataset = load_dataset("fancyzhx/ag_news", split="train")
        
        texts = []
        labels = []
        
        # Mapeo de índices a categorías
        label_names = {
            0: "World",
            1: "Sports", 
            2: "Business",
            3: "Sci/Tech"
        }
        
        for item in dataset:
            # Combinar título y descripción
            text = f"{item['text']}"
            texts.append(text)
            labels.append(label_names[item['label']])
        
        print(f"Dataset cargado: {len(texts)} noticias")
        return texts, labels
        
    except ImportError:
        print("Error: Se requiere la librería 'datasets'")
        print("Instala con: pip install datasets")
        return None, None
    except Exception as e:
        print(f"Error al descargar el dataset: {e}")
        return None, None


def train_news_classifier(
    output_path: str = "models/sklearn_news_classifier.pkl",
    sample_size: Optional[int] = None
) -> SklearnNewsClassifier:
    """
    Entrena un nuevo clasificador con AG News.
    
    Args:
        output_path: Ruta donde guardar el modelo
        sample_size: Tamaño de muestra (None = usar todo el dataset)
        
    Returns:
        Clasificador entrenado
    """
    texts, labels = download_and_prepare_agnews_spanish()
    
    if texts is None:
        raise ValueError("No se pudo cargar el dataset")
    
    # Muestrear si se especifica
    if sample_size and sample_size < len(texts):
        from random import Random
        rng = Random(42)
        indices = rng.sample(range(len(texts)), sample_size)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # Crear y entrenar clasificador
    classifier = SklearnNewsClassifier()
    classifier.train_from_dataset(texts, labels)
    
    # Guardar modelo
    classifier.save_model(output_path)
    
    return classifier

