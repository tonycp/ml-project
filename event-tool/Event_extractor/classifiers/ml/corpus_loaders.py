"""
Cargadores de corpus para entrenamiento de clasificadores.
Abstracción que permite cambiar fácilmente entre diferentes corpus.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CorpusStats:
    """Estadísticas de un corpus."""
    name: str
    total_samples: int
    num_categories: int
    categories: List[str]
    language: str
    category_distribution: Dict[str, int]


class CorpusLoader(ABC):
    """Clase base para cargar corpus de noticias."""
    
    @abstractmethod
    def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Carga el corpus.
        
        Args:
            max_samples: Número máximo de muestras (None = todas)
            
        Returns:
            Tupla (texts, labels) con las noticias y sus categorías
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> CorpusStats:
        """Obtiene estadísticas del corpus."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Nombre del corpus."""
        pass


class AGNewsCorpus(CorpusLoader):
    """Cargador para AG News (127k noticias en inglés, 4 categorías)."""
    
    def __init__(self):
        self._texts = None
        self._labels = None
    
    def get_name(self) -> str:
        return "AG News"
    
    def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Carga AG News dataset."""
        try:
            from datasets import load_dataset
            
            print(f"📥 Descargando {self.get_name()}...")
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
                texts.append(item['text'])
                labels.append(label_names[item['label']])
            
            # Limitar si se especifica
            if max_samples and max_samples < len(texts):
                from random import Random
                rng = Random(42)
                indices = rng.sample(range(len(texts)), max_samples)
                texts = [texts[i] for i in indices]
                labels = [labels[i] for i in indices]
            
            self._texts = texts
            self._labels = labels
            
            print(f"✅ {self.get_name()} cargado: {len(texts)} noticias")
            return texts, labels
            
        except ImportError:
            raise ImportError("Se requiere: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Error cargando {self.get_name()}: {e}")
    
    def get_stats(self) -> CorpusStats:
        """Estadísticas de AG News."""
        if self._texts is None:
            self.load()
        
        from collections import Counter
        distribution = Counter(self._labels)
        
        return CorpusStats(
            name=self.get_name(),
            total_samples=len(self._texts),
            num_categories=len(set(self._labels)),
            categories=sorted(set(self._labels)),
            language="English",
            category_distribution=dict(distribution)
        )


class MLSUMCorpus(CorpusLoader):
    """Cargador para MLSUM Spanish (corpus grande de resúmenes en español)."""
    
    def __init__(self):
        self._texts = None
        self._labels = None
    
    def get_name(self) -> str:
        return "MLSUM Spanish"
    
    def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Carga MLSUM Spanish dataset - usa el texto del artículo como contenido."""
        try:
            from datasets import load_dataset
            
            print(f"📥 Descargando {self.get_name()} desde HuggingFace...")
            print("   ⏳ Este corpus es grande, puede tardar varios minutos...")
            
            # Cargar MLSUM español
            dataset = load_dataset("mlsum", "es", split="train", trust_remote_code=True)
            
            texts = []
            labels = []
            
            # Categorías inferidas del topic del texto (simplificado)
            for item in dataset:
                text = item.get('text', '').strip()
                if not text:
                    continue
                
                # MLSUM no tiene categorías explícitas, usamos el tema del artículo
                # Para simplificar, asignamos una categoría genérica
                texts.append(text)
                labels.append("General")  # MLSUM es un corpus de resúmenes, no clasificación
            
            # Limitar si se especifica
            if max_samples and max_samples < len(texts):
                from random import Random
                rng = Random(42)
                indices = rng.sample(range(len(texts)), max_samples)
                texts = [texts[i] for i in indices]
                labels = [labels[i] for i in indices]
            
            self._texts = texts
            self._labels = labels
            
            print(f"✅ {self.get_name()} cargado: {len(texts)} artículos")
            print(f"   ℹ️  Nota: MLSUM es un corpus de resúmenes, usa categoría 'General'")
            return texts, labels
            
        except ImportError:
            raise ImportError("Se requiere: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Error cargando {self.get_name()}: {e}")
    
    def get_stats(self) -> CorpusStats:
        """Estadísticas de MLDoc."""
        if self._texts is None:
            self.load()
        
        from collections import Counter
        distribution = Counter(self._labels)
        
        return CorpusStats(
            name=self.get_name(),
            total_samples=len(self._texts),
            num_categories=len(set(self._labels)),
            categories=sorted(set(self._labels)),
            language="Spanish",
            category_distribution=dict(distribution)
        )


class SpanishNewsCorpus(CorpusLoader):
    """Cargador para Spanish News (noticias en español con categorías)."""
    
    def __init__(self):
        self._texts = None
        self._labels = None
    
    def get_name(self) -> str:
        return "Spanish News"
    
    def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Carga Spanish News dataset desde MarcOrfilaCarreras."""
        try:
            from datasets import load_dataset
            
            print(f"📥 Descargando {self.get_name()}...")
            
            # Cargar Spanish News
            dataset = load_dataset("MarcOrfilaCarreras/spanish-news", split="train")
            
            texts = []
            labels = []
            
            # Extraer texto y categoría (los campos son: text, category, language, newspaper, hash)
            for item in dataset:
                text = item.get('text', '')
                category = item.get('category', 'Other')
                
                # Filtrar solo español
                if item.get('language') == 'es' and text.strip():
                    texts.append(text)
                    labels.append(category)
            
            # Limitar si se especifica
            if max_samples and max_samples < len(texts):
                from random import Random
                rng = Random(42)
                indices = rng.sample(range(len(texts)), max_samples)
                texts = [texts[i] for i in indices]
                labels = [labels[i] for i in indices]
            
            self._texts = texts
            self._labels = labels
            
            print(f"✅ {self.get_name()} cargado: {len(texts)} noticias")
            return texts, labels
            
        except ImportError:
            raise ImportError("Se requiere: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Error cargando {self.get_name()}: {e}")
    
    def get_stats(self) -> CorpusStats:
        """Estadísticas de Spanish News."""
        if self._texts is None:
            self.load()
        
        from collections import Counter
        distribution = Counter(self._labels)
        
        return CorpusStats(
            name=self.get_name(),
            total_samples=len(self._texts),
            num_categories=len(set(self._labels)),
            categories=sorted(set(self._labels)),
            language="Spanish",
            category_distribution=dict(distribution)
        )


class MLSUMCorpus(CorpusLoader):
    """Cargador para MLSUM Spanish (266k noticias en español, sin categorías)."""
    
    def __init__(self):
        self._texts = None
        self._labels = None
    
    def get_name(self) -> str:
        return "MLSUM Spanish"
    
    def load(self, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Carga MLSUM Spanish dataset.
        Nota: Este corpus NO tiene categorías predefinidas.
        Retorna todos con label 'News' para compatibilidad.
        """
        try:
            from datasets import load_dataset
            
            print(f"📥 Descargando {self.get_name()}...")
            print("⚠️  Advertencia: MLSUM no tiene categorías predefinidas")
            
            dataset = load_dataset("mlsum", "es", split="train")
            
            texts = []
            labels = []
            
            for item in dataset:
                texts.append(item['text'])
                labels.append('News')  # Categoría genérica
            
            # Limitar si se especifica
            if max_samples and max_samples < len(texts):
                from random import Random
                rng = Random(42)
                indices = rng.sample(range(len(texts)), max_samples)
                texts = [texts[i] for i in indices]
                labels = [labels[i] for i in indices]
            
            self._texts = texts
            self._labels = labels
            
            print(f"✅ {self.get_name()} cargado: {len(texts)} noticias")
            return texts, labels
            
        except ImportError:
            raise ImportError("Se requiere: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Error cargando {self.get_name()}: {e}")
    
    def get_stats(self) -> CorpusStats:
        """Estadísticas de MLSUM."""
        if self._texts is None:
            self.load()
        
        from collections import Counter
        distribution = Counter(self._labels)
        
        return CorpusStats(
            name=self.get_name(),
            total_samples=len(self._texts),
            num_categories=len(set(self._labels)),
            categories=sorted(set(self._labels)),
            language="Spanish",
            category_distribution=dict(distribution)
        )


# Registro de corpus disponibles
AVAILABLE_CORPUS = {
    'agnews': AGNewsCorpus,
    'spanish_news': SpanishNewsCorpus,
    'mlsum': MLSUMCorpus,
}


def get_corpus(name: str) -> CorpusLoader:
    """
    Obtiene un cargador de corpus por nombre.
    
    Args:
        name: Nombre del corpus ('agnews', 'spanish_news', 'mlsum')
        
    Returns:
        Instancia del cargador de corpus
    """
    name = name.lower()
    if name not in AVAILABLE_CORPUS:
        raise ValueError(
            f"Corpus '{name}' no disponible. "
            f"Opciones: {list(AVAILABLE_CORPUS.keys())}"
        )
    
    return AVAILABLE_CORPUS[name]()


def list_available_corpus() -> List[str]:
    """Lista los corpus disponibles."""
    return list(AVAILABLE_CORPUS.keys())
