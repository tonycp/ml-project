"""
Configuraciones de modelos de clasificación.
Permite cambiar fácilmente entre diferentes algoritmos y parámetros.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelConfig:
    """Configuración de un modelo de clasificación."""
    name: str
    classifier_class: type
    params: Dict[str, Any]
    description: str


# Configuraciones de modelos disponibles
MODEL_CONFIGS = {
    'naive_bayes': ModelConfig(
        name='Naive Bayes',
        classifier_class=MultinomialNB,
        params={'alpha': 0.1},
        description='Multinomial Naive Bayes - Rápido, bueno para texto'
    ),
    
    'naive_bayes_default': ModelConfig(
        name='Naive Bayes (alpha=1.0)',
        classifier_class=MultinomialNB,
        params={'alpha': 1.0},
        description='Naive Bayes con parámetros por defecto'
    ),
    
    'svm': ModelConfig(
        name='SVM Linear',
        classifier_class=LinearSVC,
        params={'C': 1.0, 'max_iter': 1000, 'random_state': 42},
        description='Support Vector Machine - Excelente para clasificación de texto'
    ),
    
    'svm_tuned': ModelConfig(
        name='SVM Optimizado',
        classifier_class=LinearSVC,
        params={'C': 0.5, 'max_iter': 2000, 'random_state': 42, 'class_weight': 'balanced'},
        description='SVM con parámetros optimizados'
    ),
    
    'logistic': ModelConfig(
        name='Logistic Regression',
        classifier_class=LogisticRegression,
        params={'C': 1.0, 'max_iter': 1000, 'random_state': 42, 'solver': 'lbfgs'},
        description='Regresión Logística - Balance velocidad/precisión'
    ),
    
    'logistic_l1': ModelConfig(
        name='Logistic Regression L1',
        classifier_class=LogisticRegression,
        params={'C': 1.0, 'max_iter': 1000, 'random_state': 42, 'penalty': 'l1', 'solver': 'liblinear'},
        description='Regresión Logística con regularización L1'
    ),
    
    'random_forest': ModelConfig(
        name='Random Forest',
        classifier_class=RandomForestClassifier,
        params={'n_estimators': 100, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1},
        description='Random Forest - Robusto pero más lento'
    ),
    
    'random_forest_deep': ModelConfig(
        name='Random Forest Profundo',
        classifier_class=RandomForestClassifier,
        params={'n_estimators': 200, 'max_depth': None, 'random_state': 42, 'n_jobs': -1},
        description='Random Forest sin límite de profundidad'
    ),
    
    'gradient_boosting': ModelConfig(
        name='Gradient Boosting',
        classifier_class=GradientBoostingClassifier,
        params={'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
        description='Gradient Boosting - Alta precisión, lento'
    ),
}


# Conjuntos predefinidos para comparación rápida
MODEL_SETS = {
    'fast': ['naive_bayes', 'logistic'],
    'balanced': ['naive_bayes', 'svm', 'logistic'],
    'complete': ['naive_bayes', 'svm', 'logistic', 'random_forest'],
    'all': list(MODEL_CONFIGS.keys()),
}


def get_model_config(name: str) -> ModelConfig:
    """
    Obtiene la configuración de un modelo.
    
    Args:
        name: Nombre del modelo
        
    Returns:
        Configuración del modelo
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Modelo '{name}' no disponible. "
            f"Opciones: {list(MODEL_CONFIGS.keys())}"
        )
    
    return MODEL_CONFIGS[name]


def list_available_models() -> List[str]:
    """Lista los modelos disponibles."""
    return list(MODEL_CONFIGS.keys())


def list_model_sets() -> Dict[str, List[str]]:
    """Lista los conjuntos de modelos predefinidos."""
    return MODEL_SETS


def get_model_configs(model_set: str = "balanced") -> List[ModelConfig]:
    """
    Obtiene una lista de configuraciones de modelos según el conjunto especificado.
    
    Args:
        model_set: Nombre del conjunto ('fast', 'balanced', 'complete', 'all')
        
    Returns:
        Lista de ModelConfig para el conjunto especificado
    """
    if model_set not in MODEL_SETS:
        raise ValueError(
            f"Conjunto '{model_set}' no válido. "
            f"Opciones: {list(MODEL_SETS.keys())}"
        )
    
    model_keys = MODEL_SETS[model_set]
    return [MODEL_CONFIGS[key] for key in model_keys]


def print_available_models():
    """Imprime información de todos los modelos disponibles."""
    print("=" * 70)
    print("MODELOS DISPONIBLES")
    print("=" * 70)
    
    for key, config in MODEL_CONFIGS.items():
        print(f"\n🔹 {key}")
        print(f"   Nombre: {config.name}")
        print(f"   Descripción: {config.description}")
        print(f"   Parámetros: {config.params}")
    
    print("\n" + "=" * 70)
    print("CONJUNTOS PREDEFINIDOS")
    print("=" * 70)
    
    for set_name, models in MODEL_SETS.items():
        print(f"\n🔸 {set_name}: {', '.join(models)}")
