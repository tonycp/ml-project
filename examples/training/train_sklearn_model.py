"""
Script para entrenar el clasificador sklearn con Spanish News + SVM Linear.
Este modelo entrenado será usado en el pipeline de extracción de eventos.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Event_extractor.classifiers.news_type import SklearnNewsClassifier
from Event_extractor.classifiers.ml.model_configs import get_model_config
from Event_extractor.classifiers.ml.corpus_loaders import SpanishNewsCorpus

def train_and_save_model():
    """Entrena y guarda el modelo SVM Linear con Spanish News."""
    print("=" * 80)
    print("ENTRENANDO MODELO PARA EL PIPELINE")
    print("=" * 80)
    print()
    
    # Cargar corpus Spanish News
    print("📚 Cargando corpus Spanish News...")
    corpus = SpanishNewsCorpus()
    texts, labels = corpus.load(max_samples=None)
    stats = corpus.get_stats()
    print(f"   ✅ Cargado: {stats.total_samples} noticias")
    print(f"   📊 Categorías: {stats.categories}")
    print()
    
    # Obtener configuración del modelo SVM Linear
    print("🔧 Configurando modelo SVM Linear...")
    svm_config = get_model_config('svm')
    print(f"   ✅ Modelo: {svm_config.name}")
    print()
    
    # Crear clasificador con SpaCy tokenizer
    print("🤖 Creando clasificador...")
    classifier = SklearnNewsClassifier(
        model_config=svm_config,
        use_spacy_tokenizer=True  # Usar SpaCy para mejor rendimiento
    )
    print("   ✅ Clasificador creado")
    print()
    
    # Entrenar el modelo
    print("🎓 Entrenando modelo...")
    print("   (Esto puede tardar varios minutos con SpaCy tokenization)")
    print()
    results = classifier.train_from_dataset(
        texts=texts,
        labels=labels,
        test_size=0.2,
        random_state=42
    )
    print()
    
    # Mostrar resultados
    print("📊 RESULTADOS DEL ENTRENAMIENTO:")
    print("-" * 80)
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print()
    
    # Guardar el modelo
    model_path = project_root / "models" / "sklearn_spanish_svm.pkl"
    model_path.parent.mkdir(exist_ok=True)
    
    print(f"💾 Guardando modelo en: {model_path}")
    classifier.save_model(str(model_path))
    print("   ✅ Modelo guardado exitosamente")
    print()
    
    print("=" * 80)
    print("✅ MODELO ENTRENADO Y LISTO PARA USAR EN EL PIPELINE")
    print("=" * 80)
    print()
    print("Para usar el modelo en el pipeline:")
    print("   from Event_extractor.pipeline.event_pipeline import EventExtractionPipeline")
    print("   pipeline = EventExtractionPipeline(use_sklearn_classifier=True)")
    print()
    
    return str(model_path)

if __name__ == "__main__":
    model_path = train_and_save_model()
