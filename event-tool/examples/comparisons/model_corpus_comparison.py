"""
Comparación de Modelos de Clasificación en Múltiples Corpus.

Este script ejecuta una comparación completa de diferentes modelos de clasificación
(Naive Bayes, SVM, Logistic Regression) en dos corpus de noticias:
- AG News (inglés, 120k muestras, 4 categorías)
- Spanish News (español, 10k muestras, 12 categorías)

Características:
- Pre-tokenización eficiente: Tokeniza cada corpus UNA SOLA VEZ con SpaCy
- Los tokens se reutilizan para todos los modelos (ahorro masivo de tiempo)
- Lematización y filtrado de stop words para mejor calidad
- Métricas detalladas (accuracy, F1, precision, recall)
- Guarda resultados en JSON para análisis posterior
- Muestra tabla comparativa y mejores modelos por corpus

Optimización:
    El script tokeniza cada corpus una sola vez al inicio y reutiliza esos tokens
    para todos los modelos. Esto reduce drásticamente el tiempo total:
    - Sin optimización: tokeniza N veces (una por modelo)
    - Con optimización: tokeniza 1 vez (se comparte entre todos)
    
    Ejemplo con 5000 muestras y 3 modelos:
    - Antes: ~600s (200s × 3 modelos)
    - Ahora: ~200s + 3×5s = ~215s (ahorro de ~65%)

Uso:
    python examples/model_corpus_comparison.py

El script genera:
- Tabla comparativa en consola
- Archivo comparison_results.json con resultados detallados
- Identificación del mejor modelo para cada corpus
"""

import sys
import os
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.Event_extractor.classifiers.ml.corpus_loaders import get_corpus
from src.Event_extractor.classifiers.news_type import SklearnNewsClassifier
from src.Event_extractor.classifiers.ml.model_configs import get_model_configs
from src.Event_extractor.utils.text_preprocessor import tokenize_texts_batch
import time
import json


def compare_corpus_with_models(
    corpus_name: str,
    max_samples: int = 5000,
    model_set: str = "balanced",
    use_spacy: bool = True
):
    """
    Compara un corpus con múltiples modelos de clasificación.
    
    Args:
        corpus_name: Nombre del corpus ('agnews' o 'spanish_news')
        max_samples: Número máximo de muestras a usar
        model_set: Set de modelos a probar ('fast', 'balanced', 'complete', 'all')
        use_spacy: Si True, usa tokenizador SpaCy (se tokeniza una sola vez y se reutiliza)
    """
    print(f"\n{'='*80}")
    print(f"🎯 COMPARANDO CORPUS: {corpus_name.upper()}")
    print(f"{'='*80}\n")
    
    # Cargar corpus
    print(f"📥 Cargando corpus '{corpus_name}'...")
    corpus_loader = get_corpus(corpus_name)
    texts, labels = corpus_loader.load(max_samples=max_samples)
    
    stats = corpus_loader.get_stats()
    print(f"\n📊 Estadísticas del corpus:")
    print(f"   • Total: {stats.total_samples} muestras")
    print(f"   • Categorías: {stats.num_categories}")
    print(f"   • Idioma: {stats.language}")
    print(f"   • Distribución: {stats.category_distribution}")
    
    # PRE-TOKENIZACIÓN: Tokenizar UNA SOLA VEZ si usamos SpaCy
    texts_to_use = texts
    tokenization_time = 0
    
    if use_spacy:
        print(f"\n🔄 Pre-tokenizando corpus con SpaCy (una sola vez para todos los modelos)...")
        print(f"   ⏳ Procesando {len(texts)} textos...")
        
        start_time = time.time()
        # Tokenizar en lotes para eficiencia
        tokens_list = tokenize_texts_batch(texts, batch_size=100)
        # Convertir listas de tokens a strings (separados por espacio)
        texts_to_use = [' '.join(tokens) for tokens in tokens_list]
        tokenization_time = time.time() - start_time
        
        print(f"   ✅ Tokenización completada en {tokenization_time:.2f}s")
        print(f"   ℹ️  Los textos tokenizados se reutilizarán para todos los modelos")
    
    # Obtener modelos a probar
    model_configs = get_model_configs(model_set)
    print(f"\n🔧 Modelos a probar: {len(model_configs)}")
    for config in model_configs:
        print(f"   • {config.name}")
    
    # Resultados
    results = []
    
    # Probar cada modelo
    for i, config in enumerate(model_configs, 1):
        print(f"\n{'─'*80}")
        print(f"[{i}/{len(model_configs)}] 🧪 Probando: {config.name}")
        print(f"{'─'*80}")
        
        try:
            start_time = time.time()
            
            # Crear y entrenar clasificador
            # use_spacy_tokenizer=False porque ya pre-tokenizamos los textos
            classifier = SklearnNewsClassifier(
                model_config=config,
                use_spacy_tokenizer=False  # Ya están pre-tokenizados
            )
            
            # Pasar los textos ya tokenizados (no se volverán a tokenizar)
            metrics = classifier.train_from_dataset(
                texts_to_use,  # Textos originales o pre-tokenizados
                labels,
                test_size=0.2,
                random_state=42
            )
            
            training_time = time.time() - start_time
            
            # Mostrar resultados
            print(f"\n✅ Resultados:")
            print(f"   • Precisión: {metrics['accuracy']:.4f}")
            print(f"   • F1-Score: {metrics['f1_score']:.4f}")
            print(f"   • Tiempo entrenamiento: {training_time:.2f}s")
            if use_spacy and i == 1:  # Mostrar tiempo de tokenización solo en el primero
                print(f"   • Tiempo tokenización (compartido): {tokenization_time:.2f}s")
            
            # Guardar resultado
            results.append({
                'corpus': corpus_name,
                'model': config.name,
                'classifier': config.classifier_class.__name__,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'training_time': training_time
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'corpus': corpus_name,
                'model': config.name,
                'classifier': config.classifier_class.__name__,
                'error': str(e)
            })
    
    return results


def main():
    """Ejecuta comparaciones en ambos corpus."""
    
    print("\n" + "="*80)
    print("🚀 COMPARACIÓN DE CORPUS: AG NEWS vs SPANISH NEWS")
    print("="*80)
    
    # Configuración
    corpus_list = ['agnews', 'spanish_news']
    max_samples = 5000  # Limitar para que sea rápido
    model_set = 'all'  # Probar todos los modelos
    
    all_results = []
    
    # Ejecutar comparaciones para cada corpus
    for corpus_name in corpus_list:
        try:
            results = compare_corpus_with_models(
                corpus_name=corpus_name,
                max_samples=max_samples,
                model_set=model_set
            )
            all_results.extend(results)
            
        except Exception as e:
            print(f"\n❌ Error en corpus '{corpus_name}': {e}")
            continue
    
    # Resumen final
    print(f"\n{'='*80}")
    print("📈 RESUMEN COMPARATIVO")
    print(f"{'='*80}\n")
    
    # Agrupar por corpus
    by_corpus = {}
    for result in all_results:
        if 'error' in result:
            continue
        corpus = result['corpus']
        if corpus not in by_corpus:
            by_corpus[corpus] = []
        by_corpus[corpus].append(result)
    
    # Mostrar tabla comparativa
    print(f"{'Corpus':<20} {'Modelo':<25} {'Accuracy':<12} {'F1-Score':<12} {'Tiempo (s)':<12}")
    print("─" * 80)
    
    for corpus, results in by_corpus.items():
        # Ordenar por accuracy
        results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        
        for i, result in enumerate(results_sorted):
            corpus_label = corpus if i == 0 else ""
            print(
                f"{corpus_label:<20} "
                f"{result['model']:<25} "
                f"{result['accuracy']:.4f}      "
                f"{result['f1_score']:.4f}      "
                f"{result['training_time']:.2f}"
            )
        
        if corpus != list(by_corpus.keys())[-1]:
            print("─" * 80)
    
    # Guardar resultados en JSON
    output_file = project_root / "comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {output_file}")
    
    # Mejor modelo por corpus
    print(f"\n{'='*80}")
    print("🏆 MEJORES MODELOS POR CORPUS")
    print(f"{'='*80}\n")
    
    for corpus, results in by_corpus.items():
        if not results:
            continue
        
        best = max(results, key=lambda x: x['accuracy'])
        print(f"📊 {corpus.upper()}:")
        print(f"   🥇 Mejor modelo: {best['model']}")
        print(f"   • Accuracy: {best['accuracy']:.4f}")
        print(f"   • F1-Score: {best['f1_score']:.4f}")
        print(f"   • Tiempo: {best['training_time']:.2f}s\n")


if __name__ == "__main__":
    main()
