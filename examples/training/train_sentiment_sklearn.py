"""
Ejemplo de entrenamiento del clasificador sklearn con el corpus TASS.

Este script:
1. Carga el corpus TASS de sentiment analysis en español
2. Entrena un clasificador sklearn (TF-IDF + SVM)
3. Evalúa el modelo en el conjunto de test
4. Guarda el modelo entrenado
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier
from Event_extractor.models.event import EventSentiment


def main():
    print("🔬 ENTRENAMIENTO DE CLASIFICADOR SKLEARN CON TASS")
    print("=" * 70)
    
    # 1. Cargar corpus TASS-2019
    print("\n📊 Cargando corpus TASS...")
    try:
        ds_train = load_dataset("mrm8488/tass-2019", split="train")
        print(f"   ✅ Dataset cargado: {len(ds_train):,} muestras")
    except Exception as e:
        print(f"   ❌ Error cargando TASS: {e}")
        return
    
    # 2. Preparar y dividir datos
    print("\n📋 Preparando datos...")
    
    # Extraer textos y labels
    all_texts = list(ds_train['sentence'])
    all_labels = list(ds_train['sentiments'])  # 'N' (negativo), 'NEU' (neutral), 'P' (positivo)
    
    # Filtrar ejemplos sin label válida (NONE o None)
    valid_indices = [i for i, label in enumerate(all_labels) 
                    if label is not None and label != 'NONE']
    all_texts = [all_texts[i] for i in valid_indices]
    all_labels = [all_labels[i] for i in valid_indices]
    
    print(f"   ✅ Ejemplos válidos: {len(all_texts)}")
    
    # Dividir en train/test (80/20) manteniendo distribución de clases
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, 
        test_size=0.2, 
        random_state=42,
        stratify=all_labels
    )
    
    print(f"   📝 Train: {len(train_texts)} | Test: {len(test_texts)}")
    
    # Mostrar distribución de clases
    from collections import Counter
    train_dist = Counter(train_labels)
    print(f"\n   📊 Distribución de clases (train):")
    for label, count in train_dist.items():
        pct = (count / len(train_labels)) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    # 3. Entrenar clasificador
    print("\n🎓 Entrenando clasificador sklearn...")
    print("   Modelo: LinearSVC")
    print("   Features: TF-IDF (max 10k features, bigrams)")
    
    classifier = SklearnSentimentClassifier(model_name="SVM Linear TASS")
    
    # Configuración para español
    vectorizer_params = {
        'max_features': 10000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95
    }
    
    model_params = {
        'max_iter': 1000,
        'random_state': 42
    }
    
    classifier.train(
        texts=train_texts,
        labels=train_labels,
        vectorizer_params=vectorizer_params,
        model_params=model_params
    )
    
    print("   ✅ Entrenamiento completado")
    
    # 4. Evaluar en conjunto de test
    print("\n📈 Evaluando en conjunto de test...")
    
    predictions = []
    confidences = []
    
    # Clasificar cada texto del test set
    for text in test_texts:
        sentiment, confidence = classifier.classify(text)
        
        # Mapear EventSentiment de vuelta a labels TASS ('N', 'NEU', 'P')
        if sentiment == EventSentiment.NEGATIVE:
            pred = 'N'
        elif sentiment == EventSentiment.NEUTRAL:
            pred = 'NEU'
        elif sentiment == EventSentiment.POSITIVE:
            pred = 'P'
        else:
            pred = 'NEU'  # Fallback
        
        predictions.append(pred)
        confidences.append(confidence)
    
    # Calcular métricas
    accuracy = accuracy_score(test_labels, predictions)
    
    # Mostrar resultados
    avg_confidence = sum(confidences) / len(confidences)
    
    print(f"\n   📊 Resultados:")
    print(f"      • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"      • Confianza promedio: {avg_confidence:.4f}")
    
    print(f"\n   📋 Reporte detallado por clase:")
    print(classification_report(test_labels, predictions, 
                               target_names=['Negativo', 'Neutral', 'Positivo'],
                               digits=4))
    
    # 5. Probar con ejemplos reales
    print("\n" + "=" * 70)
    print("🧪 PRUEBA CON EJEMPLOS")
    print("=" * 70)
    
    ejemplos = [
        "El festival de música fue un éxito rotundo con miles de asistentes felices",
        "Cancelan el concierto debido a problemas técnicos, gran decepción",
        "Se anuncia la fecha de la conferencia de tecnología",
        "Terrible accidente en la autopista deja varios heridos",
        "Inauguran nuevo parque con juegos para niños"
    ]
    
    for texto in ejemplos:
        sentiment, confidence = classifier.classify(texto)
        emoji = "😊" if sentiment == EventSentiment.POSITIVE else "😐" if sentiment == EventSentiment.NEUTRAL else "😞"
        print(f"\n   {emoji} '{texto[:65]}...'")
        print(f"      → {sentiment.value} ({confidence:.2%})")
    
    # 6. Guardar modelo entrenado
    print("\n" + "=" * 70)
    print("💾 GUARDANDO MODELO")
    print("=" * 70)
    
    model_path = "models/sklearn_tass_sentiment.pkl"
    try:
        Path("models").mkdir(exist_ok=True)
        classifier.save_model(model_path)
        print(f"\n   ✅ Modelo guardado exitosamente")
        print(f"   📁 Ubicación: {model_path}")
    except Exception as e:
        print(f"\n   ❌ Error guardando modelo: {e}")
    
    # 7. Analizar features más importantes
    print("\n" + "=" * 70)
    print("🔍 PALABRAS MÁS IMPORTANTES POR SENTIMIENTO")
    print("=" * 70)
    
    try:
        importance = classifier.get_feature_importance(top_n=10)
        
        for sentiment, features in importance.items():
            emoji = "😊" if sentiment == EventSentiment.POSITIVE else "😐" if sentiment == EventSentiment.NEUTRAL else "😞"
            print(f"\n   {emoji} {sentiment.value.upper()}:")
            for i, (word, score) in enumerate(features[:10], 1):
                print(f"      {i:2d}. {word:20s} ({score:+.4f})")
    except Exception as e:
        print(f"\n   ⚠️  No se pudieron extraer features: {e}")
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
