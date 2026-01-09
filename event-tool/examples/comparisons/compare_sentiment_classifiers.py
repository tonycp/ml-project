"""
Ejemplo de comparación de clasificadores de sentimiento.

Compara los diferentes clasificadores disponibles:
1. KeywordSentimentClassifier (basado en palabras clave)
2. HuggingFaceSentimentClassifier (MarIA/RoBERTa)
3. SklearnSentimentClassifier (TF-IDF + SVM con TASS)
"""

import sys
import os
from pathlib import Path

# Agregar directorio raíz al path (dos niveles arriba desde examples/ml_classification/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.Event_extractor.classifiers.sentiment import (
    KeywordSentimentClassifier,
    MarIASentimentClassifier,
    SklearnSentimentClassifier
)


def main():
    print("🔬 COMPARACIÓN DE CLASIFICADORES DE SENTIMIENTO")
    print("=" * 70)
    
    # Ejemplos de texto en español
    ejemplos = [
        {
            "texto": "El festival de música fue un éxito rotundo con miles de asistentes felices y emocionados",
            "esperado": "POSITIVE"
        },
        {
            "texto": "Cancelan el concierto debido a problemas técnicos, gran decepción entre los fans",
            "esperado": "NEGATIVE"
        },
        {
            "texto": "Se anuncia la fecha de la conferencia de tecnología para el próximo mes",
            "esperado": "NEUTRAL"
        },
        {
            "texto": "Terrible accidente en la autopista deja varios heridos y causa retrasos",
            "esperado": "NEGATIVE"
        },
        {
            "texto": "Inauguran nuevo parque con juegos para niños y áreas verdes",
            "esperado": "POSITIVE"
        },
        {
            "texto": "La reunión del comité se llevará a cabo el jueves",
            "esperado": "NEUTRAL"
        },
        {
            "texto": "Gran victoria del equipo local con gol en el último minuto",
            "esperado": "POSITIVE"
        },
        {
            "texto": "Protesta masiva contra las nuevas medidas económicas",
            "esperado": "NEGATIVE"
        }
    ]
    
    # Inicializar clasificadores
    print("\n🔧 Inicializando clasificadores...\n")
    
    clasificadores = []
    
    # 1. Keyword classifier
    try:
        keyword_clf = KeywordSentimentClassifier()
        clasificadores.append(("Keyword-based", keyword_clf))
        print("   ✅ KeywordSentimentClassifier")
    except Exception as e:
        print(f"   ❌ KeywordSentimentClassifier: {e}")
    
    # 2. MarIA/RoBERTa (requiere transformers)
    try:
        maria_clf = MarIASentimentClassifier()
        clasificadores.append(("MarIA RoBERTa", maria_clf))
        print("   ✅ MarIASentimentClassifier")
    except Exception as e:
        print(f"   ⚠️  MarIASentimentClassifier no disponible: {str(e)[:50]}")
    
    # 3. Sklearn (requiere modelo entrenado)
    try:
        sklearn_clf = SklearnSentimentClassifier.load_model("models/sklearn_tass_sentiment.pkl")
        clasificadores.append(("Sklearn (TASS)", sklearn_clf))
        print("   ✅ SklearnSentimentClassifier")
    except Exception as e:
        print(f"   ⚠️  SklearnSentimentClassifier no disponible: {str(e)[:50]}")
    
    if not clasificadores:
        print("\n❌ No hay clasificadores disponibles")
        return
    
    # Comparar clasificadores
    print("\n" + "=" * 70)
    print("📊 RESULTADOS DE CLASIFICACIÓN")
    print("=" * 70)
    
    aciertos_por_clf = {nombre: 0 for nombre, _ in clasificadores}
    
    for idx, ejemplo in enumerate(ejemplos, 1):
        texto = ejemplo["texto"]
        esperado = ejemplo["esperado"]
        
        print(f"\n📝 Ejemplo {idx}: {texto[:60]}...")
        print(f"   Sentimiento esperado: {esperado}")
        print(f"   Clasificaciones:")
        
        for nombre, clf in clasificadores:
            try:
                sentiment, confidence = clf.classify(texto)
                # Normalizar comparación: sentiment.value es lowercase, esperado es UPPERCASE
                correcto = "✅" if sentiment.value.upper() == esperado else "❌"
                
                if sentiment.value.upper() == esperado:
                    aciertos_por_clf[nombre] += 1
                
                print(f"      {correcto} {nombre:20s}: {sentiment.value:8s} (conf: {confidence:.2%})")
            except Exception as e:
                print(f"      ❌ {nombre:20s}: Error - {str(e)[:30]}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("📈 RESUMEN DE PRECISIÓN")
    print("=" * 70)
    
    total_ejemplos = len(ejemplos)
    
    for nombre, aciertos in aciertos_por_clf.items():
        precision = (aciertos / total_ejemplos) * 100
        print(f"\n   {nombre}:")
        print(f"      Aciertos: {aciertos}/{total_ejemplos}")
        print(f"      Precisión: {precision:.2f}%")
    
    # Comparación de velocidad (opcional)
    print("\n" + "=" * 70)
    print("⚡ COMPARACIÓN DE VELOCIDAD (10 textos)")
    print("=" * 70)
    
    import time
    
    texto_prueba = "El concierto fue increíble y todos disfrutaron muchísimo"
    
    for nombre, clf in clasificadores:
        try:
            start = time.time()
            for _ in range(10):
                clf.classify(texto_prueba)
            elapsed = time.time() - start
            
            print(f"\n   {nombre}:")
            print(f"      Tiempo total: {elapsed:.4f}s")
            print(f"      Tiempo por clasificación: {elapsed/10:.4f}s")
        except Exception as e:
            print(f"\n   {nombre}: Error - {str(e)[:30]}")
    
    print("\n" + "=" * 70)
    print("✅ COMPARACIÓN COMPLETADA")
    print("=" * 70)
    
    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    print("\n   • Keyword-based: Rápido, sin dependencias, bueno para casos básicos")
    print("   • MarIA RoBERTa: Mejor precisión, requiere transformers, más lento")
    print("   • Sklearn: Balance entre precisión y velocidad, requiere entrenamiento")


if __name__ == "__main__":
    main()
