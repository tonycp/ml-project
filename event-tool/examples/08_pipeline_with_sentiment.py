"""
Ejemplo de uso del pipeline con diferentes clasificadores de sentimiento.

Demuestra cómo integrar los clasificadores de sentimiento en el pipeline principal.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from src.Event_extractor.pipeline.event_pipeline import EventExtractionPipeline
from src.Event_extractor.models.news import NewsContent
from src.Event_extractor.classifiers.sentiment import (
    KeywordSentimentClassifier,
    MarIASentimentClassifier,
    SklearnSentimentClassifier
)


def main():
    print("🔄 PIPELINE CON CLASIFICADORES DE SENTIMIENTO")
    print("=" * 70)
    
    # Noticia de ejemplo
    noticia = NewsContent(
        title="Festival de Música Cancelado por Mal Tiempo",
        url="https://example.com/festival-cancelado",
        text="""
        El esperado Festival de Música de Verano ha sido cancelado debido a las 
        condiciones climáticas adversas. El evento estaba programado para el 15 de 
        julio y se esperaba la asistencia de más de 10,000 personas.
        
        Los organizadores expresaron su profunda decepción y anunciaron que los 
        boletos serán reembolsados. "Es una situación muy triste para todos", 
        comentó el director del festival.
        
        Se está evaluando reprogramar el evento para agosto, pero aún no hay 
        confirmación oficial.
        """,
        published_date=datetime(2024, 7, 1)
    )
    
    print(f"\n📰 Noticia: {noticia.title}")
    print(f"   Fecha: {noticia.published_date.strftime('%Y-%m-%d')}")
    print(f"   Texto: {noticia.text[:100]}...")
    
    # Probar con diferentes clasificadores de sentimiento
    clasificadores = []
    
    # 1. Keyword-based (por defecto)
    try:
        keyword_clf = KeywordSentimentClassifier()
        clasificadores.append(("Keyword-based", keyword_clf))
    except Exception as e:
        print(f"⚠️  KeywordSentimentClassifier: {e}")
    
    # 2. MarIA/RoBERTa
    try:
        maria_clf = MarIASentimentClassifier()
        clasificadores.append(("MarIA RoBERTa", maria_clf))
    except Exception as e:
        print(f"⚠️  MarIASentimentClassifier: {str(e)[:50]}")
    
    # 3. Sklearn
    try:
        sklearn_clf = SklearnSentimentClassifier.load_model("models/sklearn_tass_sentiment.pkl")
        clasificadores.append(("Sklearn (TASS)", sklearn_clf))
    except Exception as e:
        print(f"⚠️  SklearnSentimentClassifier: {str(e)[:50]}")
    
    if not clasificadores:
        print("\n❌ No hay clasificadores disponibles")
        return
    
    print(f"\n✅ Clasificadores disponibles: {len(clasificadores)}")
    
    # Extraer eventos con cada clasificador
    print("\n" + "=" * 70)
    print("🔍 EXTRACCIÓN DE EVENTOS")
    print("=" * 70)
    
    for nombre, clf in clasificadores:
        print(f"\n{'─' * 70}")
        print(f"📊 Usando: {nombre}")
        print(f"{'─' * 70}")
        
        try:
            # Crear pipeline con el clasificador de sentimiento
            pipeline = EventExtractionPipeline(
                reference_date=datetime(2024, 7, 1),
                classify_sentiment=True,
                sentiment_classifier=clf
            )
            
            # Extraer eventos
            eventos = pipeline.extract_events(noticia)
            
            print(f"\n   Eventos encontrados: {len(eventos)}")
            
            for idx, evento in enumerate(eventos, 1):
                print(f"\n   Evento {idx}:")
                print(f"      Fecha: {evento.date.strftime('%Y-%m-%d')}")
                print(f"      Tipo: {evento.event_type.value}")
                print(f"      Sentimiento: {evento.sentiment.value}")
                print(f"      Confianza tipo: {evento.confidence:.2%}")
                print(f"      Confianza sentimiento: {evento.sentiment_confidence:.2%}")
                print(f"      Descripción: {evento.description[:80]}...")
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}")
    
    # Ejemplo comparativo
    print("\n" + "=" * 70)
    print("📊 COMPARACIÓN DE SENTIMIENTOS")
    print("=" * 70)
    
    texto_ejemplo = noticia.title + ". " + noticia.text[:200]
    
    print(f"\n📝 Texto: {texto_ejemplo[:100]}...")
    print(f"\nSentimientos detectados:")
    
    for nombre, clf in clasificadores:
        try:
            sentiment, confidence = clf.classify(texto_ejemplo)
            print(f"   • {nombre:20s}: {sentiment.value:8s} (conf: {confidence:.2%})")
        except Exception as e:
            print(f"   • {nombre:20s}: Error - {str(e)[:30]}")
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 70)
    
    print("\n💡 CÓMO USAR EN TU CÓDIGO:")
    print("""
    # Con clasificador keyword (por defecto)
    pipeline = EventExtractionPipeline(classify_sentiment=True)
    
    # Con MarIA/RoBERTa
    from src.Event_extractor.classifiers.sentiment import MarIASentimentClassifier
    pipeline = EventExtractionPipeline(
        sentiment_classifier=MarIASentimentClassifier()
    )
    
    # Con Sklearn entrenado
    from src.Event_extractor.classifiers.sentiment import SklearnSentimentClassifier
    clf = SklearnSentimentClassifier.load_model("models/sklearn_tass_sentiment.pkl")
    pipeline = EventExtractionPipeline(sentiment_classifier=clf)
    """)


if __name__ == "__main__":
    main()
