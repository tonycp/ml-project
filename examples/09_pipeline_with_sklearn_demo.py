"""
Demo del pipeline de extracción de eventos usando el clasificador sklearn.
Usa Spanish News corpus con modelo SVM Linear entrenado.
"""

import sys
from pathlib import Path
from datetime import datetime

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Event_extractor.pipeline.event_pipeline import EventExtractionPipeline
from Event_extractor.models.news import NewsContent

def demo_pipeline_sklearn():
    """Demuestra el uso del pipeline con clasificador sklearn."""
    
    print("=" * 80)
    print("DEMO: PIPELINE CON CLASIFICADOR SKLEARN (Spanish News + SVM Linear)")
    print("=" * 80)
    print()
    
    # Verificar que el modelo existe
    model_path = project_root / "models" / "sklearn_spanish_svm.pkl"
    if not model_path.exists():
        print("⚠️  MODELO NO ENCONTRADO")
        print()
        print("Primero necesitas entrenar el modelo:")
        print("   python examples/train_sklearn_model.py")
        print()
        return
    
    print(f"✅ Modelo encontrado: {model_path}")
    print()
    
    # Crear pipeline con clasificador sklearn
    print("🔧 Inicializando pipeline con clasificador sklearn...")
    pipeline = EventExtractionPipeline(
        reference_date=datetime(2024, 1, 15),
        use_sklearn_classifier=True,  # ← Usar sklearn en lugar de keywords
        sklearn_model_path=str(model_path)
    )
    print("   ✅ Pipeline inicializado")
    print()
    
    # Noticias de prueba en español
    news_samples = [
        {
            "title": "Real Madrid gana la Champions League",
            "text": """El Real Madrid se coronó campeón de la Champions League 
            el sábado pasado tras vencer 2-1 al Liverpool en la final disputada 
            en París. Los goles fueron anotados por Vinicius Jr. y Karim Benzema. 
            El próximo partido será el 20 de enero.""",
            "category": "Deportes"
        },
        {
            "title": "El gobierno anuncia reforma económica",
            "text": """El gobierno español anunció ayer una nueva reforma económica 
            que entrará en vigor el próximo mes. Las medidas incluyen reducción 
            de impuestos para empresas pequeñas y medianas. La conferencia de 
            prensa se realizó el martes 10 de enero.""",
            "category": "Economía"
        },
        {
            "title": "Festival de música en Barcelona",
            "text": """El Primavera Sound se celebrará del 25 al 30 de enero en 
            Barcelona. El festival contará con la participación de artistas 
            internacionales como Coldplay, The Strokes y Bad Bunny. Las entradas 
            salieron a la venta ayer.""",
            "category": "Cultural"
        },
        {
            "title": "Tensiones diplomáticas en Europa",
            "text": """Los líderes europeos se reunirán mañana en Bruselas para 
            discutir la crisis diplomática con Rusia. La cumbre está programada 
            para el 16 de enero y se espera que asistan todos los ministros de 
            relaciones exteriores.""",
            "category": "Política"
        }
    ]
    
    print("📰 Procesando noticias de ejemplo...")
    print("=" * 80)
    print()
    
    for i, news_data in enumerate(news_samples, 1):
        print(f"NOTICIA {i}: {news_data['title']}")
        print(f"Categoría esperada: {news_data['category']}")
        print("-" * 80)
        
        # Crear objeto NewsContent
        news = NewsContent(
            text=news_data['text'],
            title=news_data['title'],
            publication_date=datetime(2024, 1, 15)
        )
        
        # Extraer eventos
        events = pipeline.extract_events(news)
        
        if events:
            print(f"   ✅ Eventos extraídos: {len(events)}")
            for j, event in enumerate(events, 1):
                print(f"\n   Evento {j}:")
                print(f"      📅 Fecha: {event.date.strftime('%Y-%m-%d')}")
                print(f"      🏷️  Tipo: {event.event_type.value}")
                print(f"      📊 Confianza: {event.confidence:.2%}")
                print(f"      😊 Sentimiento: {event.sentiment.value}")
        else:
            print("   ⚠️  No se extrajeron eventos")
        
        print()
    
    print("=" * 80)
    print("✅ DEMO COMPLETADA")
    print("=" * 80)
    print()
    print("Comparación de clasificadores:")
    print("   • Basado en keywords: Reglas manuales, rápido pero limitado")
    print("   • Sklearn (TF-IDF + SVM): Machine learning, más preciso")
    print()
    print("El clasificador sklearn:")
    print("   ✅ Usa Spanish News corpus (10k noticias en español)")
    print("   ✅ Modelo SVM Linear (mejor rendimiento: 95.1% accuracy)")
    print("   ✅ Tokenización con SpaCy (lematización)")
    print("   ✅ Vectorización TF-IDF (10k features, bigrams)")
    print()

if __name__ == "__main__":
    demo_pipeline_sklearn()
