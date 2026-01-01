"""
Ejemplo de clasificación de sentimiento de eventos.

Este ejemplo muestra cómo el sistema clasifica eventos como
positivos, negativos o neutrales.
"""

from Event_extractor import (
    EventExtractionPipeline,
    EventSentimentClassifier,
    NewsContent,
    NewsMetadata,
    EventSentiment
)
from Event_extractor.utils.text_preprocessor import _tokenize_text
from datetime import datetime


def test_sentiment_classifier():
    """Prueba el clasificador de sentimiento con diferentes textos."""
    print("="*70)
    print("CLASIFICADOR DE SENTIMIENTO DE EVENTOS")
    print("="*70)
    
    classifier = EventSentimentClassifier()
    
    test_cases = [
        ("Gran festival de música en la ciudad", "Festival cultural"),
        ("Cancelación del concierto por mal tiempo", "Cancelación"),
        ("Grave accidente en la autopista principal", "Incidente"),
        ("Inauguración del nuevo hospital regional", "Inauguración"),
        ("Protestas contra las nuevas medidas económicas", "Protesta"),
        ("Victoria del equipo nacional en el campeonato", "Victoria deportiva"),
        ("Reunión de expertos sobre cambio climático", "Reunión técnica"),
        ("Cierre definitivo de la fábrica textil", "Cierre empresarial"),
        ("Celebración del aniversario de la ciudad", "Celebración"),
        ("Desastre natural deja cientos de damnificados", "Desastre"),
    ]
    
    print("\n📊 Clasificación de textos:\n")
    
    for text, label in test_cases:
        tokens = _tokenize_text(text)
        sentiment, confidence = classifier.classify(' '.join(tokens))
        
        # Emoji según sentimiento
        emoji = "✅" if sentiment == EventSentiment.POSITIVE else "❌" if sentiment == EventSentiment.NEGATIVE else "⚪"
        
        print(f"{emoji} {label}")
        print(f"   Texto: \"{text}\"")
        print(f"   Sentimiento: {sentiment.value.upper()}")
        print(f"   Confianza: {confidence:.0%}")
        
        # Mostrar desglose detallado
        detailed = classifier.classify_detailed(' '.join(tokens))
        print(f"   Desglose: Positivo={detailed[EventSentiment.POSITIVE]:.2f}, "
              f"Negativo={detailed[EventSentiment.NEGATIVE]:.2f}, "
              f"Neutral={detailed[EventSentiment.NEUTRAL]:.2f}")
        print()


def test_with_pipeline():
    """Prueba el pipeline completo con clasificación de sentimiento."""
    print("\n" + "="*70)
    print("PIPELINE COMPLETO CON CLASIFICACIÓN DE SENTIMIENTO")
    print("="*70)
    
    # Pipeline con clasificación de sentimiento
    pipeline = EventExtractionPipeline(classify_sentiment=True)
    
    news_examples = [
        {
            "title": "Gran Festival de Música 2025",
            "text": "El gran festival de música se realizará del 10 al 15 de enero de 2025. "
                    "El evento contará con artistas internacionales y será una celebración "
                    "inolvidable para todos los asistentes.",
            "expected": "POSITIVE"
        },
        {
            "title": "Cancelación de Vuelos por Tormenta",
            "text": "Cientos de vuelos fueron cancelados hoy debido a la fuerte tormenta. "
                    "Miles de pasajeros afectados están varados en el aeropuerto. "
                    "La situación es crítica y preocupante.",
            "expected": "NEGATIVE"
        },
        {
            "title": "Reunión del Comité Técnico",
            "text": "El comité técnico se reunirá el próximo 20 de enero para evaluar "
                    "los informes y presentar recomendaciones sobre el proyecto.",
            "expected": "NEUTRAL"
        },
        {
            "title": "Grave Accidente en Carretera",
            "text": "Un grave accidente en la carretera principal dejó varios heridos "
                    "el día de hoy. Los servicios de emergencia trabajan en el rescate "
                    "de las víctimas.",
            "expected": "NEGATIVE"
        },
        {
            "title": "Inauguración del Nuevo Parque",
            "text": "La inauguración del nuevo parque ecológico será el 25 de enero. "
                    "Es un logro importante para la ciudad y mejorará la calidad de vida "
                    "de los habitantes.",
            "expected": "POSITIVE"
        }
    ]
    
    print("\n📰 Procesando noticias...\n")
    
    for i, news_data in enumerate(news_examples, 1):
        metadata = NewsMetadata(
            title=news_data["title"],
            date=datetime(2024, 12, 20),
            source="Diario de Prueba"
        )
        
        news = NewsContent(
            text=news_data["text"],
            metadata=metadata,
            title=news_data["title"]
        )
        
        events = pipeline.extract_events(news)
        
        if events:
            event = events[0]  # Tomar el primer evento
            
            # Emoji según sentimiento
            emoji = "✅" if event.sentiment == EventSentiment.POSITIVE else \
                    "❌" if event.sentiment == EventSentiment.NEGATIVE else "⚪"
            
            print(f"{emoji} Noticia {i}: {news_data['title']}")
            print(f"   Tipo: {event.event_type.value}")
            print(f"   Sentimiento: {event.sentiment.value.upper()} (confianza: {event.sentiment_confidence:.0%})")
            print(f"   Esperado: {news_data['expected']}")
            
            # Verificar si coincide con lo esperado
            is_correct = event.sentiment.value.upper() == news_data['expected']
            print(f"   {'✓ Correcto' if is_correct else '✗ Incorrecto'}")
            print()


def test_custom_keywords():
    """Ejemplo de cómo añadir palabras clave personalizadas."""
    print("\n" + "="*70)
    print("PERSONALIZACIÓN DE PALABRAS CLAVE")
    print("="*70)
    
    classifier = EventSentimentClassifier()
    
    text = "Gran vernissage de arte contemporáneo en la galería"
    
    print("\n🎨 Caso: Evento cultural específico")
    print(f"   Texto: \"{text}\"")
    
    # Antes de añadir palabras clave
    sentiment1, conf1 = classifier.classify(text)
    print(f"\n   ANTES de añadir 'vernissage' como positivo:")
    print(f"   → Sentimiento: {sentiment1.value}, Confianza: {conf1:.0%}")
    
    # Añadir palabra clave personalizada
    classifier.add_positive_keywords(["vernissage", "bienal", "retrospectiva"])
    
    # Después de añadir palabras clave
    sentiment2, conf2 = classifier.classify(text)
    print(f"\n   DESPUÉS de añadir 'vernissage' como positivo:")
    print(f"   → Sentimiento: {sentiment2.value}, Confianza: {conf2:.0%}")
    print()


def main():
    """Función principal."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "CLASIFICACIÓN DE SENTIMIENTO DE EVENTOS" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    print()
    print("Este ejemplo demuestra cómo el sistema clasifica eventos como")
    print("positivos, negativos o neutrales basándose en el contenido.")
    print()
    
    test_sentiment_classifier()
    test_with_pipeline()
    test_custom_keywords()
    
    print("\n" + "="*70)
    print("RESUMEN DE SENTIMIENTOS")
    print("="*70)
    print("""
✅ POSITIVO:
   • Celebraciones: festivales, inauguraciones, aniversarios
   • Logros: victorias, premios, éxitos, récords
   • Eventos positivos: bodas, graduaciones, reconocimientos
   • Mejoras: avances, progresos, recuperaciones

❌ NEGATIVO:
   • Cancelaciones y suspensiones
   • Protestas, conflictos, huelgas
   • Accidentes, incidentes, desastres
   • Pérdidas, derrotas, crisis
   • Cierres, despidos, sanciones

⚪ NEUTRAL:
   • Reuniones, conferencias, congresos
   • Anuncios, comunicados, informes
   • Procesos administrativos
   • Eventos informativos

💡 El sentimiento se puede personalizar añadiendo palabras clave específicas
   según el dominio de aplicación.
""")


if __name__ == "__main__":
    main()
