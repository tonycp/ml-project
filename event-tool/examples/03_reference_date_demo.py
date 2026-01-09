"""
Ejemplo que demuestra el comportamiento con y sin reference_date.

Este ejemplo muestra cómo el extractor evita fechas erróneas cuando
no hay una fecha de referencia explícita.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.Event_extractor import DateExtractor
from src.Event_extractor.utils.text_preprocessor import _tokenize_text
from datetime import datetime


def test_without_reference_date():
    """Prueba extracción sin reference_date - solo fechas explícitas."""
    print("="*70)
    print("EXTRACCIÓN SIN REFERENCE_DATE (fecha de referencia)")
    print("="*70)
    print("\n⚠️  Sin reference_date, solo se extraen fechas EXPLÍCITAS Y COMPLETAS\n")
    
    # Sin reference_date
    extractor = DateExtractor(reference_date=None)
    
    texts = [
        ("Fecha explícita completa", "El evento será el 25 de diciembre de 2024"),
        ("Fecha numérica completa", "La reunión es el 15/01/2025"),
        ("Fecha sin año", "El concierto es el 20 de marzo"),  # NO se extrae
        ("Fecha relativa", "La conferencia es mañana"),  # NO se extrae
        ("Rango sin año", "El festival es del 10 al 15 de enero"),  # NO se extrae
        ("Rango con año", "El festival es del 10 al 15 de enero de 2025"),  # SÍ se extrae
    ]
    
    for label, text in texts:
        tokens = _tokenize_text(text)
        dates = extractor.extract_dates(tokens)
        print(f"📄 {label}:")
        print(f"   Texto: \"{text}\"")
        if dates:
            print(f"   ✅ Fechas encontradas: {len(dates)}")
            for date in dates:
                print(f"      → {date.strftime('%d/%m/%Y')}")
        else:
            print(f"   ❌ No se encontraron fechas (requiere fecha de referencia)")
        print()


def test_with_reference_date():
    """Prueba extracción con reference_date - todos los tipos de fechas."""
    print("\n" + "="*70)
    print("EXTRACCIÓN CON REFERENCE_DATE (fecha de referencia)")
    print("="*70)
    print("\n✅ Con reference_date, se extraen TODO TIPO de fechas\n")
    
    # Con reference_date
    reference = datetime(2024, 12, 25)
    extractor = DateExtractor(reference_date=reference)
    
    print(f"Fecha de referencia: {reference.strftime('%d/%m/%Y')}\n")
    
    texts = [
        ("Fecha explícita completa", "El evento será el 10 de enero de 2025"),
        ("Fecha numérica completa", "La reunión es el 15/01/2025"),
        ("Fecha sin año", "El concierto es el 20 de marzo"),  # Usa año 2024
        ("Fecha relativa 'mañana'", "La conferencia es mañana"),  # 26/12/2024
        ("Fecha relativa 'hoy'", "El evento es hoy"),  # 25/12/2024
        ("Rango sin año", "El festival es del 10 al 15 de enero"),  # Usa año 2024
        ("Rango con año", "El festival es del 10 al 15 de enero de 2025"),
    ]
    
    for label, text in texts:
        tokens = _tokenize_text(text)
        dates = extractor.extract_dates(tokens)
        print(f"📄 {label}:")
        print(f"   Texto: \"{text}\"")
        if dates:
            print(f"   ✅ Fechas encontradas: {len(dates)}")
            for date in dates:
                print(f"      → {date.strftime('%d/%m/%Y')}")
        else:
            print(f"   ⚠️  No se encontraron fechas")
        print()


def test_with_pipeline():
    """Prueba el pipeline que usa la fecha de metadata automáticamente."""
    print("\n" + "="*70)
    print("USO CON PIPELINE - Usa fecha de metadata automáticamente")
    print("="*70)
    
    from src.Event_extractor import EventExtractionPipeline, NewsContent
    
    # Pipeline sin reference_date explícito
    pipeline = EventExtractionPipeline()
    
    # Noticia con metadata que incluye fecha de publicación
    news = NewsContent(
        text="El gran festival se realizará del 10 al 15 de enero. Habrá conciertos todos los días.",
        metadata=metadata,
        title="Festival Musical"
    )
    
    print(f"\n📰 Procesando noticia:")
    print(f"   Título: {news.title}")
    print(f"   Fecha de publicación: {metadata.date.strftime('%d/%m/%Y')}")
    print(f"   Texto: \"{news.text[:60]}...\"")
    
    events = pipeline.extract_events(news)
    
    print(f"\n✅ Eventos extraídos: {len(events)}")
    for i, event in enumerate(events, 1):
        print(f"\n   Evento {i}:")
        print(f"   • Fecha: {event.date.strftime('%d/%m/%Y')}")
        print(f"   • Tipo: {event.event_type.value}")
        print(f"   • Título: {event.title}")
        print(f"   • Confianza: {event.confidence:.0%}")
    
    print("\n💡 Nota: Como el rango no tiene año explícito, usa el año de la")
    print("   fecha de publicación (2024) de la metadata de la noticia.")


def main():
    """Función principal."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*10 + "DEMOSTRACIÓN: MANEJO DE FECHAS CON/SIN REFERENCIA" + " "*10 + "║")
    print("╚" + "="*68 + "╝")
    print()
    print("Este ejemplo muestra cómo el extractor previene fechas erróneas")
    print("cuando no hay una fecha de referencia disponible.")
    print()
    
    test_without_reference_date()
    test_with_reference_date()
    test_with_pipeline()
    
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print("""
✅ SIN reference_date:
   • Solo extrae fechas explícitas completas (con año)
   • Evita fechas erróneas por falta de contexto
   • Ideal cuando no conoces la fecha de publicación

✅ CON reference_date:
   • Extrae todo tipo de fechas (relativas, sin año, rangos)
   • Usa la fecha de referencia para resolver ambigüedades
   • Ideal cuando tienes la fecha de publicación de la noticia

💡 El Pipeline usa automáticamente la fecha de metadata si está disponible
""")


if __name__ == "__main__":
    main()
