"""
Ejemplo de uso de componentes individuales del Event Extractor.

Este ejemplo muestra cómo usar cada componente por separado.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Event_extractor import DateExtractor, EventTypeClassifier, EventType
from Event_extractor.utils.text_preprocessor import _tokenize_text
from datetime import datetime


def example_date_extraction():
    """Ejemplo de uso del DateExtractor."""
    print("="*60)
    print("Ejemplo: Extracción de Fechas")
    print("="*60)
    
    extractor = DateExtractor(reference_date=datetime(2025, 1, 1))
    
    texts = [
        "El evento será el 15 de enero de 2025",
        "La conferencia se realizará mañana",
        "El festival durará del 10 al 15 de marzo",
        "La reunión es hoy a las 3 PM",
        "El concierto será el próximo 25/12/2024"
    ]
    
    for text in texts:
        tokens = _tokenize_text(text)
        dates = extractor.extract_dates(tokens)
        print(f"\nTexto: {text}")
        print(f"Fechas encontradas: {len(dates)}")
        for date in dates:
            print(f"  - {date.strftime('%d/%m/%Y %H:%M')}")


def example_event_classification():
    """Ejemplo de uso del EventTypeClassifier."""
    print("\n" + "="*60)
    print("Ejemplo: Clasificación de Tipos de Evento")
    print("="*60)
    
    classifier = EventTypeClassifier()
    
    texts = [
        "Gran concierto de rock este sábado en el estadio",
        "Se espera una fuerte tormenta para mañana",
        "Final del campeonato de fútbol este domingo",
        "Elecciones presidenciales en noviembre",
        "La bolsa cerró con pérdidas significativas",
        "Manifestación por los derechos laborales",
        "Grave accidente en la autopista",
        "Nueva regulación para el sector aéreo"
    ]
    
    for text in texts:
        tokens = _tokenize_text(text)
        event_type, confidence = classifier.classify(' '.join(tokens))
        print(f"\nTexto: {text}")
        print(f"Tipo: {event_type.value}")
        print(f"Confianza: {confidence:.2f}")
        
        # También mostrar los top 3 tipos más probables
        top_types = classifier.classify_multiple(text, top_k=3)
        print("Top 3 tipos:")
        for et, conf in top_types:
            print(f"  - {et.value}: {conf:.2f}")


def example_custom_keywords():
    """Ejemplo de añadir palabras clave personalizadas."""
    print("\n" + "="*60)
    print("Ejemplo: Palabras Clave Personalizadas")
    print("="*60)
    
    classifier = EventTypeClassifier()
    
    text = "Gran bienal de arte contemporáneo"
    
    # Antes de añadir palabras clave
    event_type, confidence = classifier.classify(text)
    print(f"\nAntes de añadir 'bienal':")
    print(f"Texto: {text}")
    print(f"Tipo: {event_type.value}")
    print(f"Confianza: {confidence:.2f}")
    
    # Añadir palabras clave personalizadas
    classifier.add_keywords(EventType.CULTURAL, ["bienal", "vernissage"])
    
    # Después de añadir palabras clave
    event_type, confidence = classifier.classify(text)
    print(f"\nDespués de añadir 'bienal':")
    print(f"Tipo: {event_type.value}")
    print(f"Confianza: {confidence:.2f}")


def main():
    """Función principal del ejemplo."""
    print("\nEjemplos de Componentes Individuales\n")
    
    example_date_extraction()
    example_event_classification()
    example_custom_keywords()
    
    print("\n" + "="*60)
    print("Fin de los ejemplos")
    print("="*60)


if __name__ == "__main__":
    main()
