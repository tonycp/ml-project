"""
Ejemplo básico de uso del Event Extractor.

Este ejemplo muestra cómo usar el pipeline completo para extraer
eventos de noticias.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Event_extractor import (
    EventExtractionPipeline,
    NewsContent,
    EventAggregator,
    EventType
)
from Event_extractor.utils.text_preprocessor import _tokenize_text
from datetime import datetime


def main():
    """Función principal del ejemplo."""
    
    # Crear el pipeline
    print("Inicializando pipeline de extracción de eventos...")
    pipeline = EventExtractionPipeline()
    
    # Ejemplo 1: Festival de música
    print("\n" + "="*60)
    print("Ejemplo 1: Festival de música")
    print("="*60)
    
    news1 = NewsContent(
        id="news_001",
        text=(
            "El gran festival de música de la ciudad se realizará del 10 al 15 de "
            "enero de 2025 en el parque central. El evento contará con la participación "
            "de artistas nacionales e internacionales. Habrá conciertos de rock, pop "
            "y música electrónica. Las entradas ya están a la venta."
        ),
        publication_date=datetime(2024, 12, 1)
    )
    
    events1 = pipeline.extract_events(news1)
    print_events(events1)
    
    # Ejemplo 2: Partido de fútbol
    print("\n" + "="*60)
    print("Ejemplo 2: Evento deportivo")
    print("="*60)
    
    news2 = NewsContent(
        id="news_002",
        text=(
            "La final del campeonato nacional de fútbol se jugará el próximo "
            "25 de diciembre de 2024 en el estadio principal. Los dos mejores "
            "equipos de la liga se enfrentarán por el título. El partido comenzará "
            "a las 20:00 horas."
        ),
        publication_date=datetime(2024, 12, 1)
    )
    
    events2 = pipeline.extract_events(news2)
    print_events(events2)
    
    # Ejemplo 3: Alerta meteorológica
    print("\n" + "="*60)
    print("Ejemplo 3: Alerta meteorológica")
    print("="*60)
    
    news3 = NewsContent(
        id="news_003",
        text=(
            "Se ha emitido una alerta meteorológica para mañana debido a una "
            "fuerte tormenta que afectará la región. Se esperan lluvias intensas "
            "y vientos de hasta 80 km/h. Se recomienda a la población tomar "
            "precauciones y evitar salir de sus hogares."
        ),
        publication_date=datetime(2024, 12, 20)
    )
    
    events3 = pipeline.extract_events(news3)
    print_events(events3)
    
    # Procesar todas las noticias en batch
    print("\n" + "="*60)
    print("Procesamiento en batch")
    print("="*60)
    
    all_news = [news1, news2, news3]
    all_events = pipeline.extract_events_batch(all_news)
    
    print(f"\nTotal de eventos extraídos: {len(all_events)}")
    
    # Eliminar duplicados
    unique_events = EventAggregator.remove_duplicates(all_events)
    print(f"Eventos únicos: {len(unique_events)}")
    
    # Filtrar por tipo
    cultural_events = EventAggregator.filter_by_type(
        unique_events,
        [EventType.CULTURAL]
    )
    print(f"Eventos culturales: {len(cultural_events)}")
    
    deportive_events = EventAggregator.filter_by_type(
        unique_events,
        [EventType.DEPORTIVO]
    )
    print(f"Eventos deportivos: {len(deportive_events)}")
    
    # Ordenar por fecha
    sorted_events = EventAggregator.sort_by_date(unique_events)
    
    print("\nEventos ordenados por fecha:")
    for i, event in enumerate(sorted_events, 1):
        print(f"{i}. {event.date.strftime('%d/%m/%Y')} - {event.event_type.value} - {event.title}")


def print_events(events):
    """
    Imprime los eventos de forma legible.
    
    Args:
        events: Lista de eventos a imprimir
    """
    if not events:
        print("No se encontraron eventos")
        return
    
    print(f"\nEventos encontrados: {len(events)}")
    for i, event in enumerate(events, 1):
        print(f"\nEvento {i}:")
        print(f"  Fecha: {event.date.strftime('%d/%m/%Y %H:%M')}")
        print(f"  Tipo: {event.event_type.value}")
        print(f"  Título: {event.title}")
        if event.description:
            print(f"  Descripción: {event.description[:100]}...")
        print(f"  Confianza: {event.confidence:.2f}")


if __name__ == "__main__":
    main()
