"""
Plantilla para crear un adaptador de formato de noticias.

Este archivo es una guía para cuando conozcas el formato exacto en el que
llegarán las noticias. Deberás implementar la función `load_news_from_source()`
según tu fuente de datos específica.
"""

from typing import List, Dict, Any
from datetime import datetime
from Event_extractor import NewsContent, NewsMetadata


def load_news_from_json_file(filepath: str) -> List[NewsContent]:
    """
    Ejemplo: Carga noticias desde un archivo JSON.
    
    Formato esperado del JSON:
    [
        {
            "title": "Título de la noticia",
            "text": "Contenido completo de la noticia...",
            "date": "2024-12-25",
            "source": "Nombre del medio",
            "author": "Nombre del autor",
            "category": "Categoría",
            "url": "https://ejemplo.com/noticia"
        },
        ...
    ]
    
    Args:
        filepath: Ruta al archivo JSON
        
    Returns:
        Lista de objetos NewsContent
    """
    import json
    from dateutil import parser
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_news = json.load(f)
    
    news_list = []
    for item in raw_news:
        # Parsear la fecha
        try:
            news_date = parser.parse(item.get('date', ''))
        except:
            news_date = datetime.now()
        
        # Crear metadata
        metadata = NewsMetadata(
            title=item.get('title', ''),
            date=news_date,
            source=item.get('source', 'Unknown'),
            author=item.get('author'),
            category=item.get('category'),
            url=item.get('url'),
            tags=item.get('tags')
        )
        
        # Crear NewsContent
        news = NewsContent(
            text=item.get('text', ''),
            metadata=metadata,
            title=item.get('title'),
            raw_data=item
        )
        
        news_list.append(news)
    
    return news_list


def load_news_from_csv_file(filepath: str) -> List[NewsContent]:
    """
    Ejemplo: Carga noticias desde un archivo CSV.
    
    Formato esperado del CSV:
    title,text,date,source,author,category,url
    "Título","Contenido...","2024-12-25","Medio","Autor","Categoría","https://..."
    
    Args:
        filepath: Ruta al archivo CSV
        
    Returns:
        Lista de objetos NewsContent
    """
    import csv
    from dateutil import parser
    
    news_list = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Parsear la fecha
            try:
                news_date = parser.parse(row.get('date', ''))
            except:
                news_date = datetime.now()
            
            # Crear metadata
            metadata = NewsMetadata(
                title=row.get('title', ''),
                date=news_date,
                source=row.get('source', 'Unknown'),
                author=row.get('author'),
                category=row.get('category'),
                url=row.get('url')
            )
            
            # Crear NewsContent
            news = NewsContent(
                text=row.get('text', ''),
                metadata=metadata,
                title=row.get('title'),
                raw_data=row
            )
            
            news_list.append(news)
    
    return news_list


def load_news_from_api(api_url: str, api_key: str = None) -> List[NewsContent]:
    """
    Ejemplo: Carga noticias desde una API REST.
    
    Args:
        api_url: URL de la API
        api_key: Clave de API (opcional)
        
    Returns:
        Lista de objetos NewsContent
    """
    import requests
    from dateutil import parser
    
    # Configurar headers
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    # Hacer petición
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    news_list = []
    
    # Adaptar según la estructura de tu API
    for item in data.get('articles', []):  # Ajustar según tu API
        # Parsear la fecha
        try:
            news_date = parser.parse(item.get('publishedAt', ''))
        except:
            news_date = datetime.now()
        
        # Crear metadata
        metadata = NewsMetadata(
            title=item.get('title', ''),
            date=news_date,
            source=item.get('source', {}).get('name', 'Unknown'),
            author=item.get('author'),
            url=item.get('url')
        )
        
        # Crear NewsContent
        news = NewsContent(
            text=item.get('content', '') or item.get('description', ''),
            metadata=metadata,
            title=item.get('title'),
            raw_data=item
        )
        
        news_list.append(news)
    
    return news_list


def load_news_from_database(connection_string: str, query: str = None) -> List[NewsContent]:
    """
    Ejemplo: Carga noticias desde una base de datos.
    
    Args:
        connection_string: Cadena de conexión a la BD
        query: Consulta SQL personalizada (opcional)
        
    Returns:
        Lista de objetos NewsContent
    """
    # Ejemplo con SQLite - adaptar según tu BD
    import sqlite3
    from dateutil import parser
    
    conn = sqlite3.connect(connection_string)
    cursor = conn.cursor()
    
    # Query por defecto
    if query is None:
        query = """
            SELECT title, text, date, source, author, category, url
            FROM news
            ORDER BY date DESC
        """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    news_list = []
    
    for row in rows:
        title, text, date_str, source, author, category, url = row
        
        # Parsear la fecha
        try:
            news_date = parser.parse(date_str)
        except:
            news_date = datetime.now()
        
        # Crear metadata
        metadata = NewsMetadata(
            title=title,
            date=news_date,
            source=source,
            author=author,
            category=category,
            url=url
        )
        
        # Crear NewsContent
        news = NewsContent(
            text=text,
            metadata=metadata,
            title=title,
            raw_data=dict(zip(
                ['title', 'text', 'date', 'source', 'author', 'category', 'url'],
                row
            ))
        )
        
        news_list.append(news)
    
    conn.close()
    return news_list


def load_news_from_rss(rss_url: str) -> List[NewsContent]:
    """
    Ejemplo: Carga noticias desde un feed RSS.
    
    Args:
        rss_url: URL del feed RSS
        
    Returns:
        Lista de objetos NewsContent
    """
    import feedparser
    from dateutil import parser
    
    feed = feedparser.parse(rss_url)
    
    news_list = []
    
    for entry in feed.entries:
        # Parsear la fecha
        try:
            if hasattr(entry, 'published_parsed'):
                news_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed'):
                news_date = datetime(*entry.updated_parsed[:6])
            else:
                news_date = datetime.now()
        except:
            news_date = datetime.now()
        
        # Crear metadata
        metadata = NewsMetadata(
            title=entry.get('title', ''),
            date=news_date,
            source=feed.feed.get('title', 'RSS Feed'),
            author=entry.get('author'),
            url=entry.get('link')
        )
        
        # Crear NewsContent
        text = entry.get('summary', '') or entry.get('description', '')
        # Limpiar HTML si es necesario
        if '<' in text:
            from html import unescape
            import re
            text = unescape(re.sub('<[^<]+?>', '', text))
        
        news = NewsContent(
            text=text,
            metadata=metadata,
            title=entry.get('title'),
            raw_data=dict(entry)
        )
        
        news_list.append(news)
    
    return news_list


# =============================================================================
# USO COMPLETO: Carga + Extracción
# =============================================================================

def complete_pipeline_example():
    """
    Ejemplo completo: desde la carga hasta la extracción de eventos.
    """
    from Event_extractor import EventExtractionPipeline, EventAggregator
    
    # 1. Cargar noticias (usa la función apropiada para tu fuente)
    print("Cargando noticias...")
    # Descomentar y adaptar según tu fuente:
    # news_list = load_news_from_json_file('noticias.json')
    # news_list = load_news_from_csv_file('noticias.csv')
    # news_list = load_news_from_api('https://api.ejemplo.com/news')
    # news_list = load_news_from_database('noticias.db')
    # news_list = load_news_from_rss('https://ejemplo.com/rss')
    
    # Para este ejemplo, creamos noticias de prueba
    news_list = create_sample_news()
    
    print(f"Noticias cargadas: {len(news_list)}")
    
    # 2. Crear pipeline de extracción
    print("\nInicializando pipeline...")
    pipeline = EventExtractionPipeline()
    
    # 3. Extraer eventos
    print("Extrayendo eventos...")
    all_events = pipeline.extract_events_batch(news_list)
    print(f"Eventos extraídos: {len(all_events)}")
    
    # 4. Procesar eventos
    print("\nProcesando eventos...")
    unique_events = EventAggregator.remove_duplicates(all_events)
    sorted_events = EventAggregator.sort_by_date(unique_events)
    
    # 5. Mostrar resultados
    print(f"\nEventos únicos: {len(unique_events)}")
    print("\nPrimeros 10 eventos:")
    for i, event in enumerate(sorted_events[:10], 1):
        print(f"{i}. {event.date.strftime('%d/%m/%Y')} - "
              f"{event.event_type.value} - {event.title}")
    
    # 6. Guardar resultados (opcional)
    save_events_to_json(sorted_events, 'eventos_extraidos.json')
    print("\nEventos guardados en 'eventos_extraidos.json'")


def create_sample_news() -> List[NewsContent]:
    """Crea noticias de muestra para testing."""
    sample_data = [
        {
            "title": "Festival de Música 2025",
            "text": "El gran festival de música se realizará del 10 al 15 de enero de 2025.",
            "date": "2024-12-01",
            "source": "Diario Cultural"
        },
        {
            "title": "Alerta Meteorológica",
            "text": "Se espera una fuerte tormenta para mañana en toda la región.",
            "date": "2024-12-24",
            "source": "Servicio Meteorológico"
        }
    ]
    
    news_list = []
    for item in sample_data:
        from dateutil import parser
        
        metadata = NewsMetadata(
            title=item['title'],
            date=parser.parse(item['date']),
            source=item['source']
        )
        
        news = NewsContent(
            text=item['text'],
            metadata=metadata,
            title=item['title']
        )
        
        news_list.append(news)
    
    return news_list


def save_events_to_json(events: List, filepath: str):
    """Guarda eventos en un archivo JSON."""
    import json
    
    events_data = []
    for event in events:
        events_data.append({
            'date': event.date.isoformat(),
            'event_type': event.event_type.value,
            'title': event.title,
            'description': event.description,
            'confidence': event.confidence,
            'source_news_id': event.source_news_id
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(events_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Ejecutar el ejemplo completo
    complete_pipeline_example()
