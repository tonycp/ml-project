"""
Ejemplo completo usando noticias reales desde la base de datos SQLite.

Este script:
1. Carga noticias desde noticias.db (SQLite)
2. Convierte a objetos NewsContent
3. Extrae eventos usando el pipeline completo
4. Muestra estadísticas y resultados
"""

import sys
from pathlib import Path
import sqlite3
from datetime import datetime
from collections import Counter

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Event_extractor.pipeline.event_pipeline import EventExtractionPipeline
from Event_extractor.models.news import NewsContent
from Event_extractor.classifiers.sentiment import KeywordSentimentClassifier

print("📰 EXTRACCIÓN DE EVENTOS DESDE NOTICIAS.DB")
print("=" * 70)

# Ruta a la base de datos
db_path = project_root / "noticias.db"

if not db_path.exists():
    print(f"\n❌ Error: No se encontró la base de datos en {db_path}")
    sys.exit(1)

print(f"\n📂 Base de datos: {db_path}")

# Conectar a la base de datos
print("\n🔌 Conectando a la base de datos...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Verificar estructura de la tabla
print("\n🔍 Inspeccionando estructura de la tabla...")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"   Tablas encontradas: {[t[0] for t in tables]}")

# Obtener información de las columnas
cursor.execute("PRAGMA table_info(noticias);")
columns = cursor.fetchall()
print(f"\n   Columnas en 'noticias':")
for col in columns:
    print(f"      • {col[1]} ({col[2]})")

# Cargar noticias desde la base de datos
print("\n📥 Cargando noticias...")
cursor.execute("SELECT id, msg_id, fecha, texto FROM noticias LIMIT 10")
rows = cursor.fetchall()

print(f"   ✅ {len(rows)} noticias cargadas (mostrando primeras 10)")

# Convertir a objetos NewsContent
news_list = []
for row in rows:
    id_val, msg_id, fecha_str, texto = row
    
    # Parsear la fecha
    try:
        # Intentar varios formatos comunes
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                publication_date = datetime.strptime(fecha_str, fmt)
                break
            except ValueError:
                continue
        else:
            # Si ninguno funciona, usar fecha actual
            publication_date = datetime.now()
    except Exception:
        publication_date = datetime.now()
    
    # Crear NewsContent (usar msg_id como identificador principal)
    news = NewsContent(
        id=str(msg_id),  # Usar msg_id como ID
        publication_date=publication_date,
        text=texto
    )
    
    news_list.append(news)

conn.close()

print("\n" + "=" * 70)
print("🔧 INICIALIZANDO PIPELINE")
print("=" * 70)

# Crear pipeline
pipeline = EventExtractionPipeline(
    classify_sentiment=True,
    sentiment_classifier=KeywordSentimentClassifier(),
    min_confidence=0.3
)

print("\n   ✅ Pipeline configurado con:")
print("      • Extracción de fechas")
print("      • Clasificación de tipo de evento")
print("      • Análisis de sentimiento")
print("      • Extracción de entidades (SVO)")

# Procesar noticias
print("\n" + "=" * 70)
print("📊 PROCESANDO NOTICIAS")
print("=" * 70)

all_events = []
noticias_con_eventos = 0

for idx, news in enumerate(news_list, 1):
    print(f"\n{'─' * 70}")
    print(f"📰 Noticia {idx}/{len(news_list)}")
    print(f"   ID: {news.id}")
    print(f"   Fecha: {news.publication_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Texto: {news.text[:100]}...")
    
    # Extraer eventos
    eventos = pipeline.extract_events(news)
    
    if not eventos:
        print("   ⚠️  No se encontraron eventos")
        continue
    
    noticias_con_eventos += 1
    all_events.extend(eventos)
    
    print(f"\n   ✅ {len(eventos)} evento(s) extraído(s):")
    
    for i, evento in enumerate(eventos, 1):
        print(f"\n      📅 Evento {i}:")
        print(f"         Fecha: {evento.date.strftime('%d/%m/%Y')}")
        print(f"         Tipo: {evento.event_type.value} (conf: {evento.confidence:.1%})")
        print(f"         Sentimiento: {evento.sentiment.value} (conf: {evento.sentiment_confidence:.1%})")
        print(f"         Fuente: {evento.source_news_id}")
        
        if evento.entidades_asociadas:
            # Mostrar solo las principales entidades
            subjects = [e for e in evento.entidades_asociadas if e['role'] == 'subject']
            named_ents = [e for e in evento.entidades_asociadas if e['role'] == 'named_entity']
            
            if named_ents:
                print(f"         Entidades: {', '.join([e['text'] for e in named_ents[:3]])}")
            elif subjects:
                print(f"         Sujetos: {', '.join([e['text'] for e in subjects[:3]])}")

# Resumen global
print("\n" + "=" * 70)
print("📈 RESUMEN GLOBAL")
print("=" * 70)

print(f"\n   📚 Noticias procesadas: {len(news_list)}")
print(f"   📰 Noticias con eventos: {noticias_con_eventos}")
print(f"   📅 Total de eventos: {len(all_events)}")

if all_events:
    # Estadísticas por tipo
    tipos = Counter([e.event_type.value for e in all_events])
    sentimientos = Counter([e.sentiment.value for e in all_events])
    
    print(f"\n   📊 Distribución por tipo de evento:")
    for tipo, count in tipos.most_common():
        porcentaje = (count / len(all_events)) * 100
        print(f"      • {tipo}: {count} ({porcentaje:.1f}%)")
    
    print(f"\n   😊 Distribución por sentimiento:")
    for sent, count in sentimientos.most_common():
        emoji = "😊" if sent == "positive" else "😢" if sent == "negative" else "😐"
        porcentaje = (count / len(all_events)) * 100
        print(f"      {emoji} {sent}: {count} ({porcentaje:.1f}%)")
    
    # Promedio de confianza
    avg_confidence = sum(e.confidence for e in all_events) / len(all_events)
    avg_sent_conf = sum(e.sentiment_confidence for e in all_events) / len(all_events)
    
    print(f"\n   🎯 Confianza promedio:")
    print(f"      • Tipo de evento: {avg_confidence:.1%}")
    print(f"      • Sentimiento: {avg_sent_conf:.1%}")
    
    # Entidades más frecuentes
    all_entities = []
    for evento in all_events:
        if evento.entidades_asociadas:
            all_entities.extend([e['text'] for e in evento.entidades_asociadas])
    
    if all_entities:
        entity_counts = Counter(all_entities)
        print(f"\n   🏷️  Top 10 entidades más frecuentes:")
        for entity, count in entity_counts.most_common(10):
            print(f"      • {entity}: {count}")
    
    # Línea de tiempo
    print("\n" + "=" * 70)
    print("📅 LÍNEA DE TIEMPO DE EVENTOS")
    print("=" * 70)
    
    sorted_events = sorted(all_events, key=lambda e: e.date)
    
    print(f"\n   Mostrando primeros 10 eventos ordenados por fecha:\n")
    
    for evento in sorted_events[:10]:
        sent_emoji = "😊" if evento.sentiment.value == "positive" else "😢" if evento.sentiment.value == "negative" else "😐"
        print(f"   {evento.date.strftime('%d/%m/%Y')} {sent_emoji} [{evento.event_type.value}]")
        
        # Buscar la noticia original para mostrar contexto
        noticia_orig = next((n for n in news_list if n.id == evento.source_news_id), None)
        if noticia_orig:
            print(f"      {noticia_orig.text[:80]}...")
        
        if evento.entidades_asociadas:
            subjects = [e['text'] for e in evento.entidades_asociadas if e['role'] in ['subject', 'named_entity']]
            if subjects:
                print(f"      Involucrados: {', '.join(subjects[:2])}")
        print()

else:
    print("\n   ⚠️  No se extrajeron eventos de las noticias procesadas")

print("\n" + "=" * 70)
print("✅ PROCESAMIENTO COMPLETADO")
print("=" * 70)

print("""
💡 RESUMEN:

• Se cargaron noticias reales desde noticias.db
• Se aplicó el pipeline completo de extracción
• Se identificaron fechas, tipos, sentimientos y entidades
• Los eventos están listos para análisis posterior

📊 Próximos pasos sugeridos:
   - Guardar eventos en una base de datos
   - Generar visualizaciones de la línea de tiempo
   - Analizar patrones en tipos de eventos
   - Estudiar correlaciones entre entidades
""")
