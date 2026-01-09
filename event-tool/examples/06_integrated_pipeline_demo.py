"""
Demostración del pipeline completo integrado con todas las funcionalidades.

Este ejemplo muestra:
1. Extracción de fechas
2. Clasificación de tipo de evento
3. Clasificación de sentimiento
4. Extracción de entidades relacionadas (SVO)
5. Generación de eventos completos
"""

import sys
from pathlib import Path
from datetime import datetime

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.Event_extractor.pipeline.event_pipeline import EventExtractionPipeline
from src.Event_extractor.models.news import NewsContent
from src.Event_extractor.classifiers.sentiment import KeywordSentimentClassifier

print("🔬 DEMOSTRACIÓN DEL PIPELINE COMPLETO INTEGRADO")
print("=" * 70)

# Crear noticias de ejemplo
noticias = [
    NewsContent(
        id="news_001",
        publication_date=datetime(2024, 7, 1),
        text="""
        El Gobierno español anunció ayer que celebrará un referéndum sobre 
        las nuevas políticas económicas el próximo 15 de agosto. El presidente 
        declaró que esta decisión histórica permitirá a los ciudadanos expresar 
        su opinión sobre las reformas propuestas.
        
        Los sindicatos principales organizarán manifestaciones masivas durante 
        la primera semana de agosto en protesta por las medidas de austeridad. 
        Los líderes sindicales convocaron a todos los trabajadores a participar 
        en las protestas pacíficas.
        """
    ),
    NewsContent(
        id="news_002",
        publication_date=datetime(2024, 6, 15),
        text="""
        El Festival Internacional de Música inaugurará su 25ª edición mañana 
        con un concierto espectacular en la Plaza Mayor. Artistas de renombre 
        mundial actuarán durante tres días consecutivos, del 16 al 18 de junio.
        
        Los organizadores esperan la asistencia de más de 50,000 personas. El 
        alcalde expresó su entusiasmo por este evento cultural que enriquece 
        la vida de la ciudad y atrae turismo internacional.
        """
    ),
    NewsContent(
        id="news_003",
        publication_date=datetime(2024, 8, 20),
        text="""
        Un terremoto de magnitud 6.2 sacudió la región norte el pasado domingo, 
        causando graves daños en infraestructuras y dejando a miles de familias 
        sin hogar. Los servicios de emergencia trabajan intensamente en las 
        labores de rescate.
        
        El gobierno declaró el estado de emergencia y destinó fondos especiales 
        para la reconstrucción. Organizaciones internacionales ofrecieron ayuda 
        humanitaria inmediata para los afectados.
        """
    )
]

print("\n📰 NOTICIAS A PROCESAR:")
print("-" * 70)
for noticia in noticias:
    print(f"\n{noticia.id} ({noticia.publication_date.strftime('%Y-%m-%d')}):")
    print(f"   {noticia.text[:100].strip()}...")

# Crear pipeline con todas las funcionalidades activadas
print("\n" + "=" * 70)
print("🔧 INICIALIZANDO PIPELINE")
print("=" * 70)

pipeline = EventExtractionPipeline(
    reference_date=datetime(2024, 7, 1),
    classify_sentiment=True,
    sentiment_classifier=KeywordSentimentClassifier(),
    min_confidence=0.3
)

print("\n   ✅ Pipeline inicializado con:")
print("      • Extracción de fechas: ✓")
print("      • Clasificación de tipo: ✓")
print("      • Clasificación de sentimiento: ✓")
print("      • Extracción de entidades (SVO): ✓")

# Procesar cada noticia
print("\n" + "=" * 70)
print("📊 EXTRACCIÓN DE EVENTOS")
print("=" * 70)

all_events = []

for noticia in noticias:
    print(f"\n{'─' * 70}")
    print(f"📰 Procesando: {noticia.id}")
    print(f"{'─' * 70}")
    
    eventos = pipeline.extract_events(noticia)
    all_events.extend(eventos)
    
    if not eventos:
        print("   ⚠️  No se encontraron eventos en esta noticia")
        continue
    
    print(f"\n   ✅ {len(eventos)} evento(s) extraído(s):\n")
    
    for i, evento in enumerate(eventos, 1):
        print(f"   📅 Evento {i}:")
        print(f"      Fecha: {evento.date.strftime('%d/%m/%Y')}")
        print(f"      Tipo: {evento.event_type.value} (conf: {evento.confidence:.2%})")
        print(f"      Sentimiento: {evento.sentiment.value} (conf: {evento.sentiment_confidence:.2%})")
        print(f"      Fuente: {evento.source_news_id}")
        
        if evento.entidades_asociadas:
            print(f"\n      🏷️  Entidades relacionadas ({len(evento.entidades_asociadas)}):")
            
            # Agrupar por rol
            subjects = [e for e in evento.entidades_asociadas if e['role'] == 'subject']
            actions = [e for e in evento.entidades_asociadas if e['role'] == 'action']
            objects = [e for e in evento.entidades_asociadas if e['role'] == 'object']
            
            if subjects:
                print(f"\n         Sujetos/Agentes:")
                for ent in subjects[:5]:  # Limitar a 5
                    ent_type = f" [{ent['ent_type']}]" if ent['ent_type'] else ""
                    action = f" → {ent['action']}" if ent['action'] else ""
                    print(f"            • {ent['text']}{ent_type}{action}")
            
            if actions:
                print(f"\n         Acciones principales:")
                for ent in actions[:5]:
                    print(f"            • {ent['text']} ({ent['lemma']})")
            
            if objects:
                print(f"\n         Objetos/Temas:")
                for ent in objects[:5]:
                    ent_type = f" [{ent['ent_type']}]" if ent['ent_type'] else ""
                    print(f"            • {ent['text']}{ent_type}")
        
        print()

# Resumen global
print("\n" + "=" * 70)
print("📈 RESUMEN GLOBAL")
print("=" * 70)

print(f"\n   Total de noticias procesadas: {len(noticias)}")
print(f"   Total de eventos extraídos: {len(all_events)}")

# Estadísticas por tipo
from collections import Counter

tipos = Counter([e.event_type.value for e in all_events])
sentimientos = Counter([e.sentiment.value for e in all_events])

print(f"\n   📊 Distribución por tipo:")
for tipo, count in tipos.most_common():
    print(f"      • {tipo}: {count}")

print(f"\n   😊 Distribución por sentimiento:")
for sent, count in sentimientos.most_common():
    emoji = "😊" if sent == "positive" else "😢" if sent == "negative" else "😐"
    print(f"      {emoji} {sent}: {count}")

# Análisis de entidades
total_entidades = sum(len(e.entidades_asociadas or []) for e in all_events)
print(f"\n   🏷️  Total de entidades extraídas: {total_entidades}")

if total_entidades > 0:
    avg_per_event = total_entidades / len(all_events)
    print(f"   📊 Promedio por evento: {avg_per_event:.1f}")

# Mostrar eventos ordenados por fecha
print("\n" + "=" * 70)
print("📅 LÍNEA DE TIEMPO DE EVENTOS")
print("=" * 70)

sorted_events = sorted(all_events, key=lambda e: e.date)

for evento in sorted_events:
    sent_emoji = "😊" if evento.sentiment.value == "positive" else "😢" if evento.sentiment.value == "negative" else "😐"
    print(f"\n   {evento.date.strftime('%d/%m/%Y')} {sent_emoji} [{evento.event_type.value}]")
    if evento.entidades_asociadas:
        # Mostrar los sujetos principales
        subjects = [e['text'] for e in evento.entidades_asociadas if e['role'] == 'subject']
        if subjects:
            print(f"      Actores: {', '.join(subjects[:3])}")

print("\n" + "=" * 70)
print("✅ DEMOSTRACIÓN COMPLETADA")
print("=" * 70)

print("""
💡 RESUMEN DE FUNCIONALIDADES:

1. ✅ Extracción de fechas múltiples por noticia
2. ✅ Clasificación automática del tipo de evento
3. ✅ Análisis de sentimiento (positivo/negativo/neutral)
4. ✅ Extracción de entidades con roles (sujeto/verbo/objeto)
5. ✅ Generación de eventos completos con toda la información
6. ✅ Métricas de confianza para clasificaciones
7. ✅ Agrupación y análisis estadístico

📚 El pipeline ahora integra todas las funcionalidades desarrolladas
   y genera eventos ricos en información listos para análisis.
""")
