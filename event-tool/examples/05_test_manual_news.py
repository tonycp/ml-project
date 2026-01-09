"""
Script interactivo para probar el pipeline con noticias ingresadas manualmente.

Permite ingresar una noticia y ver paso a paso cómo el pipeline la procesa.
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

print("=" * 70)
print("🧪 TEST MANUAL DEL PIPELINE DE EXTRACCIÓN DE EVENTOS")
print("=" * 70)

print("""
Este script te permite probar el pipeline con tus propias noticias.
Ingresa el texto de una noticia y verás todo el proceso de extracción.
""")

# Solicitar texto de la noticia
print("📝 INGRESA EL TEXTO DE LA NOTICIA:")
print("-" * 70)
print("(Escribe o pega el texto. Presiona Enter dos veces para finalizar)\n")

lines = []
empty_line_count = 0

while True:
    try:
        line = input()
        if line == "":
            empty_line_count += 1
            if empty_line_count >= 2:
                break
        else:
            empty_line_count = 0
            lines.append(line)
    except EOFError:
        break

news_text = "\n".join(lines).strip()

if not news_text:
    print("\n❌ No se ingresó ningún texto. Saliendo...")
    sys.exit(1)

# Solicitar fecha de publicación (opcional)
print("\n" + "=" * 70)
print("📅 FECHA DE PUBLICACIÓN")
print("-" * 70)
print("Formato: DD/MM/YYYY o deja en blanco para usar fecha actual")
fecha_input = input("Fecha: ").strip()

if fecha_input:
    try:
        publication_date = datetime.strptime(fecha_input, "%d/%m/%Y")
    except ValueError:
        print("⚠️  Formato inválido, usando fecha actual")
        publication_date = datetime.now()
else:
    publication_date = datetime.now()

print(f"✅ Fecha: {publication_date.strftime('%d/%m/%Y %H:%M:%S')}")

# Crear objeto NewsContent
news_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
news = NewsContent(
    id=news_id,
    publication_date=publication_date,
    text=news_text
)

# Mostrar resumen de la noticia
print("\n" + "=" * 70)
print("📰 NOTICIA INGRESADA")
print("=" * 70)
print(f"\nID: {news.id}")
print(f"Fecha: {news.publication_date.strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Longitud: {len(news.text)} caracteres")
print(f"\nTexto completo:")
print("-" * 70)
print(news.text)
print("-" * 70)

# Inicializar pipeline
print("\n" + "=" * 70)
print("🔧 INICIALIZANDO PIPELINE")
print("=" * 70)

pipeline = EventExtractionPipeline(
    reference_date=publication_date,
    classify_sentiment=True,
    sentiment_classifier=KeywordSentimentClassifier(),
    min_confidence=0.3
)

print("""
✅ Pipeline configurado con:
   • Extracción de fechas
   • Clasificación de tipo de evento (keywords)
   • Análisis de sentimiento (keywords)
   • Extracción de entidades (SVO + NER)
""")

# Procesar la noticia
print("\n" + "=" * 70)
print("⚙️  PROCESANDO NOTICIA...")
print("=" * 70)

print("\n🔍 Paso 1: Preprocesando texto...")
from src.Event_extractor.utils.text_preprocessor import _tokenize_text, get_processed_text

tokens = _tokenize_text(news.text)
print(f"   ✅ {len(tokens)} tokens extraídos")
print(f"   Tokens: {' '.join(tokens[:20])}{'...' if len(tokens) > 20 else ''}")

print("\n🔍 Paso 2: Procesando con spaCy...")
doc = get_processed_text(news.text, force=True)
print(f"   ✅ Documento procesado")
print(f"   Entidades nombradas encontradas: {len(doc.ents)}")
if doc.ents:
    for ent in list(doc.ents)[:10]:
        print(f"      • {ent.text} [{ent.label_}]")
    if len(doc.ents) > 10:
        print(f"      ... y {len(doc.ents) - 10} más")

print("\n🔍 Paso 3: Extrayendo fechas...")
from src.Event_extractor.extractors.date_extractor import DateExtractor
date_extractor = DateExtractor(reference_date=publication_date)
dates = date_extractor.extract_dates(tokens)
print(f"   ✅ {len(dates)} fecha(s) encontrada(s):")
for date in dates:
    print(f"      • {date.strftime('%d/%m/%Y')}")

if not dates:
    print("   ⚠️  Sin fechas, no se pueden extraer eventos")
    print("\n" + "=" * 70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 70)
    sys.exit(0)

print("\n🔍 Paso 4: Clasificando tipo de evento...")
text_to_classify = ' '.join(tokens)
event_type, confidence = pipeline.type_classifier.classify(text_to_classify, threshold=0.3)
print(f"   ✅ Tipo: {event_type.value}")
print(f"   📊 Confianza: {confidence:.1%}")

print("\n🔍 Paso 5: Analizando sentimiento...")
sentiment, sent_conf = pipeline.sentiment_classifier.classify(text_to_classify, threshold=0.3)
emoji = "😊" if sentiment.value == "positive" else "😢" if sentiment.value == "negative" else "😐"
print(f"   {emoji} Sentimiento: {sentiment.value}")
print(f"   📊 Confianza: {sent_conf:.1%}")

print("\n🔍 Paso 6: Extrayendo entidades SVO...")
from src.Event_extractor.utils.text_preprocessor import extract_svo
svo_triples = extract_svo(doc)
print(f"   ✅ {len(svo_triples)} triples SVO encontrados:")
for subj, verb, obj in svo_triples[:5]:
    print(f"      • {subj} → {verb} → {obj}")
if len(svo_triples) > 5:
    print(f"      ... y {len(svo_triples) - 5} más")

# Extraer eventos completos
print("\n" + "=" * 70)
print("📊 EXTRAYENDO EVENTOS COMPLETOS")
print("=" * 70)

eventos = pipeline.extract_events(news)

print(f"\n✅ {len(eventos)} evento(s) extraído(s)\n")

for i, evento in enumerate(eventos, 1):
    print("=" * 70)
    print(f"📅 EVENTO {i}/{len(eventos)}")
    print("=" * 70)
    
    print(f"\n📆 Fecha del evento: {evento.date.strftime('%d de %B de %Y')}")
    print(f"🏷️  Tipo: {evento.event_type.value} (confianza: {evento.confidence:.1%})")
    
    sent_emoji = "😊" if evento.sentiment.value == "positive" else "😢" if evento.sentiment.value == "negative" else "😐"
    print(f"{sent_emoji} Sentimiento: {evento.sentiment.value} (confianza: {evento.sentiment_confidence:.1%})")
    
    print(f"📰 Fuente: {evento.source_news_id}")
    
    if evento.entidades_asociadas:
        print(f"\n🏷️  ENTIDADES RELACIONADAS ({len(evento.entidades_asociadas)} total):")
        
        # Agrupar por rol
        named_entities = [e for e in evento.entidades_asociadas if e['role'] == 'named_entity']
        subjects = [e for e in evento.entidades_asociadas if e['role'] == 'subject']
        actions = [e for e in evento.entidades_asociadas if e['role'] == 'action']
        objects = [e for e in evento.entidades_asociadas if e['role'] == 'object']
        
        if named_entities:
            print(f"\n   👤 Entidades Nombradas ({len(named_entities)}):")
            for ent in named_entities[:10]:
                ent_type = f" [{ent['ent_type']}]" if ent['ent_type'] else ""
                print(f"      • {ent['text']}{ent_type}")
            if len(named_entities) > 10:
                print(f"      ... y {len(named_entities) - 10} más")
        
        if subjects:
            print(f"\n   🎯 Sujetos/Agentes ({len(subjects)}):")
            for ent in subjects[:5]:
                action = f" → {ent['action']}" if ent['action'] else ""
                ent_type = f" [{ent['ent_type']}]" if ent['ent_type'] else ""
                print(f"      • {ent['text']}{ent_type}{action}")
            if len(subjects) > 5:
                print(f"      ... y {len(subjects) - 5} más")
        
        if actions:
            print(f"\n   ⚡ Acciones principales ({len(actions)}):")
            for ent in actions[:5]:
                print(f"      • {ent['text']}")
            if len(actions) > 5:
                print(f"      ... y {len(actions) - 5} más")
        
        if objects:
            print(f"\n   📦 Objetos/Temas ({len(objects)}):")
            for ent in objects[:5]:
                ent_type = f" [{ent['ent_type']}]" if ent['ent_type'] else ""
                print(f"      • {ent['text']}{ent_type}")
            if len(objects) > 5:
                print(f"      ... y {len(objects) - 5} más")
    else:
        print("\n⚠️  No se extrajeron entidades")
    
    print()

# Resumen final
print("=" * 70)
print("✅ PROCESAMIENTO COMPLETADO")
print("=" * 70)

print(f"""
📊 RESUMEN:

   Noticia ID: {news.id}
   Longitud: {len(news.text)} caracteres
   
   Fechas encontradas: {len(dates)}
   Eventos extraídos: {len(eventos)}
   
   Tipo predominante: {event_type.value} ({confidence:.1%})
   Sentimiento: {sentiment.value} ({sent_conf:.1%})
   
   Total entidades: {sum(len(e.entidades_asociadas or []) for e in eventos)}

💡 El pipeline procesó exitosamente tu noticia.
   Los eventos están listos para análisis o almacenamiento.
""")
