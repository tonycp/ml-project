"""
Ejemplo de uso del clasificador Sklearn/TASS para sentiment analysis en español.

Este ejemplo muestra:
1. Cómo entrenar con el corpus TASS
2. Cómo usar SklearnSentimentClassifier
3. Comparación con KeywordSentimentClassifier
4. Uso en el pipeline principal
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("🤖 CLASIFICADOR SKLEARN/TASS PARA SENTIMENT ANALYSIS")
print("=" * 70)

# Verificar si datasets está instalado
try:
    import datasets
    import sklearn
    print(f"✅ datasets instalado (versión {datasets.__version__})")
    print(f"✅ sklearn instalado (versión {sklearn.__version__})")
except ImportError as e:
    print(f"\n❌ Falta instalar: {e}")
    print("\n📦 Para instalar:")
    print("   pip install datasets scikit-learn")
    sys.exit(1)

print("\n" + "=" * 70)
print("🔧 ENTRENANDO CLASIFICADOR CON CORPUS TASS")
print("=" * 70)

from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier, KeywordSentimentClassifier
from datasets import load_dataset

print("\n📥 Cargando corpus TASS...")
print("   Corpus: mrm8488/tass-2019 (tweets en español)")

try:
    # Cargar dataset TASS
    dataset = load_dataset("mrm8488/tass-2019")
    print(f"   ✅ Corpus cargado: {len(dataset['train'])} train, {len(dataset['test'])} test")
    
    # Preparar datos de entrenamiento (usar columnas correctas)
    train_texts = dataset['train']['sentence']
    train_labels = dataset['train']['sentiments']
    
    print("\n🏋️  Entrenando clasificador Sklearn...")
    sklearn_clf = SklearnSentimentClassifier()
    sklearn_clf.train(train_texts, train_labels)
    print("   ✅ Clasificador entrenado")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n💡 Puede que no tengas conexión a internet para descargar TASS")
    print("   Intentando cargar modelo pre-entrenado...")
    
    try:
        sklearn_clf = SklearnSentimentClassifier.load_model("models/sklearn_tass_sentiment.pkl")
        print("   ✅ Modelo pre-entrenado cargado")
    except Exception as e2:
        print(f"   ❌ No se pudo cargar modelo: {e2}")
        print("\n⚠️  Usando KeywordSentimentClassifier como fallback")
        sklearn_clf = None

# Cargar keyword para comparar
keyword_clf = KeywordSentimentClassifier()
print("   ✅ KeywordSentimentClassifier cargado para comparación")

# Ejemplos de texto en español
ejemplos = [
    {
        "texto": "El festival de música fue un éxito rotundo, miles de personas disfrutaron de un espectáculo increíble",
        "categoria": "positive"  # En inglés para comparar con EventSentiment
    },
    {
        "texto": "Cancelan el concierto debido a problemas técnicos, los fans están muy decepcionados",
        "categoria": "negative"
    },
    {
        "texto": "Se anuncia la fecha de la conferencia de tecnología para el próximo mes",
        "categoria": "neutral"
    },
    {
        "texto": "Terrible accidente en la autopista causa múltiples víctimas y heridos graves",
        "categoria": "negative"
    },
    {
        "texto": "Inauguran nuevo parque con juegos para niños y áreas verdes, gran alegría en la comunidad",
        "categoria": "positive"
    },
    {
        "texto": "La empresa reporta pérdidas millonarias y anuncia despidos masivos",
        "categoria": "negative"
    },
    {
        "texto": "Gran victoria del equipo nacional, gol en el último minuto",
        "categoria": "positive"
    },
    {
        "texto": "Protesta masiva contra las nuevas medidas económicas del gobierno",
        "categoria": "negative"
    },
    {
        "texto": "El comité se reunirá el próximo jueves para discutir el presupuesto",
        "categoria": "neutral"
    },
    {
        "texto": "Record histórico de ventas supera todas las expectativas",
        "categoria": "positive"
    }
]

print("\n" + "=" * 70)
print("📊 CLASIFICACIÓN CON SKLEARN/TASS")
print("=" * 70)

aciertos_sklearn = 0
aciertos_keyword = 0

for idx, ejemplo in enumerate(ejemplos, 1):
    texto = ejemplo["texto"]
    esperado = ejemplo["categoria"]
    
    print(f"\n📝 Ejemplo {idx}: {texto[:60]}...")
    print(f"   Sentimiento esperado: {esperado}")
    
    print(f"\n   Resultados:")
    
    # Clasificar con Sklearn si está disponible
    if sklearn_clf is not None:
        sentiment_sklearn, conf_sklearn = sklearn_clf.classify(texto)
        correcto_sklearn = "✅" if sentiment_sklearn.value.lower() == esperado.lower() else "❌"
        if sentiment_sklearn.value.lower() == esperado.lower():
            aciertos_sklearn += 1
        print(f"      {correcto_sklearn} Sklearn/TASS: {sentiment_sklearn.value:8s} (conf: {conf_sklearn:.2%})")
    
    # Clasificar con Keyword
    sentiment_keyword, conf_keyword = keyword_clf.classify(texto)
    correcto_keyword = "✅" if sentiment_keyword.value.lower() == esperado.lower() else "❌"
    if sentiment_keyword.value.lower() == esperado.lower():
        aciertos_keyword += 1
    print(f"      {correcto_keyword} Keyword:       {sentiment_keyword.value:8s} (conf: {conf_keyword:.2%})")

# Resumen
print("\n" + "=" * 70)
print("📈 RESUMEN DE PRECISIÓN")
print("=" * 70)

total = len(ejemplos)

if sklearn_clf is not None:
    precision_sklearn = (aciertos_sklearn / total) * 100
    print(f"\n   Sklearn/TASS:")
    print(f"      Aciertos: {aciertos_sklearn}/{total}")
    print(f"      Precisión: {precision_sklearn:.1f}%")

precision_keyword = (aciertos_keyword / total) * 100
print(f"\n   Keyword:")
print(f"      Aciertos: {aciertos_keyword}/{total}")
print(f"      Precisión: {precision_keyword:.1f}%")

if sklearn_clf is not None:
    print(f"\n   Mejora: +{precision_sklearn - precision_keyword:.1f}% con Sklearn/TASS")

# Uso en pipeline
print("\n" + "=" * 70)
print("🔄 USO EN PIPELINE")
print("=" * 70)

from Event_extractor.pipeline.event_pipeline import EventExtractionPipeline
from Event_extractor.models.news import NewsContent
from datetime import datetime

# Noticia de ejemplo
noticia = NewsContent(
    id="news_001",
    publication_date=datetime(2024, 7, 15),
    text="""
    El esperado Festival de Rock 2024 ha sido cancelado en el último momento
    debido a una fuerte tormenta que se aproxima a la ciudad. Los organizadores
    expresaron su profunda tristeza por la decisión.
    
    Miles de fans están decepcionados, ya que esperaban este evento desde hace
    meses. Los boletos serán reembolsados completamente.
    
    Se planea reprogramar el evento para septiembre, pero aún no hay
    confirmación oficial.
    """
)

print("\n📰 Noticia de ejemplo:")
print(f"   Título: Festival cancelado")
print(f"   Fecha: {noticia.publication_date.strftime('%Y-%m-%d')}")

# Crear pipeline con Sklearn si está disponible
if sklearn_clf is not None:
    print("\n🔧 Creando pipeline con Sklearn/TASS...")
    pipeline = EventExtractionPipeline(
        reference_date=datetime(2024, 7, 1),
        classify_sentiment=True,
        sentiment_classifier=sklearn_clf
    )
    classifier_name = "Sklearn/TASS"
else:
    print("\n🔧 Creando pipeline con KeywordSentimentClassifier...")
    pipeline = EventExtractionPipeline(
        reference_date=datetime(2024, 7, 1),
        classify_sentiment=True,
        sentiment_classifier=keyword_clf
    )
    classifier_name = "Keyword"

print("   ✅ Pipeline creado")

# Extraer eventos
print("\n🔍 Extrayendo eventos...")
eventos = pipeline.extract_events(noticia)

print(f"\n   Eventos encontrados: {len(eventos)}")

for idx, evento in enumerate(eventos, 1):
    print(f"\n   Evento {idx}:")
    print(f"      Fecha: {evento.date.strftime('%Y-%m-%d')}")
    print(f"      Tipo: {evento.event_type.value}")
    print(f"      Sentimiento: {evento.sentiment.value} (conf: {evento.sentiment_confidence:.2%})")
    print(f"      Descripción: {evento.description[:80]}...")

print("\n" + "=" * 70)
print("✅ EJEMPLO COMPLETADO")
print("=" * 70)

print("\n💡 CÓMO USAR EN TU CÓDIGO:")
print("""
# 1. Instalar dependencias
pip install datasets scikit-learn

# 2. Entrenar clasificador con TASS
from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier
from datasets import load_dataset

dataset = load_dataset("mrm8488/tass-2019")
clf = SklearnSentimentClassifier()
clf.train(dataset['train']['sentence'], dataset['train']['sentiments'])

# 3. Clasificar texto
sentiment, confidence = clf.classify("Tu texto en español aquí")
print(f"Sentimiento: {sentiment.value} (confianza: {confidence:.2%})")

# 4. Guardar modelo entrenado
clf.save_model("models/mi_modelo_tass.pkl")

# 5. Cargar modelo guardado
clf = SklearnSentimentClassifier.load_model("models/mi_modelo_tass.pkl")

# 6. Usar en pipeline
from Event_extractor.pipeline.event_pipeline import EventExtractionPipeline

pipeline = EventExtractionPipeline(
    sentiment_classifier=clf
)

eventos = pipeline.extract_events(news_content)
""")

print("\n📚 ALTERNATIVAS SIN ENTRENAMIENTO:")
print("""
# Clasificador basado en keywords (sin dependencias)
from Event_extractor.classifiers.sentiment import KeywordSentimentClassifier
clf = KeywordSentimentClassifier()
sentiment, conf = clf.classify("Texto positivo")
""")
