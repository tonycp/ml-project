"""
Comparación de diferentes modelos sklearn entrenados con TASS vs Keyword classifier.

Este ejemplo compara:
1. KeywordSentimentClassifier (baseline)
2. LinearSVC (default en SklearnSentimentClassifier)
3. MultinomialNB (Naive Bayes)
4. LogisticRegression
5. RandomForestClassifier

Todos los modelos sklearn se entrenan con el corpus TASS.
"""

import sys
from pathlib import Path
import time
from collections import defaultdict

# Agregar directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("🔬 COMPARACIÓN DE MODELOS SKLEARN CON CORPUS TASS")
print("=" * 70)

# Verificar dependencias
try:
    import datasets
    import sklearn
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    print(f"✅ datasets instalado (versión {datasets.__version__})")
    print(f"✅ sklearn instalado (versión {sklearn.__version__})")
except ImportError as e:
    print(f"\n❌ Falta instalar: {e}")
    print("\n📦 Para instalar:")
    print("   pip install datasets scikit-learn")
    sys.exit(1)

from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier, KeywordSentimentClassifier
from datasets import load_dataset

print("\n" + "=" * 70)
print("📥 CARGANDO CORPUS TASS")
print("=" * 70)

print("\n📚 Intentando descargar dataset TASS-2019...")

try:
    dataset = load_dataset("mrm8492/tass-2019")
    train_texts = dataset['train']['sentence']
    train_labels = dataset['train']['sentiments']
    test_texts = dataset['test']['sentence']
    test_labels = dataset['test']['sentiments']
    
    print(f"   ✅ Train: {len(train_texts)} ejemplos")
    print(f"   ✅ Test: {len(test_texts)} ejemplos")
    
except Exception as e:
    print(f"   ❌ No se pudo descargar TASS: {e}")
    print("\n   ⚠️  Usando datos de ejemplo sintéticos para demostración...")
    
    # Crear datos sintéticos para demostración
    train_texts = [
        "Me encanta este producto, es excelente",
        "Muy buena calidad, recomendado",
        "Genial, superó mis expectativas",
        "Perfecto, justo lo que necesitaba",
        "Excelente servicio y atención",
        "Pésimo producto, no funciona",
        "Muy malo, no lo recomiendo",
        "Decepcionante, esperaba más",
        "Horrible experiencia, nunca más",
        "Terrible calidad, no vale la pena",
        "El producto llegó bien",
        "Pedido recibido correctamente",
        "Envío estándar",
        "Producto según descripción",
        "Entrega en tiempo estimado",
    ] * 75  # Repetir para tener más datos
    
    train_labels = ["P", "P", "P", "P", "P", "N", "N", "N", "N", "N", "NEU", "NEU", "NEU", "NEU", "NEU"] * 75
    
    test_texts = [
        "Increíble producto, muy satisfecho",
        "Buena compra, lo recomiendo",
        "Malo, no me gustó",
        "Deficiente calidad",
        "Llegó el pedido",
        "Producto como se describe",
    ] * 50
    
    test_labels = ["P", "P", "N", "N", "NEU", "NEU"] * 50
    
    print(f"   ✅ Train: {len(train_texts)} ejemplos sintéticos")
    print(f"   ✅ Test: {len(test_texts)} ejemplos sintéticos")

# Distribución de clases
from collections import Counter
train_dist = Counter(train_labels)
test_dist = Counter(test_labels)

print(f"\n   Distribución train: {dict(train_dist)}")
print(f"   Distribución test:  {dict(test_dist)}")

print("\n" + "=" * 70)
print("🏋️  ENTRENANDO MODELOS SKLEARN")
print("=" * 70)

# Diccionario de modelos a comparar
modelos_config = {
    "LinearSVC": LinearSVC(random_state=42, max_iter=2000),
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

clasificadores = []
tiempos_entrenamiento = {}

for nombre, modelo in modelos_config.items():
    print(f"\n🔧 Entrenando {nombre}...")
    start_time = time.time()
    
    try:
        clf = SklearnSentimentClassifier(model=modelo)
        clf.train(train_texts, train_labels)
        
        elapsed = time.time() - start_time
        tiempos_entrenamiento[nombre] = elapsed
        clasificadores.append((nombre, clf))
        
        print(f"   ✅ Entrenado en {elapsed:.2f}s")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Agregar KeywordSentimentClassifier (sin entrenar)
print(f"\n🔧 KeywordSentimentClassifier...")
keyword_clf = KeywordSentimentClassifier()
clasificadores.append(("Keyword", keyword_clf))
tiempos_entrenamiento["Keyword"] = 0.0
print(f"   ✅ No requiere entrenamiento")

print("\n" + "=" * 70)
print("📊 EVALUACIÓN EN TEST SET")
print("=" * 70)

resultados = defaultdict(lambda: {"correctos": 0, "total": 0, "tiempo_inferencia": 0.0})

print(f"\n🧪 Evaluando en {len(test_texts)} ejemplos...")

# Mapeo de TASS labels a sentiment.value
label_mapping = {
    "P": "positive",
    "N": "negative",
    "NEU": "neutral"
}

for nombre, clf in clasificadores:
    print(f"\n   Evaluando {nombre}...")
    start_time = time.time()
    
    correctos = 0
    for texto, label_esperado in zip(test_texts, test_labels):
        try:
            sentiment, confidence = clf.classify(texto)
            # Convertir label TASS a formato sentiment.value
            label_esperado_normalizado = label_mapping.get(label_esperado, label_esperado.lower())
            if sentiment.value == label_esperado_normalizado:
                correctos += 1
        except Exception as e:
            pass  # Ignorar errores
    
    elapsed = time.time() - start_time
    
    resultados[nombre]["correctos"] = correctos
    resultados[nombre]["total"] = len(test_texts)
    resultados[nombre]["tiempo_inferencia"] = elapsed
    resultados[nombre]["precision"] = (correctos / len(test_texts)) * 100
    
    print(f"      Aciertos: {correctos}/{len(test_texts)}")
    print(f"      Precisión: {resultados[nombre]['precision']:.2f}%")
    print(f"      Tiempo: {elapsed:.2f}s")

print("\n" + "=" * 70)
print("📈 RESUMEN COMPARATIVO")
print("=" * 70)

# Ordenar por precisión
clasificadores_ordenados = sorted(
    resultados.items(),
    key=lambda x: x[1]["precision"],
    reverse=True
)

print("\n🏆 RANKING POR PRECISIÓN:")
print(f"\n{'Modelo':<20} {'Precisión':<12} {'Train Time':<12} {'Infer Time':<12}")
print("-" * 70)

for nombre, stats in clasificadores_ordenados:
    train_time = tiempos_entrenamiento.get(nombre, 0.0)
    precision = stats["precision"]
    infer_time = stats["tiempo_inferencia"]
    
    print(f"{nombre:<20} {precision:>6.2f}%      {train_time:>6.2f}s       {infer_time:>6.2f}s")

print("\n" + "=" * 70)
print("📊 MÉTRICAS DETALLADAS")
print("=" * 70)

# Calcular métricas detalladas por clase para cada modelo
from sklearn.metrics import classification_report

print("\n📋 Evaluación detallada en test set:")

# Mapeo de TASS labels a sentiment.value
label_mapping = {
    "P": "positive",
    "N": "negative",
    "NEU": "neutral"
}

for nombre, clf in clasificadores:
    if nombre == "Keyword":
        continue  # Keyword no es sklearn, no podemos usar classification_report fácilmente
    
    print(f"\n{'─' * 70}")
    print(f"🔍 {nombre}")
    print(f"{'─' * 70}")
    
    try:
        # Predecir todo el test set
        predicciones = []
        for texto in test_texts:
            sentiment, _ = clf.classify(texto)
            predicciones.append(sentiment.value)
        
        # Convertir labels TASS a formato normalizado
        test_labels_normalized = [label_mapping.get(label, label.lower()) for label in test_labels]
        
        # Mostrar classification report
        print(classification_report(
            test_labels_normalized,
            predicciones,
            target_names=["negative", "neutral", "positive"],
            labels=["negative", "neutral", "positive"],
            zero_division=0
        ))
    except Exception as e:
        print(f"   ❌ Error generando reporte: {e}")

print("\n" + "=" * 70)
print("💡 ANÁLISIS DE RESULTADOS")
print("=" * 70)

mejor_modelo = clasificadores_ordenados[0][0]
mejor_precision = clasificadores_ordenados[0][1]["precision"]

print(f"""
🏆 Mejor modelo: {mejor_modelo} ({mejor_precision:.2f}% precisión)

📊 Observaciones:

1. **Precisión en TASS**:
   - Los modelos sklearn están entrenados con tweets (TASS)
   - El dataset de test también son tweets
   - Debería haber mejor rendimiento que en noticias formales

2. **KeywordSentimentClassifier**:
   - No requiere entrenamiento (0s)
   - Diseñado para noticias formales
   - Puede tener menor rendimiento en tweets informales

3. **Tiempo de entrenamiento**:
   - LinearSVC y LogisticRegression: más rápidos
   - RandomForest: más lento pero puede capturar patrones complejos
   - MultinomialNB: muy rápido

4. **Tiempo de inferencia**:
   - Keyword: más rápido (reglas simples)
   - Modelos sklearn: depende del tamaño del vocabulario

💡 RECOMENDACIONES:

- Para tweets: Usar el modelo sklearn con mejor precisión
- Para noticias: Usar KeywordSentimentClassifier
- Para balance velocidad/precisión: MultinomialNB o LogisticRegression
- Para máxima precisión: El que tenga mejor score en tu dominio
""")

print("\n" + "=" * 70)
print("💾 GUARDAR MEJOR MODELO")
print("=" * 70)

print(f"""
Para guardar el mejor modelo ({mejor_modelo}):

from Event_extractor.classifiers.sentiment import SklearnSentimentClassifier
from sklearn.{modelos_config[mejor_modelo].__class__.__module__} import {mejor_modelo}

# Entrenar
clf = SklearnSentimentClassifier(model={mejor_modelo}())
clf.train(train_texts, train_labels)

# Guardar
clf.save_model("models/tass_{mejor_modelo.lower()}.pkl")

# Cargar después
clf = SklearnSentimentClassifier.load_model("models/tass_{mejor_modelo.lower()}.pkl")
""")

print("\n" + "=" * 70)
print("✅ COMPARACIÓN COMPLETADA")
print("=" * 70)
