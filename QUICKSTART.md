# ⚡ Guía de Inicio Rápido

## En 5 minutos con Event Extractor

### Paso 1: Instalación (2 min)

```bash
# Instalar el paquete
pip install -e .

# Descargar modelo de spaCy
python -m spacy download es_core_news_sm
```

### Paso 2: Primer uso (3 min)

Crea un archivo `test.py`:

```python
from Event_extractor import EventExtractionPipeline, NewsContent, NewsMetadata
from datetime import datetime

# Crear pipeline
pipeline = EventExtractionPipeline()

# Crear una noticia de ejemplo
metadata = NewsMetadata(
    title="Festival de Música",
    date=datetime(2024, 12, 1),
    source="Diario Local"
)

news = NewsContent(
    text="El gran festival de música se realizará del 10 al 15 de enero de 2025 en el parque central.",
    metadata=metadata,
    title="Festival de Música"
)

# Extraer eventos
events = pipeline.extract_events(news)

# Mostrar resultados
print(f"Eventos encontrados: {len(events)}\n")
for event in events:
    print(f"📅 {event.date.strftime('%d/%m/%Y')}")
    print(f"🏷️  {event.event_type.value}")
    print(f"📰 {event.title}")
    print(f"💯 Confianza: {event.confidence:.0%}\n")
```

Ejecutar:
```bash
python test.py
```

Salida esperada:
```
Eventos encontrados: 2

📅 10/01/2025
🏷️  cultural
📰 Festival de Música
💯 Confianza: 100%

📅 15/01/2025
🏷️  cultural
📰 Festival de Música
💯 Confianza: 100%
```

### ¿Ves? ¡2 eventos separados para el rango de fechas! 🎉

**Nota importante**: El pipeline usa automáticamente `metadata.date` como fecha de referencia para resolver fechas sin año o relativas. Esto previene fechas erróneas.

---

## Comportamiento con reference_date

```bash
# Ver demostración completa del manejo de fechas
python examples/reference_date_demo.py
```

**Regla de oro**: 
- ❌ Sin `reference_date`: Solo fechas explícitas completas
- ✅ Con `reference_date`: Todos los tipos de fechas

El pipeline siempre intenta usar `metadata.date` como referencia. ¡Por eso es importante incluir la fecha de publicación en la metadata!

---

## Ejemplos Incluidos

```bash
# Ejemplo básico completo
python examples/basic_usage.py

# Uso de componentes individuales
python examples/component_usage.py

# Plantillas para cargar tus datos
python examples/data_loader_template.py
```

---

## Próximo Paso

Cuando conozcas el formato de tus noticias, adapta `examples/data_loader_template.py` para cargar tus datos.

¡Listo! Ya puedes usar Event Extractor en tu proyecto. 🚀

Para más detalles: `README.md` | Para instalación: `INSTALL.md` | Para arquitectura: `ARCHITECTURE.md`
