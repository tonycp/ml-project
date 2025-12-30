---
applyTo: '**'
---
Normalización y extracción de datos relevantes de noticias

Esta tarea implica recibir, limpiar y normalizar información procedente de noticias para obtener insumos estructurados que el modelo pueda usar. El objetivo es transformar texto libre y recortes en datos homogéneos y representativos de eventos relevantes para el tráfico aéreo.

Checklist de pasos y detalles clave:

- Recolección y filtrado inicial:
  - Recopilar el conjunto de noticias relevantes (cobertura aérea, eventos, incidentes, regulaciones).
  - Filtrar por fechas, sectores y tipo de evento para la base final.

- Conversión y limpieza de texto:
  - Eliminar información irrelevante, duplicados, anuncios y formato extraño.
  - Normalizar nombres, ubicaciones y fechas mencionadas en las noticias.
  - Estandarizar vocabulario técnico (tipo de incidente, magnitud del impacto).

- Estructuración y etiquetado:
  - Organizar la información extraída en tablas estructuradas (CSV, JSON) por evento.
  - Añadir campos clave: fecha, sector, tipo de evento, actores involucrados, magnitud.
  - Etiquetar manualmente los eventos importantes si la detección automática no es suficiente.

- Validación y control de calidad:
  - Verificar la coherencia entre eventos registrados y noticias fuente.
  - Realizar muestreo de registros para asegurar calidad y completitud.

- Documentar el workflow:
  - Herramientas usadas, estrategias de extracción, problemas detectados y propuestas de mejora.