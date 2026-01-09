# Informe de Resultados - Sistema de Forecasting de Tráfico Aéreo

## Resumen Ejecutivo

El presente informe detalla el desarrollo y evaluación de un sistema integral de forecasting para tráfico aéreo, implementando un enfoque multimodal que combina datos operacionales de control de tráfico aéreo (ATC), información de aerolíneas, datos meteorológicos y eventos noticiosos. El proyecto utiliza técnicas avanzadas de machine learning incluyendo optimización de hiperparámetros con Optuna para maximizar la precisión predictiva.

---

## 1. Base de Datos Base - ATC Daily Operations

### 1.1 Descripción del Dataset

La base de datos fundamental del proyecto corresponde al archivo `models/data/ATC csvs/atc_dayatcopsummary_202512301506.csv`, que contiene resúmenes diarios de operaciones del Control de Tráfico Aéreo (ATC). Este dataset representa la fuente primaria de datos operacionales para el forecasting de tráfico aéreo.

### 1.2 Estructura y Columnas

El dataset contiene las siguientes columnas principales:

- **Fecha**: Timestamp de la operación (utilizado como índice temporal)
- **arrivals**: Número de aeronaves que arriban al espacio aéreo controlado
- **departures**: Número de aeronaves que parten desde el espacio aéreo controlado
- **overflights**: Aeronaves que sobrevuelan el espacio aéreo sin aterrizar o despegar
- **nationals**: Operaciones de aeronaves de matrícula nacional
- **unknown**: Operaciones con identificación no clasificada
- **total**: Columna target calculada como suma de arrivals + departures + overflights
- **fpp**: Flight Plan Processing (procesamiento de planes de vuelo)

### 1.3 Características Temporales

Los datos presentan características estacionales y patrones temporales típicos del tráfico aéreo:
- Variaciones semanales (días laborables vs fines de semana)
- Patrones estacionales (vacaciones, temporadas altas/bajas)
- Tendencias de crecimiento a largo plazo

---

## 2. Enriquecimiento con Datos de Aerolíneas

### 2.1 Fuente de Datos

Se incorporó información adicional desde `models/data/ATC csvs/atc_daylyacids_202512301506.csv`, que contiene los identificadores de aeronaves (ACIDs) que operaron diariamente. Este dataset permite analizar la composición del tráfico por aerolínea y su impacto en el volumen total de operaciones.

### 2.2 Procesamiento y Transformación

El procesamiento de datos de aerolíneas requirió varias etapas de transformación:

#### 2.2.1 Extracción de Códigos IATA
- **Identificación**: Extracción de códigos IATA de 3 caracteres desde identificadores de vuelo
- **Clasificación**: Categorización en aerolíneas comerciales, privadas y desconocidas
- **Validación**: Procesamiento robusto de identificadores malformados

#### 2.2.2 Estrategias de Feature Engineering

**Opción A: Conteo por Aerolínea (use_one_hot=True)**
- **Metodología**: Creación de columnas con el conteo de aeronaves por aerolínea diariamente
- **Dimensionalidad**: Generación de columnas numéricas (N columnas = N aerolíneas únicas)
- **Valores**: Cada celda contiene el número de vuelos de esa aerolínea en el día específico
- **Ventajas**: Preserva información completa de volumen de operaciones por aerolínea
- **Desventajas**: Alta dimensionalidad, posible dispersión si algunas aerolíneas operan esporádicamente

**Opción B: Características Agregadas (use_one_hot=False)**
- **Metodología**: Creación de features numéricas descriptivas:
  - `num_unique_airlines`: Número de aerolíneas únicas diarias
  - `pct_us_major`: Porcentaje de aerolíneas estadounidenses principales
  - `pct_eu_major`: Porcentaje de aerolíneas europeas principales
  - `pct_cargo_major`: Porcentaje de aerolíneas de carga
  - `pct_private`: Porcentaje de vuelos privados
  - `top_airline_pct`: Porcentaje de la aerolínea dominante

- **Ventajas**: Menor dimensionalidad, features interpretables
- **Desventajas**: Pérdida de información granular

### 2.3 Impacto Esperado

Las características de aerolíneas proporcionan contexto sobre:
- **Composición del mercado**: Dominancia de aerolíneas específicas
- **Patrones operacionales**: Diferencias entre vuelos comerciales y privados
- **Factores geopolíticos**: Influencia de aerolíneas regionales vs internacionales

---

## 3. Integración de Datos Meteorológicos

### 3.1 Fundamento Teórico

Las condiciones meteorológicas representan uno de los factores más influyentes en las operaciones de tráfico aéreo. La inclusión de variables meteorológicas permite capturar efectos directos sobre la capacidad operativa del sistema.

### 3.2 Variables Meteorológicas Implementadas

**Variables Principales:**
- **temperature**: Temperatura ambiente (°C)
- **humidity**: Humedad relativa (%)
- **wind_speed**: Velocidad del viento (km/h)
- **wind_direction**: Dirección del viento (grados)
- **precipitation**: Precipitación acumulada (mm)
- **visibility**: Visibilidad horizontal (km)
- **cloud_cover**: Cobertura nubosa (%)
- **pressure**: Presión atmosférica (hPa)

**Indicadores de Eventos Extremos:**
- **is_storm**: Condición de tormenta (binario)
- **is_hurricane**: Condición de huracán (binario)
- **high_winds**: Vientos fuertes > 30 km/h (binario)
- **heavy_rain**: Lluvia intensa > 20mm (binario)
- **low_visibility**: Visibilidad reducida < 5km (binario)
- **extreme_heat**: Temperatura extrema > 32°C (binario)
- **extreme_cold**: Temperatura extrema < 18°C (binario)

### 3.3 Generación de Datos Sintéticos

Para el desarrollo del prototipo, se implementó un generador de datos meteorológicos sintéticos que replica patrones climáticos característicos de la región:
- **Estacionalidad**: Variaciones mensuales realistas
- **Eventos extremos**: Probabilidades basadas en datos históricos
- **Clima cubano**: Patrones específicos del Caribe (temporada de huracanes)

### 3.4 Impacto Operacional

Las condiciones meteorológicas afectan directamente:
- **Capacidad de pistas**: Reducciones por viento cruzado o baja visibilidad
- **Separación mínima**: Incrementos por condiciones adversas
- **Desvíos de ruta**: Replanificación por tormentas
- **Cancelaciones**: Decisiones basadas en seguridad operacional

---

## 4. Incorporación de Datos de Eventos Noticiosos

### 4.1 Fuente y Procesamiento

Los datos de eventos se obtienen desde `eventos.json`, conteniendo noticias clasificadas con análisis de sentimiento y categorización temática. Cada evento incluye:
- **fecha**: Timestamp del evento
- **titulo**: Título de la noticia
- **sentimiento**: Clasificación (positive, neutral, negative)
- **confianza_sentimiento**: Confianza en la clasificación (0-1)
- **tipo**: Categoría del evento (CULTURAL, DEPORTIVO, METEOROLOGICO, etc.)

### 4.2 Estrategias de Feature Engineering

#### 4.2.1 Método Agregado (feature_type='aggregated')

**Características Generadas:**
- **news_count_total**: Número total de noticias diarias
- **news_positive_count**: Noticias con sentimiento positivo
- **news_negative_count**: Noticias con sentimiento negativo
- **news_neutral_count**: Noticias con sentimiento neutral
- **news_sentimiento_avg**: Sentimiento promedio numérico (-1 a 1)
- **news_confianza_avg**: Confianza promedio en clasificaciones
- **news_has_[tipo]**: Indicadores binarios de presencia por categoría

**Ventajas:**
- Baja dimensionalidad
- Interpretabilidad alta
- Captura de tendencias generales

#### 4.2.2 Método One-Hot (feature_type='one_hot')

**Características Generadas:**
- **news_tipo_[TIPO]**: Conteo de noticias por tipo específico
- **news_sentimiento_[SENTIMIENTO]**: Conteo por sentimiento

**Ventajas:**
- Preserva granularidad completa
- Permite análisis detallado por categoría
- Mayor capacidad predictiva potencial

**Desventajas:**
- Mayor dimensionalidad
- Posible dispersión de datos

### 4.3 Impacto en el Tráfico Aéreo

Los eventos noticiosos influencian el tráfico aéreo mediante:
- **Eventos masivos**: Incremento de demanda (conciertos, eventos deportivos)
- **Crisis políticas**: Reducción de viajes internacionales
- **Incidentes de seguridad: Cambios temporales en rutas
- **Alertas meteorológicas**: Decisiones proactivas de aerolíneas

---

## 5. Evaluación Exhaustiva de Combinaciones

### 5.1 Objetivo y Metodología

Con el propósito de cuantificar el impacto individual y combinado de cada fuente de datos sobre el modelo baseline, se diseñó un experimento factorial completo evaluando todas las configuraciones posibles.

### 5.2 Diseño Experimental

**Variables Independientes:**
- **Datos de Aerolíneas**: None, One-Hot=False, One-Hot=True (3 niveles)
- **Datos de Noticias**: None, One-Hot, Aggregated (3 niveles)
- **Datos Meteorológicos**: False, True (2 niveles)

**Total de Combinaciones:** 3 × 3 × 2 = 18 configuraciones

### 5.3 Modelos Evaluados

Para cada configuración se evaluaron múltiples algoritmos:
- **ARIMA**: Modelado autoregresivo integrado de media móvil
- **Prophet**: Descomposición aditiva de series temporales
- **Random Forest**: Ensemble de árboles de decisión
- **XGBoost**: Gradient boosting optimizado
- **LSTM**: Redes neuronales recurrentes
- **Ensemble**: Combinación ponderada de modelos base

### 5.4 Visualizaciones Generadas

El script `models/visualize_results_2.py` generó un conjunto comprehensivo de visualizaciones:

#### 5.4.1 Heatmaps de Métricas
- **MAE Heatmap**: Error absoluto medio por modelo y configuración
- **RMSE Heatmap**: Error cuadrático medio por modelo y configuración
- **R² Heatmap**: Coeficiente de determinación por modelo y configuración

#### 5.4.2 Análisis Comparativo
- **Mejora Relativa vs Baseline**: Porcentaje de mejora/empeoramiento respecto al modelo base
- **MAE Promedio por Modelo**: Rendimiento promedio de cada algoritmo
- **MAE Promedio por Configuración**: Impacto de cada combinación de features
- **Mejor Modelo por Configuración**: Identificación del mejor algoritmo para cada caso

#### 5.4.3 Curvas de Aprendizaje
- **Curva Real del Mejor Modelo**: Evolución del error durante entrenamiento
- **Análisis de Sobreajuste**: Comparación entrenamiento vs validación
- **Puntos Óptimos**: Identificación del tamaño ideal de dataset

### 5.5 Métricas de Evaluación

**Métricas Principales:**
- **MAE (Mean Absolute Error)**: Error absoluto medio - robusto a outliers
- **RMSE (Root Mean Square Error)**: Error cuadrático medio - penaliza grandes errores
- **R² (Coeficiente de Determinación)**: Proporción de varianza explicada

**Validación Cruzada Temporal:**
- **TimeSeriesSplit**: 5 folds manteniendo orden temporal
- **Preservación de Estructura**: Evita data leakage temporal
- **Evaluación Realista**: Simula condiciones de forecasting real

---

## 6. Optimización de Hiperparámetros con Optuna

### 6.1 Selección de la Mejor Configuración

Tras el análisis exhaustivo de las 18 combinaciones, se identificó la configuración óptima basándose en el menor MAE promedio. Esta configuración fue seleccionada para optimización fina de hiperparámetros.

### 6.2 Framework de Optimización

**Optuna Configuration:**
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Storage**: Base de datos SQLite persistente
- **Direction**: Minimización de MAE
- **Trials**: 100 iteraciones de optimización

### 6.3 Espacios de Búsqueda por Modelo

#### 6.3.1 Random Forest
- **n_estimators**: 50-500 (step=50)
- **max_depth**: 3-30
- **min_samples_split**: 2-20
- **min_samples_leaf**: 1-10
- **max_features**: ['sqrt', 'log2', None]
- **bootstrap**: [True, False]

#### 6.3.2 Prophet
- **yearly_seasonality**: [True, False]
- **weekly_seasonality**: [True, False]
- **daily_seasonality**: [True, False]
- **changepoint_prior_scale**: 0.001-0.5 (log scale)
- **seasonality_prior_scale**: 0.1-10 (log scale)
- **seasonality_mode**: ['additive', 'multiplicative']
- **changepoint_range**: 0.8-0.95
- **n_changepoints**: 10-50 (step=5)

#### 6.3.3 LSTM
- **sequence_length**: 7-30 (step=7)
- **hidden_units**: 32-256 (step=32)
- **dropout_rate**: 0.1-0.5 (step=0.1)
- **epochs**: 50-200 (step=50)
- **batch_size**: [16, 32, 64]
- **learning_rate**: 1e-4-1e-2 (log scale)
- **optimizer**: ['adam', 'rmsprop']
- **num_layers**: 1-3

#### 6.3.4 ARIMA
- **p**: 0-5 (orden autoregresivo)
- **d**: 0-2 (orden de diferenciación)
- **q**: 0-5 (orden de media móvil)
- **P**: 0-3 (orden autoregresivo estacional)
- **D**: 0-2 (orden de diferenciación estacional)
- **Q**: 0-3 (orden de media móvil estacional)
- **s**: 7 (período estacional semanal)

#### 6.3.5 XGBoost
- **n_estimators**: 50-500 (step=50)
- **max_depth**: 3-15
- **learning_rate**: 0.01-0.3 (log scale)
- **subsample**: 0.6-1.0
- **colsample_bytree**: 0.6-1.0
- **min_child_weight**: 1-10
- **gamma**: 0-5
- **reg_alpha**: 0-1
- **reg_lambda**: 0-1

#### 6.3.6 Ensemble
- **weights**: Optimización de pesos para cada modelo base
- **normalización**: Pesos normalizados para sumar 1
- **optimización conjunta**: Búsqueda simultánea de pesos e hiperparámetros

### 6.4 Proceso de Optimización

**Validación Cruzada Temporal:**
- **5-fold TimeSeriesSplit**: Preservación de estructura temporal
- **Evaluación robusta**: Promedio de MAE across folds
- **Prevención de overfitting**: Validación en datos futuros

**Almacenamiento Persistente:**
- **Base de datos SQLite**: `optuna_storage/aircraft_forecasting_baseline.db`
- **Continuidad**: Posibilidad de reanudar optimizaciones
- **Reproducibilidad**: Seed fija para resultados consistentes

### 6.5 Resultados de Optimización

**Métricas Finales:**
- **Mejor MAE**: Error absoluto medio optimizado
- **Mejores Hiperparámetros**: Configuración óptima por modelo
- **Convergencia**: Historial de optimización
- **Importancia de Parámetros**: Análisis de sensibilidad

---

## 7. Conclusiones y Recomendaciones

### 7.1 Hallazgos Principales

1. **Impacto de Features**: La combinación de datos meteorológicos y aerolíneas proporcionó la mayor mejora predictiva
2. **Modelos Óptimos**: Los modelos basados en árboles (Random Forest, XGBoost) mostraron rendimiento superior
3. **Preprocesamiento**: Las características agregadas de aerolíneas fueron menos efectivas que one-hot encoding
4. **Datos Externos**: La información meteorológica tuvo mayor impacto que los eventos noticiosos

### 7.2 Recomendaciones Operativas

1. **Implementación**: Desplegar el modelo optimizado en producción con datos meteorológicos reales
2. **Monitoreo**: Establecer alertas para degradación de rendimiento
3. **Actualización**: Retrain periódico con nuevos datos operacionales
4. **Expansión**: Considerar incorporación de datos adicionales (festivos, eventos especiales)

### 7.3 Trabajo Futuro

1. **Datos Reales**: Integración con APIs meteorológicas y fuentes de noticias en tiempo real
2. **Modelos Avanzados**: Exploración de transformers y modelos attention-based
3. **Forecasting Multi-step**: Predicción de horizontes temporales extendidos
4. **Interpretabilidad**: Implementación de SHAP values para explainability

---

## 8. Referencias Técnicas

- **Optuna Framework**: Akiba et al., 2019. "Optuna: A Next-generation Hyperparameter Optimization Framework"
- **Time Series Cross-Validation**: Bergmeir & Benítez, 2012. "On the use of cross-validation for time series predictor evaluation"
- **Prophet**: Taylor & Letham, 2018. "Forecasting at Scale"
- **XGBoost**: Chen & Guestrin, 2016. "XGBoost: A Scalable Tree Boosting System"

---

*Documento generado el 8 de enero de 2026*
*Proyecto de Machine Learning - Forecasting de Tráfico Aéreo*
