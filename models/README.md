# Aircraft Forecasting Models

Sistema completo de modelos de machine learning para forecasting del número de aeronaves en el espacio aéreo cubano usando datos ATC/ATFM.

## 📊 Información General

Este módulo implementa múltiples modelos de forecasting para predecir el tráfico aéreo cubano:

- **ARIMA/SARIMA**: Modelos estadísticos tradicionales para series temporales
- **Prophet**: Modelo de Facebook optimizado para datos con estacionalidad
- **LSTM**: Redes neuronales recurrentes para forecasting avanzado
- **Ensemble**: Combinación de múltiples modelos para mayor robustez

## 🏗️ Arquitectura

```
models/
├── __init__.py          # Imports principales
├── config.py            # Configuración del sistema
├── data_loader.py       # Carga de datos ATC/ATFM
├── preprocessing.py     # Limpieza y preprocesamiento
├── features.py          # Ingeniería de características
├── model.py             # Implementación de modelos
├── train.py             # Script de entrenamiento
├── evaluate.py          # Script de evaluación
└── README.md           # Este archivo
```

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenamiento de Modelos

```bash
# Entrenar todos los modelos con datos diarios
python models/train.py --data-type daily_atc --models arima prophet lstm ensemble

# Entrenar solo ARIMA con horizonte de predicción de 7 días
python models/train.py --data-type daily_atc --models arima --forecast-horizon 7

# Entrenar y guardar modelos
python models/train.py --save-models --log-level DEBUG
```

### 3. Evaluación de Modelos

```bash
# Evaluar modelos en múltiples horizontes
python models/evaluate.py --horizons 1 7 14 30 --output-dir evaluation_results

# Evaluar solo modelos específicos
python models/evaluate.py --models arima prophet --horizons 1 7
```

## 📈 Uso Programático

### Entrenamiento Básico

```python
from models import (
    ModelConfig,
    ATCAircraftDataLoader,
    AircraftDataPreprocessor,
    AircraftFeatureEngineer,
    AircraftForecaster,
    ARIMAModel,
    ProphetModel
)

# Configuración
config = ModelConfig()

# Cargar y preparar datos
data_loader = ATCAircraftDataLoader(config)
df = data_loader.get_training_data('daily_atc')

preprocessor = AircraftDataPreprocessor(config)
df_processed = preprocessor.preprocess_daily_data(df)

feature_engineer = AircraftFeatureEngineer(config)
df_featured = feature_engineer.create_features(df_processed)
df_featured = feature_engineer.create_lagged_target(df_featured, forecast_horizon=1)

X, y = feature_engineer.select_features_for_model(df_featured)

# Entrenar modelos
forecaster = AircraftForecaster(config)
forecaster.add_model(ARIMAModel(config))
forecaster.add_model(ProphetModel(config))

results = forecaster.train_all_models(X, y)
print(f"Mejor modelo: {forecaster.best_model.name}")
```

### Forecasting

```python
# Realizar predicciones
predictions = forecaster.forecast(X_test, forecast_horizon=7)
print(f"Predicciones para 7 días: {predictions}")

# Forecast futuro (sin datos históricos)
future_predictions = forecaster.forecast(forecast_horizon=30)
print(f"Predicciones futuras: {future_predictions}")
```

## 📊 Datos de Entrada

### Formatos Soportados

1. **Datos Diarios ATC** (`atc_dayatcopsummary_*.csv`)
   - Resúmenes diarios de operaciones de control de tráfico aéreo
   - Target: `total` (aeronaves totales por día)

2. **Datos Horarios ATFM** (`atfm_hourlyaoigroupflights_*.csv`)
   - Vuelos agrupados por hora y área
   - Target: Conteo de GUFIs por hora

3. **Datos Mensuales por Ruta** (`atfm_monthrouteflights_*.csv`)
   - Vuelos mensuales por ruta específica
   - Target: `total` por ruta

### Estructura de Características

El sistema crea automáticamente las siguientes características:

- **Temporales**: día de semana, mes, trimestre, fin de semana
- **Estacionales**: codificación sinusoidal de patrones cíclicos
- **Lags**: Valores anteriores (1, 7, 14, 30 días)
- **Móviles**: Estadísticas móviles (media, std, min, max)
- **Festivos**: Indicadores de días festivos cubanos
- **Tendencia**: Días desde el inicio de la serie

## 🎯 Modelos Disponibles

### ARIMA/SARIMA
- **Ventajas**: Simple, interpretable, bueno para tendencias lineales
- **Uso**: Horizontes cortos (1-7 días)
- **Configuración**: Ajustable order y seasonal_order

### Prophet
- **Ventajas**: Maneja estacionalidad automáticamente, robusto
- **Uso**: Datos con patrones estacionales claros
- **Características**: Detección automática de changepoints

### Random Forest
- **Ventajas**: Robusto a outliers, interpretable, no paramétrico
- **Uso**: Datos tabulares con features complejas
- **Características**: Ensemble de árboles de decisión, importancia de features

### LSTM
- **Ventajas**: Captura patrones complejos no lineales
- **Uso**: Horizontes largos, datos con dependencias temporales complejas
- **Configuración**: Sequence length, unidades ocultas, dropout

### Ensemble
- **Ventajas**: Combina fortalezas de múltiples modelos
- **Uso**: Mayor robustez y precisión general
- **Configuración**: Pesos ajustables por modelo

## 📊 Métricas de Evaluación

- **MAE** (Mean Absolute Error): Error absoluto medio
- **RMSE** (Root Mean Square Error): Raíz del error cuadrático medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio
- **R²**: Coeficiente de determinación

## 🔧 Configuración Avanzada

### Personalización de Features

```python
config = ModelConfig()
config.feature_config = {
    'temporal_features': True,
    'lag_features': [1, 7, 14, 30],
    'rolling_features': [7, 14, 30],
    'seasonal_features': True,
    'holiday_features': True,
    'covid_adjustment': False
}
```

### Configuración de Modelos

```python
config.models['arima'] = {
    'order': (2, 1, 2),
    'seasonal_order': (1, 1, 1, 7)
}

config.models['lstm'] = {
    'sequence_length': 21,  # 3 semanas
    'hidden_units': 128,
    'dropout_rate': 0.3,
    'epochs': 200
}
```

## 📈 Resultados Esperados

### Rendimiento Típico (Datos de 2022-2025)

| Modelo | Horizonte | MAE | RMSE | R² |
|--------|-----------|-----|------|----|
| ARIMA | 1 día | 8-12 | 10-15 | 0.85-0.90 |
| Prophet | 1 día | 7-10 | 9-13 | 0.87-0.92 |
| Random Forest | 1 día | 6-10 | 8-14 | 0.88-0.93 |
| LSTM | 1 día | 6-9 | 8-12 | 0.89-0.94 |
| Ensemble | 1 día | 5-8 | 7-11 | 0.91-0.95 |

*Los valores son aproximados y dependen de la calidad de los datos y configuración.*

## 🚨 Notas Importantes

1. **Datos Faltantes**: El sistema maneja automáticamente gaps en series temporales
2. **Outliers**: Detección y corrección automática de valores atípicos
3. **Estacionalidad**: Considera patrones semanales y mensuales del tráfico aéreo
4. **Festivos**: Incluye calendario cubano de días festivos
5. **COVID**: Ajuste opcional por impacto de restricciones COVID

## 🔍 Troubleshooting

### Problemas Comunes

1. **ImportError**: Asegurar que todas las dependencias estén instaladas
2. **MemoryError**: Reducir sequence_length en LSTM o usar menos features
3. **Poor Performance**: Verificar calidad de datos y ajustar configuración
4. **Convergence Issues**: Ajustar hiperparámetros del modelo

### Logs y Debugging

```bash
# Logging detallado
python models/train.py --log-level DEBUG --log-file training.log

# Evaluar con más métricas
python models/evaluate.py --horizons 1 3 7 14 --output-dir debug_results
```

## 📝 Licencia

Este proyecto está bajo la misma licencia que el repositorio principal.