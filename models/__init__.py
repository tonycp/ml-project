"""
Aircraft Forecasting Models - Modelos de predicción de tráfico aéreo cubano

Este módulo contiene implementaciones de modelos de machine learning para
forecasting del número de aeronaves en el espacio aéreo cubano usando datos
ATC (Air Traffic Control) y ATFM (Air Traffic Flow Management).

Modelos implementados:
- ARIMA/SARIMA: Modelos estadísticos tradicionales
- Prophet: Modelo de Facebook para series temporales
- LSTM: Redes neuronales recurrentes para forecasting

Datasets utilizados:
- atc_dayatcopsummary: Resúmenes diarios de operaciones ATC
- atfm_hourlyaoigroupflights: Vuelos horarios por área
- atfm_monthrouteflights: Vuelos mensuales por ruta
"""

__version__ = "0.1.0"

from .config import ModelConfig
from .data_loader import ATCAircraftDataLoader
from .external_data_loaders import WeatherDataLoader, NewsDataLoader, MultiModalDataLoader
from .preprocessing import AircraftDataPreprocessor
from .features import AircraftFeatureEngineer
from .model import (
    ARIMAModel,
    ProphetModel,
    RandomForestModel,
    LSTMModel,
    EnsembleModel,
    AircraftForecaster
)

__all__ = [
    # Configuración
    'ModelConfig',

    # Carga de datos
    'ATCAircraftDataLoader',
    'WeatherDataLoader',
    'NewsDataLoader',
    'MultiModalDataLoader',

    # Preprocesamiento
    'AircraftDataPreprocessor',

    # Features
    'AircraftFeatureEngineer',

    # Modelos
    'ARIMAModel',
    'ProphetModel',
    'RandomForestModel',
    'LSTMModel',
    'EnsembleModel',
    'AircraftForecaster'
]