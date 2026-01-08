"""
Configuración para los modelos de forecasting de aeronaves.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuración general para los modelos de forecasting."""

    # Rutas de datos
    data_dir: Path = Path("data/ATC csvs")
    atc_daily_file: str = "atc_dayatcopsummary_202512301506.csv"
    atc_daily_acids_file: str = "atc_daylyacids_202512301506.csv"
    atfm_hourly_file: str = "atfm_hourlyaoigroupflights_202512301506.csv"
    atfm_monthly_file: str = "atfm_monthrouteflights_202512301506.csv"
    news_file: str = "eventos.json"

    # Configuración temporal
    date_column: str = "time"
    target_column: str = "total"
    freq: str = "D"  # Frecuencia diaria

    # Split de datos
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1

    # Fechas de entrenamiento/validación
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = "2024-06-01"  # Usar datos hasta junio 2024 para training
    val_end_date: Optional[str] = "2024-12-01"    # Validación hasta diciembre 2024
    test_end_date: Optional[str] = None          # Test con datos más recientes

    # Configuración de modelos
    models: Dict[str, Dict] = None
    
    # Configuración de features
    feature_config: Dict = None
    
    # Configuración de evaluación
    evaluation_config: Dict = None
   
    # Configuración de logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/aircraft_forecasting.log"

    # Configuración de guardado de modelos
    model_save_dir: Path = Path("models/saved_models")
    results_save_dir: Path = Path("models/results")

    # Configuración de datos externos
    external_data: Dict = None

    def __post_init__(self):

        if self.models is None:
            self.models = {
                "arima": {
                    "order": (1, 1, 1),
                    "seasonal_order": (1, 1, 1, 7),  # Estacionalidad semanal
                },
                "prophet": {
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "changepoint_prior_scale": 0.05,
                },
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                },
                "lstm": {
                    "sequence_length": 30,  # 30 días de historia
                    "hidden_units": 64,
                    "dropout_rate": 0.2,
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                },
                "ensemble": {
                    "weights": {"arima": 0.25, "prophet": 0.25, "random_forest": 0.25, "lstm": 0.25}
                }
            }

        if self.feature_config is None:
            self.feature_config = {
                "temporal_features": True,
                "lag_features": [1, 7, 14, 30],  # Lags de 1, 7, 14, 30 días
                "rolling_features": [7, 14, 30],  # Ventanas móviles
                "seasonal_features": True,
                "holiday_features": True,
                "covid_adjustment": True,  # Ajuste por impacto COVID
            }

        if self.evaluation_config is None:
            self.evaluation_config = {
                "metrics": ["mae", "rmse", "mape", "r2"],
                "cv_folds": 5,
                "forecast_horizon": [1, 7, 14, 30],  # Predicciones a 1, 7, 14, 30 días
            }

        if self.external_data is None:
            self.external_data = {
                'weather': {
                    'enabled': True,
                    'data_dir': Path('data/weather'),
                    'stations': ['HAVANA_INTERNATIONAL'],
                    'variables': [
                        'temperature', 'humidity', 'wind_speed', 'wind_direction',
                        'precipitation', 'visibility', 'cloud_cover', 'pressure',
                        'is_storm', 'is_hurricane', 'high_winds', 'heavy_rain',
                        'low_visibility', 'extreme_heat', 'extreme_cold'
                    ]
                },
                'news': {
                    'enabled': True,
                    'data_dir': Path('data/news'),
                    'sources': ['granma', 'juventud_rebelde', 'cuba_debate'],
                    'event_types': ['ACCIDENTE', 'METEOROLOGICO', 'POLITICO', 'SOCIAL', 'INCIDENTE'],
                    'event_features': [
                        'accident_count', 'storm_alert_count', 'political_event_count',
                        'social_event_count', 'incident_count', 'international_event_count',
                        'has_major_event', 'event_impact_score'
                    ]
                },
                'multimodal': {
                    'enabled': True,
                    'include_weather': True,
                    'include_news': True,
                    'feature_selection': 'auto'  # 'auto', 'manual', 'importance'
                }
            }

    def get_data_path(self, filename: str) -> Path:
        """Obtiene la ruta completa a un archivo de datos."""
        return self.data_dir / filename

    def get_model_path(self, model_name: str) -> Path:
        """Obtiene la ruta para guardar un modelo."""
        return self.model_save_dir / f"{model_name}.pkl"

    def get_results_path(self, experiment_name: str) -> Path:
        """Obtiene la ruta para guardar resultados."""
        return self.results_save_dir / f"{experiment_name}_results.json"


# Configuración por defecto
default_config = ModelConfig()