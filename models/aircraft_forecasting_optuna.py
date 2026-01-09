# aircraft_forecasting_optuna.py

# Importaciones básicas
import sys
import os
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Añadir el directorio padre al path
sys.path.append(str(Path().cwd().parent))

# Importar bibliotecas de análisis
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Importar módulos personalizados
from models import (
    ModelConfig,
    ATCAircraftDataLoader,
    AircraftDataPreprocessor,
    AircraftFeatureEngineer,
    ARIMAModel,
    ProphetModel,
    RandomForestModel,
    LSTMModel,
    XGBoostModel,
    EnsembleModel,
    AircraftForecaster
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración
config = ModelConfig()
RANDOM_STATE = 42

# Configuración de almacenamiento para Optuna
import sqlite3
from pathlib import Path

# Crear directorio para almacenamiento si no existe
storage_dir = Path("optuna_storage")
storage_dir.mkdir(exist_ok=True)

# Configurar el almacenamiento
storage_name = f"sqlite:///{storage_dir}/aircraft_forecasting.db"
study_name = "aircraft_forecasting_study"

def load_and_prepare_data(forecast_horizon=7):
    """Carga y prepara los datos para el entrenamiento."""
    logger.info("Cargando y preparando datos...")
    
    # 1. Cargar datos
    data_loader = ATCAircraftDataLoader(config)
    df = data_loader.get_training_data('daily_atc')
    
    # 2. Preprocesar datos
    preprocessor = AircraftDataPreprocessor(config)
    df_processed = preprocessor.preprocess_daily_data(df)
    
    # 3. Ingeniería de características
    feature_engineer = AircraftFeatureEngineer(config)
    df_featured = feature_engineer.create_features(df_processed)
    df_featured = feature_engineer.create_lagged_target(
        df_featured, 
        forecast_horizon=forecast_horizon
    )
    
    # 4. Preparar datos para modelado
    X, y = feature_engineer.select_features_for_model(df_featured)
    
    logger.info(f"Datos preparados: {len(X)} muestras, {len(X.columns)} características")
    
    return X, y, df_featured

def objective(trial, X, y):
    """
    Función objetivo para la optimización con Optuna.
    
    Args:
        trial: Objeto de prueba de Optuna
        X: Características de entrenamiento
        y: Variable objetivo
        
    Returns:
        Error de validación (MAE) a minimizar
    """
    # Selección del modelo
    algorithm = trial.suggest_categorical('algorithm', ['random_forest', 'prophet', 'lstm', 'arima', 'xgboost'])

    # Crear una copia de la configuración
    trial_config = ModelConfig()

    if algorithm == 'random_forest':
        # Espacio de búsqueda para Random Forest
        trial_config.models['random_forest'] = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('rf_bootstrap', [True, False]),
            'random_state': RANDOM_STATE
        }
        
        model = RandomForestModel(trial_config)
        
    elif algorithm == 'prophet':
        # Espacio de búsqueda para Prophet
        trial_config.models['prophet'] = {
            'yearly_seasonality': trial.suggest_categorical('prophet_yearly', [True, False]),
            'weekly_seasonality': trial.suggest_categorical('prophet_weekly', [True, False]),
            'daily_seasonality': trial.suggest_categorical('prophet_daily', [True, False]),
            'changepoint_prior_scale': trial.suggest_float('prophet_changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('prophet_seasonality_prior_scale', 0.1, 10, log=True),
            'seasonality_mode': trial.suggest_categorical('prophet_seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_range': trial.suggest_float('prophet_changepoint_range', 0.8, 0.95),
            'n_changepoints': trial.suggest_int('prophet_n_changepoints', 10, 50, step=5)
        }
        
        model = ProphetModel(trial_config)
        
    elif algorithm == 'lstm':
        # Espacio de búsqueda para LSTM
        trial_config.models['lstm'] = {
            'sequence_length': trial.suggest_int('lstm_sequence_length', 7, 30, step=7),
            'hidden_units': trial.suggest_int('lstm_hidden_units', 32, 256, step=32),
            'dropout_rate': trial.suggest_float('lstm_dropout', 0.1, 0.5, step=0.1),
            'epochs': trial.suggest_int('lstm_epochs', 50, 200, step=50),
            'batch_size': trial.suggest_categorical('lstm_batch_size', [16, 32, 64]),
            'learning_rate': trial.suggest_float('lstm_learning_rate', 1e-4, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('lstm_optimizer', ['adam', 'rmsprop']),
            'num_layers': trial.suggest_int('lstm_num_layers', 1, 3)
        }
        
        model = LSTMModel(trial_config)
        
    elif algorithm == 'arima':
        # Espacio de búsqueda para ARIMA
        p = trial.suggest_int('arima_p', 0, 5)
        d = trial.suggest_int('arima_d', 0, 2)
        q = trial.suggest_int('arima_q', 0, 5)
        P = trial.suggest_int('arima_P', 0, 3)
        D = trial.suggest_int('arima_D', 0, 2)
        Q = trial.suggest_int('arima_Q', 0, 3)
        s = 7  # Estacionalidad semanal
        
        trial_config.models['arima'] = {
            'order': (p, d, q),
            'seasonal_order': (P, D, Q, s)
        }
        
        model = ARIMAModel(trial_config)
    
    elif algorithm == 'xgboost':
        # Espacio de búsqueda para XGBoost
        trial_config.models['xgboost'] = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'gamma': trial.suggest_float('xgb_gamma', 0, 5),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 1),
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
        
        model = XGBoostModel(trial_config)
    
    else:
        raise ValueError(f"Tipo de modelo no soportado: {algorithm}")
    
    # Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Ajustar el modelo
        model.fit(X_train, y_train)
        
        # Predecir
        y_pred = model.predict(X_val)

        # Alinear predicciones con el target y filtrar valores no finitos
        y_pred = np.asarray(y_pred).ravel()

        if len(y_pred) != len(y_val):
            min_len = min(len(y_pred), len(y_val))
            y_pred = y_pred[-min_len:]
            y_val_aligned = y_val.iloc[-min_len:]
        else:
            y_val_aligned = y_val

        valid_mask = np.isfinite(y_pred)
        if not valid_mask.any():
            raise ValueError("Predicciones no válidas: todas son NaN o infinitas")

        y_pred = y_pred[valid_mask]
        y_val_aligned = y_val_aligned.iloc[np.where(valid_mask)[0]]

        # Calcular métricas
        mae = mean_absolute_error(y_val_aligned, y_pred)
        scores.append(mae)
    
    # Devolver el MAE promedio
    return np.mean(scores)

def optimize_hyperparameters(X, y, n_trials=50):
    """
    Optimiza los hiperparámetros usando Optuna.
    
    Args:
        X: Características
        y: Variable objetivo
        n_trials: Número de pruebas a realizar
        
    Returns:
        study: Objeto de estudio de Optuna
    """
    # Crear o cargar estudio con almacenamiento persistente
    try:
        # Intentar cargar un estudio existente
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_STATE)
        )
        logger.info(f"Estudio cargado. Número de trials existentes: {len(study.trials)}")
    except Exception as e:
        # Si no existe, crear uno nuevo
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='minimize',
            sampler=TPESampler(seed=RANDOM_STATE)
        )
        logger.info("Nuevo estudio creado")
    
    # Función objetivo parcial
    def objective_wrapper(trial):
        return objective(trial, X, y)
    
    # Calcular cuántos trials nuevos necesitamos
    remaining_trials = max(0, n_trials - len(study.trials))
    
    if remaining_trials > 0:
        logger.info(f"Iniciando optimización con {remaining_trials} pruebas nuevas...")
        study.optimize(objective_wrapper, n_trials=remaining_trials, show_progress_bar=True)
    else:
        logger.info(f"Ya se han completado {len(study.trials)} trials. No se necesitan más pruebas.")
    
    # Mostrar resultados
    logger.info("\nResumen de la optimización:")
    logger.info(f"Número total de trials: {len(study.trials)}")
    logger.info(f"Mejor valor (MAE): {study.best_value:.4f}")
    logger.info("Mejores parámetros encontrados:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    return study

def main():
    """Función principal para ejecutar la optimización."""
    # Cargar y preparar datos
    X, y, _ = load_and_prepare_data(forecast_horizon=7)
    
    # Optimizar
    study = optimize_hyperparameters(X, y, n_trials=50)
    
    return study

if __name__ == "__main__":
    study = main()