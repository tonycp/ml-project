"""
Ejemplo de uso de MultiModalDataLoader para combinar datos ATC, clima y noticias.
"""
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path

# Añadir el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

# Importar las clases necesarias
from models import (
    ModelConfig,
    ATCAircraftDataLoader,
    AircraftDataPreprocessor,
    AircraftFeatureEngineer,
    AircraftForecaster,
    ARIMAModel,
    ProphetModel,
    RandomForestModel,
    MultiModalDataLoader
)

def load_and_preprocess_data(config):
    """
    Carga y preprocesa los datos de múltiples fuentes.
    """
    logger.info("Iniciando carga de datos...")
    
    # 1. Cargar datos usando MultiModalDataLoader
    logger.info("Cargando datos ATC, clima y noticias...")
    multimodal_loader = MultiModalDataLoader(config)
    
    # Cargar datos combinados
    combined_data = multimodal_loader.load_multimodal_data(
        data_type='daily_atc',
        include_weather=True,
        include_news=False
    )
    
    # 2. Preprocesar datos
    preprocessor = AircraftDataPreprocessor(config)
    processed_data = preprocessor.preprocess_daily_data(combined_data)
    
    # 3. Crear características adicionales
    feature_engineer = AircraftFeatureEngineer(config)
    featured_data = feature_engineer.create_features(processed_data)
    featured_data = feature_engineer.create_lagged_target(featured_data, forecast_horizon=1)

    # 4. Seleccionar features para modelado
    X, y = feature_engineer.select_features_for_model(featured_data)
    
    logger.info(f"Datos preparados para modelado: {len(X)} registros, {len(X.columns)} características")
    
    return X, y, featured_data

def train_and_evaluate(X, y, config):
    """
    Entrena y evalúa los modelos con los datos combinados.
    """
    logger.info("Preparando datos para entrenamiento...")
    
    # Ensure feature names are set for RandomForest
    if hasattr(X, 'columns'):
        X.columns = X.columns.astype(str)
    
    # Inicializar el forecaster
    forecaster = AircraftForecaster(config)
    
    # Añadir modelos al forecaster
    forecaster.add_model(ARIMAModel(config))
    forecaster.add_model(ProphetModel(config))
    forecaster.add_model(RandomForestModel(config))
    
    # Entrenar modelos
    logger.info("Entrenando modelos...")
    results = forecaster.train_all_models(X, y)
    
    # Mostrar resultados
    for model_name, metrics in results.items():
        logger.info(f"\nResultados para {model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"  {metric}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v:.4f}" if isinstance(v, (int, float)) else f"    {k}: {v}")
            else:
                logger.info(f"  {metric}: {value}")
    
    return forecaster, results

def plot_results(y_true, predictions, title="Predicción vs Real"):
    """
    Grafica los valores reales vs predichos.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot real values (last 90 days for better visualization)
    plot_days = min(90, len(y_true))
    plt.plot(y_true.index[-plot_days:], y_true.values[-plot_days:], 
             label='Real', color='blue', linewidth=2)
    
    # Plot predictions
    for model_name, preds in predictions.items():
        if preds is not None and len(preds) > 0:
            pred_dates = pd.date_range(
                start=y_true.index[-1] + pd.Timedelta(days=1),
                periods=len(preds)
            )
            plt.plot(pred_dates, preds, '--', 
                    label=f'Predicho - {model_name}',
                    linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Número de aeronaves', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    # Configuración
    config = ModelConfig()
    
    try:
        # 1. Cargar y preprocesar datos
        X, y , combined_data = load_and_preprocess_data(config)
        
        # 2. Entrenar y evaluar modelos
        forecaster, results = train_and_evaluate(X, y, config)
        
        # 3. Hacer predicciones
        forecast_horizon = 7  # Predecir 7 días
        last_date = combined_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )
        
        # Hacer predicciones
        predictions = {}
        for model_name in ['arima', 'prophet', 'random_forest']:
            try:
                # Get the last sequence of features for prediction
                if model_name == 'random_forest':
                    # For RandomForest, we need to prepare features for the forecast period
                    future_features = X.iloc[-forecast_horizon:].copy()
                    future_features.index = future_dates  # Update index to future dates
                    preds = forecaster.models[model_name].predict(future_features)
                else:
                    # For time series models
                    preds = forecaster.forecast(
                        X=None,  # Some models might not need X
                        forecast_horizon=forecast_horizon,
                        model_name=model_name
                    )
                
                if preds is not None and len(preds) > 0:
                    predictions[model_name] = preds
                    logger.info(f"Predicciones exitosas con {model_name}")
                else:
                    logger.warning(f"No se obtuvieron predicciones de {model_name}")
                    
            except Exception as e:
                logger.error(f"Error al hacer predicciones con {model_name}: {str(e)}")
                predictions[model_name] = None
        
        # 4. Visualizar resultados
        plot_results(
            y,
            predictions,
            "Predicción de tráfico aéreo con datos multimodales"
        )
        
        logger.info("Proceso completado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error en el proceso: {e}", exc_info=True)

if __name__ == "__main__":
    main()
