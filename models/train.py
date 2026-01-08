#!/usr/bin/env python3
"""
Script de entrenamiento para modelos de forecasting de aeronaves.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Añadir el directorio padre al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from models import (
    ModelConfig,
    ATCAircraftDataLoader,
    AircraftDataPreprocessor,
    AircraftFeatureEngineer,
    AircraftForecaster,
    ARIMAModel,
    ProphetModel,
    RandomForestModel,
    LSTMModel,
    EnsembleModel,
    NewsDataLoader
)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configura el logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def create_models(config: ModelConfig) -> AircraftForecaster:
    """Crea y configura los modelos de forecasting."""
    forecaster = AircraftForecaster(config)

    # Añadir modelos individuales
    forecaster.add_model(ARIMAModel(config))
    forecaster.add_model(ProphetModel(config))
    forecaster.add_model(RandomForestModel(config))

    # Solo añadir LSTM si TensorFlow está disponible
    try:
        forecaster.add_model(LSTMModel(config))
    except ImportError:
        logging.warning("TensorFlow no disponible, omitiendo modelo LSTM")

    # Crear ensemble
    ensemble = EnsembleModel(config)
    ensemble.add_model(ARIMAModel(config), weight=config.models['ensemble']['weights'].get('arima', 0.25))
    ensemble.add_model(ProphetModel(config), weight=config.models['ensemble']['weights'].get('prophet', 0.25))
    ensemble.add_model(RandomForestModel(config), weight=config.models['ensemble']['weights'].get('random_forest', 0.25))

    try:
        ensemble.add_model(LSTMModel(config), weight=config.models['ensemble']['weights'].get('lstm', 0.25))
    except ImportError:
        pass

    forecaster.add_model(ensemble)

    return forecaster


def main():
    """Función principal de entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de forecasting de aeronaves")
    parser.add_argument("--config", type=str, help="Archivo de configuración YAML")
    parser.add_argument("--data-type", type=str, default="daily_atc",
                       choices=["daily_atc", "hourly_atfm", "monthly_route"],
                       help="Tipo de datos a usar")
    parser.add_argument("--models", nargs="+", default=["arima", "prophet", "lstm", "random_forest", "ensemble"],
                       help="Modelos a entrenar")
    parser.add_argument("--forecast-horizon", type=int, default=1,
                       help="Horizonte de predicción")
    parser.add_argument("--save-models", action="store_true",
                       help="Guardar modelos entrenados")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Nivel de logging")
    parser.add_argument("--log-file", type=str,
                       help="Archivo de log")

    args = parser.parse_args()

    # Configurar logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando entrenamiento de modelos de forecasting de aeronaves")
    logger.info(f"Configuración: data_type={args.data_type}, models={args.models}")

    try:
        # Cargar configuración
        config = ModelConfig()

        # Cargar datos
        logger.info("Cargando datos...")

        data_loader = ATCAircraftDataLoader(config)
        df = data_loader.load_daily_atc_data()
        
        acids = data_loader.load_daily_acids_data(use_one_hot=True)
        df = pd.merge(df, acids, left_index=True, right_index=True, how='left')

        news_loader = NewsDataLoader(config)
        news = news_loader.load_news_events(feature_type='aggregated')
        df = pd.merge(df, news, left_index=True, right_index=True, how='left')

        logger.info(f"Datos cargados: {len(df)} registros del {df.index.min()} al {df.index.max()}")

        # Preprocesar datos
        logger.info("Preprocesando datos...")
        preprocessor = AircraftDataPreprocessor(config)
        df_processed = preprocessor.preprocess_daily_data(df) if args.data_type == "daily_atc" else \
                      preprocessor.preprocess_hourly_data(df)

        # Crear features
        logger.info("Creando características...")
        feature_engineer = AircraftFeatureEngineer(config)
        df_featured = feature_engineer.create_features(df_processed)
        df_featured = feature_engineer.create_lagged_target(df_featured, args.forecast_horizon)

        # Preparar datos para modelado
        X, y = feature_engineer.select_features_for_model(df_featured)

        logger.info(f"Datos preparados: {len(X)} muestras, {len(X.columns)} features")

        # Crear y entrenar modelos
        forecaster = create_models(config)

        # Filtrar modelos según argumentos
        available_models = list(forecaster.models.keys())
        models_to_train = [m for m in args.models if m in available_models]

        if not models_to_train:
            logger.error(f"Ninguno de los modelos especificados está disponible: {args.models}")
            logger.info(f"Modelos disponibles: {available_models}")
            return 1

        # Mantener solo modelos seleccionados
        forecaster.models = {k: v for k, v in forecaster.models.items() if k in models_to_train}

        logger.info(f"Entrenando modelos: {models_to_train}")

        # Entrenar modelos
        training_results = forecaster.train_all_models(X, y)

        # Mostrar resultados
        logger.info("\n" + "="*60)
        logger.info("RESULTADOS DE ENTRENAMIENTO")
        logger.info("="*60)

        successful_models = []
        for model_name, result in training_results.items():
            if result.get('success', False):
                metrics = result.get('metrics', {})
                mae = metrics.get('mae', 'N/A')
                rmse = metrics.get('rmse', 'N/A')
                r2 = metrics.get('r2', 'N/A')

                logger.info(f"{model_name.upper():<10} | MAE: {mae:<8.2f} | RMSE: {rmse:<8.2f} | R²: {r2:<8.2f}")
                successful_models.append((model_name, mae))
            else:
                logger.error(f"{model_name.upper():<10} | ERROR: {result.get('error', 'Unknown')}")

        # Guardar modelos si se solicita
        if args.save_models and successful_models:
            logger.info("Guardando modelos...")
            forecaster.save_models()

        # Reporte final
        if successful_models:
            best_model = min(successful_models, key=lambda x: x[1])
            logger.info(f"\nMejor modelo: {best_model[0]} (MAE: {best_model[1]:.2f})")

            # Asegurarse de que el directorio de resultados exista
            results_dir = Path(__file__).parent / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Crear nombre de archivo con timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"training_{timestamp}_results.json"
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'config': config.__dict__,
                'data_info': {
                    'type': args.data_type,
                    'records': len(df),
                    'date_range': f"{df.index.min()} to {df.index.max()}"
                },
                'training_results': training_results,
                'best_model': best_model[0]
            }

            import json
            try:
                with results_file.open('w') as f:
                    json.dump(results_data, f, indent=2, default=str)
                logger.info(f"Resultados guardados en {results_file}")
            except Exception as e:
                logger.error(f"Error al guardar los resultados: {e}")
                # Intentar guardar en el directorio actual si falla
                try:
                    backup_file = Path(f"training_results_{timestamp}.json")
                    with backup_file.open('w') as f:
                        json.dump(results_data, f, indent=2, default=str)
                    logger.info(f"Resultados guardados en {backup_file.absolute()}")
                except Exception as e2:
                    logger.error(f"Error al guardar el archivo de respaldo: {e2}")
                    logger.info("Imprimiendo resultados en la consola...")
                    print("\nResultados del entrenamiento:")
                    print(json.dumps(results_data, indent=2, default=str))

        logger.info("Entrenamiento completado exitosamente")
        return 0

    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())