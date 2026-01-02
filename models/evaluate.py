#!/usr/bin/env python3
"""
Script de evaluación para modelos de forecasting de aeronaves.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    EnsembleModel
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


def load_and_prepare_data(config: ModelConfig, data_type: str = "daily_atc") -> tuple:
    """Carga y prepara datos para evaluación."""
    logger = logging.getLogger(__name__)

    # Cargar datos
    data_loader = ATCAircraftDataLoader(config)
    df = data_loader.get_training_data(data_type)

    # Preprocesar
    preprocessor = AircraftDataPreprocessor(config)
    if data_type == "daily_atc":
        df_processed = preprocessor.preprocess_daily_data(df)
    else:
        df_processed = preprocessor.preprocess_hourly_data(df)

    # Crear features
    feature_engineer = AircraftFeatureEngineer(config)
    df_featured = feature_engineer.create_features(df_processed)

    # Dividir en train/val/test
    train_df, val_df, test_df = data_loader.split_train_val_test(df_featured)

    logger.info(f"Datos preparados: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_df, val_df, test_df, feature_engineer


def evaluate_models(forecaster: AircraftForecaster,
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   feature_engineer: AircraftFeatureEngineer,
                   forecast_horizons: list = [1, 7, 14]) -> dict:
    """Evalúa todos los modelos en diferentes horizontes de predicción."""
    logger = logging.getLogger(__name__)

    results = {}

    for horizon in forecast_horizons:
        logger.info(f"Evaluando horizonte de predicción: {horizon} {'días' if horizon > 1 else 'día'}")

        # Preparar target con lag
        train_featured = feature_engineer.create_lagged_target(train_df.copy(), horizon)
        val_featured = feature_engineer.create_lagged_target(val_df.copy(), horizon)
        test_featured = feature_engineer.create_lagged_target(test_df.copy(), horizon)

        # Seleccionar features
        X_train, y_train = feature_engineer.select_features_for_model(train_featured)
        X_val, y_val = feature_engineer.select_features_for_model(val_featured)
        X_test, y_test = feature_engineer.select_features_for_model(test_featured)

        # Entrenar modelos
        training_results = forecaster.train_all_models(X_train, y_train)

        # Evaluar en conjunto de test
        test_results = {}
        predictions = {}

        for model_name, model in forecaster.models.items():
            if model.is_trained:
                try:
                    # Predicciones
                    y_pred = model.predict(X_test)

                    # Calcular métricas
                    metrics = calculate_metrics(y_test, y_pred)

                    test_results[model_name] = {
                        'metrics': metrics,
                        'predictions': y_pred.tolist(),
                        'actual': y_test.tolist()
                    }

                    predictions[model_name] = y_pred

                    logger.info(f"{model_name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

                except Exception as e:
                    logger.error(f"Error evaluando {model_name}: {e}")
                    test_results[model_name] = {'error': str(e)}

        results[f'horizon_{horizon}'] = {
            'training_results': training_results,
            'test_results': test_results,
            'predictions': {k: v.tolist() for k, v in predictions.items()},
            'dates': test_df.index.strftime('%Y-%m-%d').tolist()
        }

    return results


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de evaluación."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error
    )

    # Evitar división por cero en MAPE
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true_safe, y_pred) * 100,  # En porcentaje
        'r2': r2_score(y_true, y_pred),
        'mean_error': np.mean(y_pred - y_true),
        'std_error': np.std(y_pred - y_true)
    }

    return metrics


def plot_results(results: dict, output_dir: Path) -> None:
    """Genera gráficos de los resultados de evaluación."""
    logger = logging.getLogger(__name__)

    try:
        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)

        # Gráfico de comparación de métricas por horizonte
        horizons = []
        models = set()
        mae_scores = {}

        for horizon_key, horizon_data in results.items():
            horizon = int(horizon_key.split('_')[1])
            horizons.append(horizon)

            for model_name, model_data in horizon_data['test_results'].items():
                if 'metrics' in model_data:
                    models.add(model_name)
                    if model_name not in mae_scores:
                        mae_scores[model_name] = []
                    mae_scores[model_name].append(model_data['metrics']['mae'])

        # Gráfico de MAE por horizonte
        plt.figure(figsize=(12, 8))
        for model_name in sorted(models):
            if model_name in mae_scores and len(mae_scores[model_name]) == len(horizons):
                plt.plot(horizons, mae_scores[model_name], marker='o', label=model_name.upper())

        plt.xlabel('Horizonte de Predicción (días)')
        plt.ylabel('MAE (Error Absoluto Medio)')
        plt.title('Comparación de Modelos - Error por Horizonte de Predicción')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'mae_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Gráfico de predicciones vs real para el mejor modelo
        best_horizon = min(results.keys(), key=lambda x: int(x.split('_')[1]))
        best_results = results[best_horizon]['test_results']

        # Encontrar mejor modelo por MAE
        best_model = min(
            [(name, data['metrics']['mae']) for name, data in best_results.items() if 'metrics' in data],
            key=lambda x: x[1]
        )[0]

        # Gráfico de predicciones
        plt.figure(figsize=(15, 8))

        dates = pd.to_datetime(results[best_horizon]['dates'])
        actual = best_results[best_model]['actual']
        predicted = best_results[best_model]['predictions']

        plt.plot(dates, actual, label='Valor Real', color='blue', linewidth=2)
        plt.plot(dates, predicted, label=f'Predicción ({best_model.upper()})',
                color='red', linestyle='--', linewidth=2)

        plt.xlabel('Fecha')
        plt.ylabel('Número de Aeronaves')
        plt.title(f'Predicciones vs Valores Reales - {best_model.upper()} (Horizonte: {best_horizon.split("_")[1]} días)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'predictions_{best_model}.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Gráficos guardados en: {output_dir}")

    except Exception as e:
        logger.error(f"Error generando gráficos: {e}")


def generate_report(results: dict, output_file: Path) -> None:
    """Genera reporte de evaluación en formato Markdown."""
    logger = logging.getLogger(__name__)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Reporte de Evaluación - Forecasting de Aeronaves\n\n")
            f.write(f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Resumen general
            f.write("## Resumen General\n\n")

            for horizon_key, horizon_data in results.items():
                horizon = horizon_key.split('_')[1]
                f.write(f"### Horizonte de Predicción: {horizon} días\n\n")

                f.write("| Modelo | MAE | RMSE | MAPE | R² |\n")
                f.write("|--------|-----|------|------|----|\n")

                for model_name, model_data in horizon_data['test_results'].items():
                    if 'metrics' in model_data:
                        metrics = model_data['metrics']
                        f.write(f"| {model_name.upper()} | {metrics['mae']:.2f} | {metrics['rmse']:.2f} | {metrics['mape']:.2f}% | {metrics['r2']:.3f} |\n")

                f.write("\n")

            # Mejor modelo por horizonte
            f.write("## Mejores Modelos por Horizonte\n\n")
            for horizon_key, horizon_data in results.items():
                horizon = horizon_key.split('_')[1]

                best_model = min(
                    [(name, data['metrics']['mae']) for name, data in horizon_data['test_results'].items() if 'metrics' in data],
                    key=lambda x: x[1]
                )

                f.write(f"- **Horizonte {horizon} días:** {best_model[0].upper()} (MAE: {best_model[1]:.2f})\n")

            f.write("\n")

            # Conclusiones
            f.write("## Conclusiones\n\n")
            f.write("Este reporte muestra el rendimiento de diferentes modelos de forecasting ")
            f.write("para predecir el número de aeronaves en el espacio aéreo cubano.\n\n")

            f.write("### Recomendaciones:\n")
            f.write("1. **Modelo principal:** Usar el modelo con mejor MAE para cada horizonte\n")
            f.write("2. **Horizontes cortos (1-7 días):** Modelos más simples como ARIMA pueden ser suficientes\n")
            f.write("3. **Horizontes largos (14+ días):** Considerar modelos más complejos como LSTM\n")
            f.write("4. **Ensemble:** Evaluar el uso de modelos ensemble para mayor robustez\n")

        logger.info(f"Reporte generado: {output_file}")

    except Exception as e:
        logger.error(f"Error generando reporte: {e}")


def main():
    """Función principal de evaluación."""
    parser = argparse.ArgumentParser(description="Evaluación de modelos de forecasting de aeronaves")
    parser.add_argument("--config", type=str, help="Archivo de configuración YAML")
    parser.add_argument("--data-type", type=str, default="daily_atc",
                       choices=["daily_atc", "hourly_atfm", "monthly_route"],
                       help="Tipo de datos a usar")
    parser.add_argument("--models", nargs="+", default=["arima", "prophet", "lstm", "ensemble"],
                       help="Modelos a evaluar")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 14],
                       help="Horizontes de predicción a evaluar")
    parser.add_argument("--output-dir", type=str, default="models/evaluation_results",
                       help="Directorio para guardar resultados")
    parser.add_argument("--load-models", action="store_true",
                       help="Cargar modelos guardados en lugar de entrenar")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Nivel de logging")
    parser.add_argument("--log-file", type=str,
                       help="Archivo de log")

    args = parser.parse_args()

    # Configurar logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Iniciando evaluación de modelos de forecasting de aeronaves")
    logger.info(f"Configuración: data_type={args.data_type}, models={args.models}, horizons={args.horizons}")

    try:
        # Cargar configuración
        config = ModelConfig()

        # Preparar datos
        logger.info("Preparando datos...")
        train_df, val_df, test_df, feature_engineer = load_and_prepare_data(config, args.data_type)

        # Crear modelos
        forecaster = AircraftForecaster(config)

        # Añadir modelos
        model_classes = {
            'arima': ARIMAModel,
            'prophet': ProphetModel,
            'random_forest': RandomForestModel,
            'lstm': LSTMModel,
            'ensemble': EnsembleModel
        }

        for model_name in args.models:
            if model_name in model_classes:
                try:
                    forecaster.add_model(model_classes[model_name](config))
                except ImportError as e:
                    logger.warning(f"No se pudo añadir {model_name}: {e}")

        # Cargar modelos si se solicita
        if args.load_models:
            logger.info("Cargando modelos guardados...")
            forecaster.load_models()

        # Evaluar modelos
        logger.info("Evaluando modelos...")
        results = evaluate_models(forecaster, train_df, val_df, test_df,
                                feature_engineer, args.horizons)

        # Crear directorio de salida
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generar gráficos
        logger.info("Generando gráficos...")
        plot_results(results, output_dir)

        # Generar reporte
        report_file = output_dir / "evaluation_report.md"
        generate_report(results, report_file)

        # Guardar resultados en JSON
        import json
        results_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convertir arrays numpy a listas para JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy)

        logger.info(f"Resultados guardados en: {results_file}")
        logger.info(f"Reporte generado: {report_file}")
        logger.info("Evaluación completada exitosamente")

        return 0

    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())