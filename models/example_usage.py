#!/usr/bin/env python3
"""
Ejemplo de uso del sistema de forecasting de aeronaves.
"""

import sys
from pathlib import Path

# Añadir el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from models import (
    ModelConfig,
    ATCAircraftDataLoader,
    AircraftDataPreprocessor,
    AircraftFeatureEngineer,
    AircraftForecaster,
    ARIMAModel,
    ProphetModel,
    RandomForestModel
)


def main():
    """Ejemplo completo de uso del sistema de forecasting."""
    print("🚀 Aircraft Forecasting System - Ejemplo de Uso")
    print("=" * 60)

    try:
        # 1. Configuración
        print("\n📋 Paso 1: Configuración")
        config = ModelConfig()
        print("✓ Configuración cargada")

        # 2. Carga de datos
        print("\n📊 Paso 2: Carga de datos ATC diarios")
        data_loader = ATCAircraftDataLoader(config)

        # Obtener información de los datos disponibles
        data_info = data_loader.get_data_info()
        if 'daily_atc' in data_info and 'records' in data_info['daily_atc']:
            records = data_info['daily_atc']['records']
            date_range = data_info['daily_atc']['date_range']
            print(f"✓ Datos diarios: {records} registros")
            print(f"✓ Rango de fechas: {date_range}")
        else:
            print("⚠️ No se pudieron cargar datos diarios ATC")
            return

        # Cargar datos de entrenamiento
        df = data_loader.get_training_data('daily_atc')
        print(f"✓ Datos de entrenamiento preparados: {len(df)} registros")

        # 3. Preprocesamiento
        print("\n🔧 Paso 3: Preprocesamiento")
        preprocessor = AircraftDataPreprocessor(config)
        df_processed = preprocessor.preprocess_daily_data(df)
        print(f"✓ Datos preprocesados: {len(df_processed)} registros")

        # 4. Ingeniería de características
        print("\n⚙️ Paso 4: Ingeniería de características")
        feature_engineer = AircraftFeatureEngineer(config)
        df_featured = feature_engineer.create_features(df_processed)
        df_featured = feature_engineer.create_lagged_target(df_featured, forecast_horizon=1)

        # Seleccionar features para modelado
        X, y = feature_engineer.select_features_for_model(df_featured)
        print(f"✓ Features creadas: {len(X.columns)} características, {len(y)} targets")

        # 5. Configuración de modelos
        print("\n🤖 Paso 5: Configuración de modelos")
        forecaster = AircraftForecaster(config)

        # Añadir modelos (ARIMA, Prophet y Random Forest para este ejemplo rápido)
        forecaster.add_model(ARIMAModel(config))
        forecaster.add_model(ProphetModel(config))
        forecaster.add_model(RandomForestModel(config))

        print("✓ Modelos configurados: ARIMA, Prophet, Random Forest")

        # 6. Entrenamiento
        print("\n🎯 Paso 6: Entrenamiento de modelos")
        print("Entrenando modelos (esto puede tomar unos minutos)...")

        training_results = forecaster.train_all_models(X, y)

        # Mostrar resultados de entrenamiento
        print("\n📈 Resultados de entrenamiento:")
        print("-" * 40)
        for model_name, result in training_results.items():
            if result.get('success', False):
                metrics = result.get('metrics', {})
                mae = metrics.get('mae', 'N/A')
                rmse = metrics.get('rmse', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                print(f"✓ {model_name.upper():<8}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
            else:
                print(f"✗ {model_name.upper():<8}: Error - {result.get('error', 'Unknown')}")

        # 7. Forecasting
        print("\n🔮 Paso 7: Forecasting")
        if forecaster.best_model:
            print(f"Mejor modelo: {forecaster.best_model.name.upper()}")

            # Forecast de los próximos 7 días usando datos históricos
            forecast_7d = forecaster.forecast(X.tail(30), forecast_horizon=7)  # Usar últimos 30 días
            print(f"✓ Forecast 7 días: {forecast_7d}")

            # Forecast futuro (sin datos históricos)
            future_forecast = forecaster.forecast(forecast_horizon=3)
            print(f"✓ Forecast futuro 3 días: {future_forecast}")

        # 8. Reporte final
        print("\n📋 Paso 8: Resumen")
        print("-" * 40)
        print("✓ Sistema de forecasting operativo")
        print(f"✓ {len(forecaster.models)} modelos disponibles")
        print(f"✓ Datos procesados: {len(df)} registros originales")
        print(f"✓ Features generadas: {len(X.columns)} características")
        print("✓ Forecasting completado exitosamente")

        print("\n🎉 ¡Ejemplo completado exitosamente!")
        print("\nPara uso avanzado:")
        print("- Ejecutar 'python models/train.py' para entrenamiento completo")
        print("- Ejecutar 'python models/evaluate.py' para evaluación detallada")
        print("- Revisar 'models/README.md' para documentación completa")

    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        print("\nPosibles soluciones:")
        print("- Verificar que los archivos de datos ATC estén en 'data/ATC csvs/'")
        print("- Instalar dependencias: pip install -r requirements.txt")
        print("- Revisar logs para más detalles")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())