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
import json

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
    XGBoostModel,
    EnsembleModel,
    NewsDataLoader,
    WeatherDataLoader
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
    forecaster.add_model(XGBoostModel(config))

    # Solo añadir LSTM si TensorFlow está disponible
    try:
        forecaster.add_model(LSTMModel(config))
    except ImportError:
        logging.warning("TensorFlow no disponible, omitiendo modelo LSTM")

    # Crear ensemble
    ensemble = EnsembleModel(config)
    ensemble.add_model(ARIMAModel(config), weight=config.models['ensemble']['weights'].get('arima', 0.20))
    ensemble.add_model(ProphetModel(config), weight=config.models['ensemble']['weights'].get('prophet', 0.20))
    ensemble.add_model(RandomForestModel(config), weight=config.models['ensemble']['weights'].get('random_forest', 0.20))
    ensemble.add_model(XGBoostModel(config), weight=config.models['ensemble']['weights'].get('xgboost', 0.20))

    try:
        ensemble.add_model(LSTMModel(config), weight=config.models['ensemble']['weights'].get('lstm', 0.20))
    except ImportError:
        pass

    forecaster.add_model(ensemble)

    return forecaster

def load_data(config: ModelConfig, acids_param=None, news_param=None, weather_param=None):
    """
    Carga datos con diferentes combinaciones de parámetros.
    
    Args:
        config: Configuración del modelo
        acids_param: None, False, True (para use_one_hot)
        news_param: None, "one_hot", "aggregated" 
        weather_param: False, True (para incluir datos meteorológicos)
    
    Returns:
        DataFrame con datos combinados
    """
    data_loader = ATCAircraftDataLoader(config)
    df = data_loader.load_daily_atc_data()
    
    # Cargar datos ACIDS si se especifica
    if acids_param is not None:
        acids = data_loader.load_daily_acids_data(use_one_hot=acids_param)
        df = pd.merge(df, acids, left_index=True, right_index=True, how='left')
    
    # Cargar datos NEWS si se especifica
    if news_param is not None:
        news_loader = NewsDataLoader(config)
        news = news_loader.load_news_events(feature_type=news_param)
        df = pd.merge(df, news, left_index=True, right_index=True, how='left')
    
    # Cargar datos WEATHER si se especifica
    if weather_param:
        weather_loader = WeatherDataLoader(config)
        weather = weather_loader.load_weather_data(
            start_date=df.index.min().strftime('%Y-%m-%d'),
            end_date=df.index.max().strftime('%Y-%m-%d')
        )
        df = pd.merge(df, weather, left_index=True, right_index=True, how='left')
  
    return df

def get_config_description(acids_param, news_param, weather_param):
    """
    Genera una descripción legible de la configuración.
    """
    parts = []
    
    if acids_param is not None:
        parts.append(f"AEROLINEAS(OneHot={acids_param})")
    
    if news_param is not None:
        if news_param == "one_hot":
            parts.append("NEWS(OneHot)")
        elif news_param == "aggregated":
            parts.append("NEWS(Aggregated)")
    
    if weather_param:
        parts.append("WEATHER")
    
    if not parts:
        return "BASELINE"
    
    return " y ".join(parts)

def run_comprehensive_training():
    """
    Ejecuta entrenamiento con todas las combinaciones de parámetros.
    """
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("ENTRENAMIENTO COMPREHENSIVO - TODAS LAS COMBINACIONES")
    print("="*80)
    
    # Configuración
    config = ModelConfig()
    
    # Parámetros a probar
    acids_params = [None, False, True]
    news_params = [None, "one_hot", "aggregated"] 
    weather_params = [False, True]
    
    # Almacenar resultados
    all_results = []
    
    total_combinations = len(acids_params) * len(news_params) * len(weather_params)
    current_combination = 0
    
    print(f"Total de combinaciones a probar: {total_combinations}")
    print("-" * 80)
    
    for acids_param in acids_params:
        for news_param in news_params:
            for weather_param in weather_params:
                current_combination += 1
                config_desc = get_config_description(acids_param, news_param, weather_param)
                
                print(f"\n[{current_combination}/{total_combinations}] Configuración: {config_desc}")
                print("-" * 50)
                
                try:
                    # Cargar datos con esta configuración específica
                    df = load_data(config, acids_param, news_param, weather_param)
                    print(f"Datos cargados: {len(df)} registros")
                    
                    # Preprocesar
                    preprocessor = AircraftDataPreprocessor(config)
                    df_processed = preprocessor.preprocess_daily_data(df)
                    
                    # Crear features
                    logger.info("Creando características...")
                    feature_engineer = AircraftFeatureEngineer(config)
                    df_featured = feature_engineer.create_features(df_processed)
                    df_featured = feature_engineer.create_lagged_target(df_featured, 1)

                    # Preparar datos para modelado
                    X, y = feature_engineer.select_features_for_model(df_featured)
 
                    # Crear y entrenar modelos
                    forecaster = create_models(config)
                    training_results = forecaster.train_all_models(X, y)
                    
                    # Extraer mejores resultados
                    successful_results = [r for r in training_results.values() if r.get('success', False)]
                    if not successful_results:
                        raise Exception("No se pudo entrenar ningún modelo exitosamente")
                    
                    best_result = min(successful_results, key=lambda x: x['metrics']['mae'])
                    best_model_name = best_result['model_info']['name']
                    
                    # Generar curva de aprendizaje del mejor modelo
                    print(f"📈 Generando curva de aprendizaje para {best_model_name}...")
                    try:
                        learning_curve_data = forecaster.generate_learning_curve(best_model_name, X, y)
                        print(f"✅ Curva de aprendizaje generada para {best_model_name}")
                    except Exception as e:
                        print(f"⚠️ No se pudo generar curva de aprendizaje: {e}")
                        learning_curve_data = None
                    
                    # Guardar resultado
                    result_entry = {
                        'config': config_desc,
                        'acids_param': acids_param,
                        'news_param': news_param, 
                        'weather_param': weather_param,
                        'best_model': best_result['model_info']['name'],
                        'best_mae': best_result['metrics']['mae'],
                        'best_rmse': best_result['metrics']['rmse'],
                        'best_r2': best_result['metrics']['r2'],
                        'all_results': training_results,
                        'learning_curve': learning_curve_data  # Agregar curva de aprendizaje
                    }
                    
                    all_results.append(result_entry)
                    
                    print(f"✅ Mejor modelo: {best_result['model_info']['name']} (MAE: {best_result['metrics']['mae']:.2f})")
                    
                except Exception as e:
                    print(f"❌ Error en configuración {config_desc}: {e}")
                    # Añadir entrada con valores nulos para failed configs
                    result_entry = {
                        'config': config_desc,
                        'acids_param': acids_param,
                        'news_param': news_param,
                        'weather_param': weather_param,
                        'best_model': 'ERROR',
                        'best_mae': float('inf'),
                        'best_rmse': float('inf'),
                        'best_r2': float('-inf'),
                        'all_results': []
                    }
                    all_results.append(result_entry)
    
    # Analizar y visualizar resultados
    print("\n" + "="*80)
    print("ANÁLISIS DE RESULTADOS")
    print("="*80)
    
    # Ordenar por MAE
    sorted_results = sorted(all_results, key=lambda x: x['best_mae'] if x['best_mae'] != float('inf') else float('inf'))
    
    print("\nTop 10 configuraciones (por MAE):")
    for i, result in enumerate(sorted_results[:10], 1):
        if result['best_mae'] != float('inf'):
            print(f"{i:2d}. {result['config']}")
            print(f"    Modelo: {result['best_model']} | MAE: {result['best_mae']:.2f} | RMSE: {result['best_rmse']:.2f} | R²: {result['best_r2']:.3f}")
    
    # Guardar resultados detallados
    import json
    with open('train_results/comprehensive_training_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 Resultados guardados en: train_results/comprehensive_training_results.json")

    # Llamar a visualize_results con los datos formateados
    try:
        visualize_comprehensive_results(all_results)
    except Exception as e:
        print(f"Error en visualización: {e}")
    
    return all_results

def visualize_comprehensive_results(all_results):
    """
    Visualiza los resultados del entrenamiento comprehensivo.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Convertir a DataFrame para análisis
    df_results = pd.DataFrame(all_results)
    
    # Filtrar resultados exitosos (excluir errores)
    successful_results = df_results[df_results['best_model'] != 'ERROR']
    
    if successful_results.empty:
        print("No hay resultados exitosos para visualizar")
        return
    
    # Crear figura con múltiples gráficos
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Análisis Comprehensivo de Modelos - Todas las Combinaciones', fontsize=16, fontweight='bold')
    
    # 1. Mejor MAE por configuración
    ax1 = axes[0, 0]
    sorted_by_mae = successful_results.sort_values('best_mae')
    bars = ax1.barh(range(len(sorted_by_mae)), sorted_by_mae['best_mae'], 
                    color='lightcoral', alpha=0.7)
    ax1.set_title('Mejor MAE por Configuración', fontweight='bold')
    ax1.set_xlabel('MAE')
    ax1.set_ylabel('Configuración')
    ax1.set_yticks(range(len(sorted_by_mae)))
    
    # Acortar etiquetas para legibilidad
    config_labels = []
    for config in sorted_by_mae['config']:
        if len(config) > 30:
            config_labels.append(config[:30] + '...')
        else:
            config_labels.append(config)
    ax1.set_yticklabels(config_labels, fontsize=8)
    
    # Añadir valores
    for i, (bar, value) in enumerate(zip(bars, sorted_by_mae['best_mae'])):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', ha='left', va='center', fontweight='bold', fontsize=8)
    
    # 2. Mejor modelo por configuración
    ax2 = axes[0, 1]
    model_counts = successful_results['best_model'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_counts)))
    wedges, texts, autotexts = ax2.pie(model_counts.values, labels=model_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Distribución de Mejores Modelos', fontweight='bold')
    
    # 3. Impacto de cada feature
    ax3 = axes[1, 0]
    
    # Analizar impacto de cada feature
    feature_impact = {
        'AEROLINEAS': {'with': [], 'without': []},
        'NEWS': {'with': [], 'without': []}, 
        'WEATHER': {'with': [], 'without': []}
    }
    
    # Iterar sobre las filas del DataFrame
    for _, result in successful_results.iterrows():
        # AEROLINEAS
        if result['acids_param'] is not None:
            feature_impact['AEROLINEAS']['with'].append(result['best_mae'])
        else:
            feature_impact['AEROLINEAS']['without'].append(result['best_mae'])
        
        # NEWS
        if result['news_param'] is not None:
            feature_impact['NEWS']['with'].append(result['best_mae'])
        else:
            feature_impact['NEWS']['without'].append(result['best_mae'])
        
        # WEATHER
        if result['weather_param']:
            feature_impact['WEATHER']['with'].append(result['best_mae'])
        else:
            feature_impact['WEATHER']['without'].append(result['best_mae'])
    
    # Crear box plot comparativo
    feature_data = []
    feature_labels = []
    
    for feature, data in feature_impact.items():
        if data['with'] and data['without']:
            feature_data.extend(data['with'])
            feature_labels.extend([f'{feature} (con)'] * len(data['with']))
            feature_data.extend(data['without'])
            feature_labels.extend([f'{feature} (sin)'] * len(data['without']))
    
    if feature_data:
        df_feature = pd.DataFrame({'MAE': feature_data, 'Configuración': feature_labels})
        sns.boxplot(data=df_feature, x='Configuración', y='MAE', ax=ax3)
        ax3.set_title('Impacto de Features en MAE', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Evolución del rendimiento
    ax4 = axes[1, 1]
    
    # Ordenar resultados por complejidad (número de features)
    complexity_scores = []
    for _, result in successful_results.iterrows():
        complexity = 0
        if result['acids_param'] is not None:
            complexity += 1
        if result['news_param'] is not None:
            complexity += 1
        if result['weather_param']:
            complexity += 1
        complexity_scores.append(complexity)
    
    df_complexity = successful_results.copy()
    df_complexity['complexity'] = complexity_scores
    
    # Agrupar por complejidad
    complexity_stats = df_complexity.groupby('complexity')['best_mae'].agg(['mean', 'std', 'count'])
    
    x_pos = range(len(complexity_stats))
    bars = ax4.bar(x_pos, complexity_stats['mean'], yerr=complexity_stats['std'], 
                    capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    ax4.set_title('Rendimiento vs Complejidad', fontweight='bold')
    ax4.set_xlabel('Número de Features')
    ax4.set_ylabel('MAE Promedio')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'{i} features' for i in complexity_stats.index])
    
    # Añadir valores
    for i, (bar, mean, count) in enumerate(zip(bars, complexity_stats['mean'], complexity_stats['count'])):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{mean:.1f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('train_results/comprehensive_training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Análisis guardado en: train_results/comprehensive_training_analysis.png")
    plt.show()
    
    

def visualize_existing_results(results_file):
    """
    Lee y visualiza resultados de un archivo JSON existente.
    
    Args:
        results_file: Ruta al archivo JSON con resultados
    """
    import json
    
    try:
        # Cargar resultados desde archivo
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        print(f"📂 Resultados cargados desde: {results_file}")
        print(f"📊 Total de configuraciones: {len(all_results)}")
        
        # Mostrar resumen
        print("\n" + "="*80)
        print("RESUMEN DE RESULTADOS EXISTENTES")
        print("="*80)
        
        # Ordenar por MAE
        sorted_results = sorted(all_results, key=lambda x: x.get('best_mae', float('inf')))
        
        print(f"\nTop {min(10, len(sorted_results))} configuraciones (por MAE):")
        for i, result in enumerate(sorted_results[:10], 1):
            if result.get('best_model') != 'ERROR':
                print(f"{i:2d}. {result['config']}")
                print(f"    Modelo: {result['best_model']} | MAE: {result['best_mae']:.2f} | RMSE: {result['best_rmse']:.2f} | R²: {result['best_r2']:.3f}")
        
        # Generar visualizaciones
        print(f"\n📊 Generando visualizaciones...")
        visualize_comprehensive_results(all_results)
        
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado: {results_file}")
    except json.JSONDecodeError:
        print(f"❌ Error: El archivo {results_file} no contiene JSON válido")
    except Exception as e:
        print(f"❌ Error procesando resultados: {e}")


def main():
    """Función principal de entrenamiento."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de forecasting de aeronaves")
    parser.add_argument("--config", type=str, help="Archivo de configuración YAML")
    parser.add_argument("--data-type", type=str, default="daily_atc",
                       choices=["daily_atc", "hourly_atfm", "monthly_route"],
                       help="Tipo de datos a usar")
    parser.add_argument("--models", nargs="+", default=["arima", "prophet", "lstm", "random_forest", "xgboost", "ensemble"],
                       help="Modelos a entrenar")
    parser.add_argument("--forecast-horizon", type=int, default=1,
                       help="Horizonte de predicción")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Ejecutar entrenamiento comprehensivo con todas las combinaciones")
    parser.add_argument("--visualize-existing", type=str, 
                       help="Visualizar resultados existentes desde archivo JSON")
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Visualizar resultados existentes
    if args.visualize_existing:
        visualize_existing_results(args.visualize_existing)
        return
    
    if args.comprehensive:
        # Ejecutar entrenamiento comprehensivo
        logger.info("Iniciando entrenamiento comprehensivo con todas las combinaciones")
        run_comprehensive_training()
        return
    
    # Código original para entrenamiento simple
    logger.info("Iniciando entrenamiento de modelos de forecasting de aeronaves")
    logger.info(f"Configuración: data_type={args.data_type}, models={args.models}")

    try:
        # Cargar configuración
        config = ModelConfig()
        
        # Cargar y preparar datos
        logger.info("Cargando datos...")
        df = load_data(config)
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

        # Seleccionar mejor modelo
        if successful_models:
            best_model = min(successful_models, key=lambda x: x[1])
            logger.info(f"\nMejor modelo: {best_model[0]} (MAE: {best_model[1]:.2f})")

            # Guardar resultados
            results_dir = Path(__file__).parent / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"training_{timestamp}_results.json"
            
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'config': args.data_type,
                'models': args.models,
                'results': training_results,
                'best_model': best_model[0] if successful_models else None
            }
            
            import json
            try:
                with results_file.open('w') as f:
                    json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)
                logger.info(f"Resultados guardados en {results_file}")
            except Exception as e:
                logger.error(f"Error guardando resultados: {e}")

        logger.info("Entrenamiento completado exitosamente")

    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()