import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
import json
import argparse
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from matplotlib.ticker import FuncFormatter

# Configuración de estilo para los gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_train_results(file_path):
    """
    Parsea el archivo train.txt y extrae los resultados de los modelos.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Dividir el contenido por configuraciones 
    sections = content.split('=' * 60)
    
    results = []
    current_config = None
    
    for section in sections:
        lines = section.strip().split('\n')
        
        # Buscar configuración - manejar BASELINE específicamente
        config_match = None
        for i, line in enumerate(lines):
            if line.strip() == 'BASELINE':
                current_config = 'BASELINE'
                break
            elif line and not line.startswith('2026') and not line.startswith('BASELINE') and not line.startswith('__main__') and not line.startswith('Mejor modelo'):
                if any(keyword in line.upper() for keyword in ['AEROLINEAS', 'NEWS', 'WEATHER']):
                    config_match = line.strip()
                    break
        
        if config_match:
            current_config = config_match
        
        # Extraer resultados de modelos
        for line in lines:
            if '| MAE:' in line and 'INFO -' in line:
                # Extraer nombre del modelo y métricas
                model_match = re.search(r'INFO - (\w+)', line)
                mae_match = re.search(r'MAE: ([\d.]+)', line)
                rmse_match = re.search(r'RMSE: ([\d.]+)', line)
                r2_match = re.search(r'R²: ([-\d.]+)', line)
                
                if all([model_match, mae_match, rmse_match, r2_match]):
                    results.append({
                        'config': current_config,
                        'model': model_match.group(1),
                        'mae': float(mae_match.group(1)),
                        'rmse': float(rmse_match.group(1)),
                        'r2': float(r2_match.group(1))
                    })
        
        # Extraer mejor modelo
        if 'Mejor modelo:' in section:
            best_match = re.search(r'Mejor modelo: (\w+) \(MAE: ([\d.]+)\)', section)
            if best_match:
                for result in results:
                    if result['config'] == current_config and result['model'] == best_match.group(1):
                        result['is_best'] = True
    
    return pd.DataFrame(results)

def shorten_config_names(configs):
    """
    Crea un mapeo de nombres largos a nombres cortos para las configuraciones.
    """
    config_map = {
        'BASELINE': 'BASELINE',
        'AEROLINEAS(OneHot=False)': 'AERO(NoOH)',
        'AEROLINEAS(OneHot=True)': 'AERO(OH)',
        'NEWS(OneHot)': 'NEWS(OH)',
        'NEWS(Aggregated)': 'NEWS(Agg)',
        'WEATHER': 'WEATHER',
        'AEROLINEAS(OneHot=False) y WEATHER': 'W+AERO(NoOH)',
        'AEROLINEAS(OneHot=True) y WEATHER': 'W+AERO(OH)',
        'NEWS(OneHot) y WEATHER': 'W+NEWS(OH)',
        'NEWS(Aggregated) y WEATHER': 'W+NEWS(Agg)',
        'AEROLINEAS(OneHot=False) y NEWS(OneHot)': 'AERO(NoOH)+NEWS(OH)',
        'AEROLINEAS(OneHot=True) y NEWS(OneHot)': 'AERO(OH)+NEWS(OH)',
        'AEROLINEAS(OneHot=False) y NEWS(Aggregated)': 'AERO(NoOH)+NEWS(Agg)',
        'AEROLINEAS(OneHot=True) y NEWS(Aggregated)': 'AERO(OH)+NEWS(Agg)',
        'AEROLINEAS(OneHot=False) y NEWS(OneHot) y WEATHER': 'W+AERO(NoOH)+NEWS(OH)',
        'AEROLINEAS(OneHot=True) y NEWS(OneHot) y WEATHER': 'W+AERO(OH)+NEWS(OH)',
        'AEROLINEAS(OneHot=False) y NEWS(Aggregated) y WEATHER': 'W+AERO(NoOH)+NEWS(Agg)',
        'AEROLINEAS(OneHot=True) y NEWS(Aggregated) y WEATHER': 'W+AERO(OH)+NEWS(Agg)'
    }
    
    # Para cualquier configuración no mapeada, crear un nombre corto
    shortened_map = {}
    for config in configs:
        if config in config_map:
            shortened_map[config] = config_map[config]
        else:
            # Acortar manualmente si no está en el mapa
            shortened_map[config] = config[:20] + '...' if len(config) > 20 else config
    
    return shortened_map

def create_heatmaps(df):
    """
    Crea heatmaps para MAE, RMSE y R² por modelo y configuración.
    """
    # Crear mapeo de configuraciones
    config_map = shorten_config_names(df['config'].unique())
    df['config_short'] = df['config'].map(config_map)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Heatmaps de Métricas por Modelo y Configuración', fontsize=16, fontweight='bold')
    
    # Pivot tables para cada métrica - asegurarse de incluir todas las configuraciones
    pivot_mae = df.pivot_table(values='mae', index='config_short', columns='model', aggfunc='mean', fill_value=np.nan)
    pivot_rmse = df.pivot_table(values='rmse', index='config_short', columns='model', aggfunc='mean', fill_value=np.nan)
    pivot_r2 = df.pivot_table(values='r2', index='config_short', columns='model', aggfunc='mean', fill_value=np.nan)
    
    # Heatmap MAE - colores más diferenciados
    sns.heatmap(pivot_mae, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[0], 
                cbar_kws={'label': 'MAE'})
    axes[0].set_title('MAE (más bajo es mejor)', fontweight='bold')
    axes[0].set_xlabel('Modelo')
    axes[0].set_ylabel('Configuración')
    
    # Heatmap RMSE - colores más diferenciados
    sns.heatmap(pivot_rmse, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[1],
                cbar_kws={'label': 'RMSE'})
    axes[1].set_title('RMSE (más bajo es mejor)', fontweight='bold')
    axes[1].set_xlabel('Modelo')
    axes[1].set_ylabel('')
    
    # Heatmap R² - colores más diferenciados
    sns.heatmap(pivot_r2, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[2],
                cbar_kws={'label': 'R²'}, center=0)
    axes[2].set_title('R² (más alto es mejor)', fontweight='bold')
    axes[2].set_xlabel('Modelo')
    axes[2].set_ylabel('')
    
    # Rotar etiquetas para mejor legibilidad
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Añadir leyenda útil de configuraciones
    legend_text = "Leyenda de Configuraciones:\n" + "\n".join([f"{short}: {long}" for long, short in config_map.items() if long in df['config'].unique()])
    fig.text(0.02, 0.02, legend_text, fontsize=8, verticalalignment='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
  
    plt.tight_layout(rect=[0, 0.20, 1, 0.95])
    
    # Guardar el gráfico
    import os
    os.makedirs('train_results', exist_ok=True)
    plt.savefig('train_results/heatmaps_metrics.png', dpi=300, bbox_inches='tight')
    print("📊 Heatmaps guardados en: train_results/heatmaps_metrics.png")
    
    return fig



def create_combined_plots(df):
    """
    Crea una imagen combinada con las gráficas solicitadas.
    """
    # Crear mapeo de configuraciones
    config_map = shorten_config_names(df['config'].unique())
    df['config_short'] = df['config'].map(config_map)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Análisis Comparativo de Resultados', fontsize=16, fontweight='bold')
    
    # 1. Mejora relativa vs baseline
    ax1 = axes[0, 0]
    baseline_data = df[df['config'] == 'BASELINE']
    if baseline_data.empty:
        print("⚠️  No se encontró configuración BASELINE")
        baseline_mae = df['mae'].mean()  # Usar promedio general como fallback
    else:
        baseline_mae = baseline_data['mae'].mean()
    
    config_improvement = df.groupby('config')['mae'].mean().sort_values()
    improvement_pct = ((baseline_mae - config_improvement) / baseline_mae * 100).sort_values()
    
    # Mapear configuraciones a nombres cortos para el gráfico
    improvement_pct.index = improvement_pct.index.map(config_map)
    
    colors = ['green' if x > 0 else 'red' for x in improvement_pct.values]
    bars = ax1.barh(range(len(improvement_pct)), improvement_pct.values, color=colors, alpha=0.7)
    ax1.set_title('Mejora Relativa vs Baseline (%)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Mejora (%)', fontsize=10)
    ax1.set_ylabel('Configuración', fontsize=10)
    ax1.set_yticks(range(len(improvement_pct)))
    ax1.set_yticklabels(improvement_pct.index, fontsize=8)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Añadir valores
    for bar, value in zip(bars, improvement_pct.values):
        ax1.text(bar.get_width() + (1 if value >= 0 else -1), bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', ha='left' if value >= 0 else 'right', va='center', 
                fontsize=8, fontweight='bold')
    
    # 2. MAE promedio por modelo
    ax2 = axes[0, 1]
    model_mae = df.groupby('model')['mae'].mean().sort_values()
    bars = ax2.bar(model_mae.index, model_mae.values, color='skyblue', edgecolor='navy', alpha=0.7)
    ax2.set_title('MAE Promedio por Modelo', fontweight='bold', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=10)
    ax2.set_xlabel('Modelo', fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    # Añadir valores sobre las barras
    for bar, value in zip(bars, model_mae.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. MAE promedio por configuración
    ax3 = axes[1, 0]
    config_mae = df.groupby('config')['mae'].mean().sort_values()
    config_mae.index = config_mae.index.map(config_map)  # Mapear a nombres cortos
    
    bars = ax3.barh(range(len(config_mae)), config_mae.values, color='lightcoral', alpha=0.7)
    ax3.set_title('MAE Promedio por Configuración', fontweight='bold', fontsize=12)
    ax3.set_xlabel('MAE', fontsize=10)
    ax3.set_ylabel('Configuración', fontsize=10)
    ax3.set_yticks(range(len(config_mae)))
    ax3.set_yticklabels(config_mae.index, fontsize=8)
    
    # Añadir valores
    for bar, value in zip(bars, config_mae.values):
        ax3.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', ha='left', va='center', fontweight='bold', fontsize=8)
    
    # 4. Mejor modelo por configuración (sin texto sobre barras)
    ax4 = axes[1, 1]
    best_results = df.loc[df.groupby('config')['mae'].idxmin()].sort_values('mae')
    best_results = best_results.copy()
    best_results['config_short'] = best_results['config'].map(config_map)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(best_results)))
    bars = ax4.bar(range(len(best_results)), best_results['mae'], color=colors, alpha=0.8)
    ax4.set_title('Mejor Modelo por Configuración', fontweight='bold', fontsize=12)
    ax4.set_ylabel('MAE', fontsize=10)
    ax4.set_xlabel('Configuración', fontsize=10)
    ax4.set_xticks(range(len(best_results)))
    ax4.set_xticklabels(best_results['config_short'], rotation=45, ha='right', fontsize=8)
    
    # Sin texto sobre las barras como solicitaste
    
    # Añadir leyenda útil de configuraciones
    if len(config_map) > 1:  # Solo si hay múltiples configuraciones
        legend_text = "Configuraciones:\n" + "\n".join([f"• {short}: {long}" for long, short in config_map.items() if long in df['config'].unique()])
        fig.text(0.02, 0.02, legend_text, fontsize=6, verticalalignment='bottom', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9))
        print("📋 Leyenda de configuraciones añadida para facilitar lectura")
    
    plt.tight_layout(rect=[0, 0.20, 1, 0.95])

    plt.savefig('train_results/combined_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Gráficos combinados guardados en: train_results/combined_analysis.png")
    
    return fig

def create_learning_curve_plot(json_file_path):
    """
    Crea una visualización de la curva de aprendizaje del mejor modelo usando datos reales.
    
    Args:
        json_file_path: Ruta al archivo JSON con resultados
    """
    try:
        # Cargar resultados
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        # Filtrar resultados exitosos y encontrar el mejor modelo
        successful_results = [r for r in all_results if r.get('best_model') != 'ERROR' and r.get('learning_curve') is not None]
        
        if not successful_results:
            print("❌ No hay resultados exitosos con curvas de aprendizaje para mostrar")
            return
        
        # Encontrar la mejor configuración (menor MAE)
        best_result = min(successful_results, key=lambda x: x['best_mae'])
        best_config = best_result['config']
        best_model_name = best_result['best_model']
        learning_curve_data = best_result['learning_curve']
        
        print(f"📈 Creando curva de aprendizaje real para el mejor modelo:")
        print(f"   Modelo: {best_model_name}")
        print(f"   Configuración: {best_config}")
        print(f"   MAE: {best_result['best_mae']:.2f}")
        
        # Crear figura para la curva de aprendizaje
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Extraer datos reales de la curva de aprendizaje
        # Convertir strings a arrays si es necesario
        train_sizes_str = learning_curve_data['train_sizes']
        train_scores_mean_str = learning_curve_data['train_scores_mean']
        train_scores_std_str = learning_curve_data['train_scores_std']
        val_scores_mean_str = learning_curve_data['val_scores_mean']
        val_scores_std_str = learning_curve_data['val_scores_std']
        
        # Convertir a arrays numéricos
        if isinstance(train_sizes_str, str):
            train_sizes = np.array([float(x) for x in train_sizes_str.strip('[]').split()])
            train_scores_mean = np.array([float(x) for x in train_scores_mean_str.strip('[]').split()])
            train_scores_std = np.array([float(x) for x in train_scores_std_str.strip('[]').split()])
            val_scores_mean = np.array([float(x) for x in val_scores_mean_str.strip('[]').split()])
            val_scores_std = np.array([float(x) for x in val_scores_std_str.strip('[]').split()])
        else:
            train_sizes = np.array(learning_curve_data['train_sizes'])
            train_scores_mean = np.array(learning_curve_data['train_scores_mean'])
            train_scores_std = np.array(learning_curve_data['train_scores_std'])
            val_scores_mean = np.array(learning_curve_data['val_scores_mean'])
            val_scores_std = np.array(learning_curve_data['val_scores_std'])
        
        # Graficar curvas con datos reales
        ax.plot(train_sizes, train_scores_mean, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='MAE Entrenamiento')
        ax.fill_between(train_sizes, 
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.2, color='#1f77b4')
        
        ax.plot(train_sizes, val_scores_mean, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='MAE Validación')
        ax.fill_between(train_sizes,
                        val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std,
                        alpha=0.2, color='#ff7f0e')
        
        # Añadir etiquetas con valores exactos en puntos clave
        for i, (x, y_train, y_val) in enumerate(zip(train_sizes[::2], train_scores_mean[::2], val_scores_mean[::2])):
            ax.annotate(f'{y_train:.1f}', (x, y_train), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8, color='#1f77b4')
            ax.annotate(f'{y_val:.1f}', (x, y_val), textcoords="offset points", 
                       xytext=(0,-15), ha='center', fontsize=8, color='#ff7f0e')
        
        # Línea horizontal para el mejor MAE alcanzado
        final_mae = best_result['best_mae']
        ax.axhline(y=final_mae, color='red', linestyle='--', alpha=0.7, label=f'Mejor MAE: {final_mae:.2f}')
        
        # Configurar el gráfico
        ax.set_xlabel('Tamaño del conjunto de entrenamiento', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax.set_title(f'Curva de Aprendizaje Real - Mejor Modelo\n{best_model_name.upper()} ({best_config})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_yscale('log')
        
        # Configurar más marcas en el eje Y para mejor legibilidad
        y_min = min(np.min(train_scores_mean), np.min(val_scores_mean)) * 0.8
        y_max = max(np.max(train_scores_mean), np.max(val_scores_mean)) * 1.2
        
        # Crear marcas más densas en el eje Y
        y_ticks = np.logspace(np.log10(y_min), np.log10(y_max), 10)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}" if y >= 1 else f"{y:.1f}"))
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=11, loc='upper right')
        
        # Añadir anotaciones importantes
        ax.annotate(f'Mejor rendimiento\nMAE: {final_mae:.2f}', 
                   xy=(train_sizes[-1], final_mae), xytext=(train_sizes[-1]*0.7, final_mae * 1.5),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=10, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Añadir información sobre el sobreajuste
        final_train_mae = train_scores_mean[-1]
        final_val_mae = val_scores_mean[-1]
        
        # Asegurarse de que los valores sean numéricos
        try:
            final_train_mae = float(final_train_mae)
            final_val_mae = float(final_val_mae)
            overfitting_ratio = final_val_mae / final_train_mae if final_train_mae > 0 else float('inf')
        except (ValueError, TypeError):
            print("⚠️ No se pudo calcular el ratio de sobreajuste")
            overfitting_ratio = 1.0
        
        ax.text(0.02, 0.98, f'Ratio Validación/Entrenamiento: {overfitting_ratio:.2f}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if overfitting_ratio < 1.2 else "lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar el gráfico
        output_file = 'train_results/learning_curve_best_model.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📈 Curva de aprendizaje real guardada en: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creando curva de aprendizaje: {e}")


def load_and_visualize_json_results(json_file_path):
    """
    Carga resultados desde un archivo JSON y genera visualizaciones.
    
    Args:
        json_file_path: Ruta al archivo JSON con resultados
    """
    try:
        # Cargar resultados
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        print(f" Resultados cargados desde: {json_file_path}")
        print(f" Total de configuraciones: {len(all_results)}")
        
        # Filtrar resultados exitosos
        successful_results = [r for r in all_results if r.get('best_model') != 'ERROR']
        print(f" Configuraciones exitosas: {len(successful_results)}")
        
        # Extraer todos los resultados individuales de modelos
        formatted_results = []
        model_names = set()
        
        for result in successful_results:
            config = result['config']
            all_models = result.get('all_results', {})
            
            for model_name, model_data in all_models.items():
                if model_data.get('success', False):
                    metrics = model_data.get('metrics', {})
                    formatted_results.append({
                        'config': config,
                        'model': model_name,
                        'mae': metrics.get('mae', 0),
                        'rmse': metrics.get('rmse', 0),
                        'r2': metrics.get('r2', 0)
                    })
                    model_names.add(model_name)
        
        print(f" Modelos encontrados: {', '.join(sorted(model_names))}")
        
        if not formatted_results:
            print(" No hay resultados para visualizar")
            return
        
        # Convertir a DataFrame
        df_results = pd.DataFrame(formatted_results)
        
        print(f" Total de resultados: {len(df_results)}")
        print(f"  Configuraciones: {df_results['config'].nunique()}")
        print(f" Modelos: {df_results['model'].nunique()}")
        
        # Generar visualizaciones usando las funciones existentes
        print("\n Generando visualizaciones...")
        
        # Asegurar que el directorio exista
        import os
        os.makedirs('train_results', exist_ok=True)
        
        # Crear heatmaps
        create_heatmaps(df_results)
        
        # Crear gráficos combinados
        create_combined_plots(df_results)
        
        # Crear curva de aprendizaje del mejor modelo
        create_learning_curve_plot(json_file_path)
        
        print(f"\n✅ Visualizaciones completadas para: {json_file_path}")
        
    except FileNotFoundError:
        print(f"❌ Error: Archivo no encontrado: {json_file_path}")
    except json.JSONDecodeError:
        print(f"❌ Error: El archivo {json_file_path} no contiene JSON válido")
    except Exception as e:
        print(f"❌ Error procesando resultados: {e}")


def main():
    """Función principal con opciones para visualizar resultados."""
    parser = argparse.ArgumentParser(description="Visualización de resultados de entrenamiento")
    parser.add_argument("--train-file", type=str, default="train.txt",
                       help="Archivo de texto con resultados de entrenamiento (default: train.txt)")
    parser.add_argument("--json-file", type=str,
                       help="Archivo JSON con resultados del entrenamiento comprehensivo")
    
    args = parser.parse_args()
    
    if args.json_file:
        # Visualizar desde archivo JSON
        load_and_visualize_json_results(args.json_file)
    else:
        load_and_visualize_json_results("models/train_results/comprehensive_training_results.json")

    # else:
    #     # Visualizar desde archivo de texto (funcionalidad original)
    #     try:
    #         results = parse_train_results(args.train_file)
            
    #         if not results:
    #             print("No se encontraron resultados válidos en el archivo.")
    #             return
            
    #         print(f"Se encontraron {len(results)} resultados válidos.")
            
    #         # Crear visualizaciones
    #         print("📈 Generando heatmaps de métricas...")
    #         fig1 = create_heatmaps(results)
    #         fig1.savefig('train_results/heatmaps_metrics.png', dpi=300, bbox_inches='tight')
            
    #         print("📊 Generando gráficos combinados...")
    #         fig2 = create_combined_plots(results)
    #         fig2.savefig('train_results/combined_analysis.png', dpi=300, bbox_inches='tight')

            
    #         print("\nVisualizaciones guardadas:")
    #         print(f"   - heatmaps_metrics.png (MAE, RMSE, R²)")
    #         print(f"   - combined_analysis.png (4 gráficos combinados)")
            
    #     except FileNotFoundError:
    #         print(f"Error: No se encontró el archivo '{args.train_file}'")
    #     except Exception as e:
    #         print(f"Error procesando el archivo: {e}")
    
    # plt.show()


if __name__ == "__main__":
    main()
