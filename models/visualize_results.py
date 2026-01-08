import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

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
        'AEROLINEAS (OneHot=False)': 'AERO(NoOH)',
        'AEROLINEAS (OneHot=True)': 'AERO(OH)',
        'NEWS (OneHot)': 'NEWS(OH)',
        'NEWS (Aggregated)': 'NEWS(Agg)',
        'WEATHER': 'WEATHER',
        'WEATHER y AEROLINEAS (OneHot=False)': 'W+AERO(NoOH)',
        'WEATHER y AEROLINEAS (OneHot=True)': 'W+AERO(OH)',
        'WEATHER y NEWS (OneHot)': 'W+NEWS(OH)',
        'WEATHER y NEWS (Aggregated)': 'W+NEWS(Agg)',
        'AEROLINEAS (OneHot=False) y NEWS (OneHot)': 'AERO(NoOH)+NEWS(OH)',
        'AEROLINEAS (OneHot=True) y NEWS (OneHot)': 'AERO(OH)+NEWS(OH)',
        'AEROLINEAS (OneHot=False) y NEWS (Aggregated)': 'AERO(NoOH)+NEWS(Agg)',
        'AEROLINEAS (OneHot=True) y NEWS (Aggregated)': 'AERO(OH)+NEWS(Agg)',
        'WEATHER y AEROLINEAS (OneHot=False) y NEWS (OneHot)': 'W+AERO(NoOH)+NEWS(OH)',
        'WEATHER y AEROLINEAS (OneHot=True) y NEWS (OneHot)': 'W+AERO(OH)+NEWS(OH)',
        'WEATHER y AEROLINEAS (OneHot=False) y NEWS (Aggregated)': 'W+AERO(NoOH)+NEWS(Agg)',
        'WEATHER y AEROLINEAS (OneHot=True) y NEWS (Aggregated)': 'W+AERO(OH)+NEWS(Agg)'
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
    
    # Pivot tables para cada métrica
    pivot_mae = df.pivot_table(values='mae', index='config_short', columns='model', aggfunc='mean')
    pivot_rmse = df.pivot_table(values='rmse', index='config_short', columns='model', aggfunc='mean')
    pivot_r2 = df.pivot_table(values='r2', index='config_short', columns='model', aggfunc='mean')
    
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
    
    # Añadir leyenda de configuraciones
    legend_text = "Leyenda de Configuraciones:\n" + "\n".join([f"{short}: {long}" for long, short in config_map.items() if long in df['config'].unique()])
    fig.text(0.02, 0.02, legend_text, fontsize=8, verticalalignment='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])
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
    
    # Añadir leyenda de configuraciones
    legend_text = "Leyenda de Configuraciones:\n" + "\n".join([f"{short}: {long}" for long, short in config_map.items() if long in df['config'].unique()])
    fig.text(0.02, 0.02, legend_text, fontsize=7, verticalalignment='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.20, 1, 0.95])
    return fig

def main():
    """
    Función principal para generar todas las visualizaciones.
    """
    print("🔍 Analizando resultados del entrenamiento...")
    
    # Parsear resultados
    df = parse_train_results('train.txt')
    
    if df.empty:
        print("❌ No se encontraron resultados en el archivo train.txt")
        return
    
    print(f"✅ Se encontraron {len(df)} resultados")
    print(f"📊 Modelos: {', '.join(df['model'].unique())}")
    print(f"⚙️  Configuraciones: {len(df['config'].unique())}")
    
    # Crear gráficos
    print("📈 Generando heatmaps de métricas...")
    fig1 = create_heatmaps(df)
    fig1.savefig('heatmaps_metrics.png', dpi=300, bbox_inches='tight')
    
    print("📊 Generando gráficos combinados...")
    fig2 = create_combined_plots(df)
    fig2.savefig('combined_analysis.png', dpi=300, bbox_inches='tight')
    
    # Estadísticas resumen
    print("\n📋 RESUMEN DE RESULTADOS:")
    print("=" * 50)
    
    # Mejor modelo general
    best_overall = df.loc[df['mae'].idxmin()]
    print(f"🏆 Mejor modelo general: {best_overall['model']} (MAE: {best_overall['mae']:.2f})")
    print(f"   Configuración: {best_overall['config']}")
    
    # Mejor modelo por tipo
    print("\n🎯 Mejores modelos por tipo:")
    for model in df['model'].unique():
        best_model = df[df['model'] == model].loc[df[df['model'] == model]['mae'].idxmin()]
        print(f"   {model}: MAE {best_model['mae']:.2f} ({best_model['config']})")
    
    # Configuración con mejor rendimiento
    best_config = df.groupby('config')['mae'].mean().idxmin()
    best_config_mae = df.groupby('config')['mae'].mean().min()
    print(f"\n⚙️  Mejor configuración: {best_config}")
    print(f"   MAE promedio: {best_config_mae:.2f}")
    
    print(f"\n💾 Gráficos guardados:")
    print(f"   - heatmaps_metrics.png (MAE, RMSE, R²)")
    print(f"   - combined_analysis.png (4 gráficos combinados)")
    
    plt.show()

if __name__ == "__main__":
    main()
