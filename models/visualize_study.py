"""
Script para visualizar los resultados del estudio de optimización de hiperparámetros.
"""
import os
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)


# Configuración global de estilo
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update(
    {
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11
    }
)

def load_study(study_name='aircraft_forecasting', storage=None):
    """Carga el estudio de Optuna."""
    if storage is None:
        # Si no se especifica storage, asumimos que es un estudio en memoria
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_name}.db")
    else:
        study = optuna.load_study(study_name=study_name, storage=storage)
    return study

def plot_optimization_summary(study):
    """Muestra un resumen visual estilizado de la optimización."""
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={"height_ratios": [2, 1.4]})
    fig.suptitle('Resumen de la Optimización de Hiperparámetros', fontsize=20, weight='bold', y=0.98)

    trials_df = study.trials_dataframe().sort_values('number')
    completed = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    best_trial = study.best_trial
    best_algorithm = best_trial.params.get('algorithm', 'N/D')

    # 1. Historial de optimización
    ax_hist = axs[0, 0]
    if not completed.empty:
        completed['mae'] = completed['value'].astype(float)
        completed['best_so_far'] = completed['mae'].cummin()

        sns.lineplot(data=completed, x='number', y='mae', ax=ax_hist, label='MAE por trial', linewidth=2)
        sns.lineplot(data=completed, x='number', y='best_so_far', ax=ax_hist,
                     label='Mejor MAE acumulado', linewidth=2.5, linestyle='--', color='#2ca02c')
        ax_hist.set_title('Historial de Optimización (todos los algoritmos)', weight='bold')
        ax_hist.set_xlabel('Número de trial')
        ax_hist.set_ylabel('MAE')
        ax_hist.set_yscale('log')
        ax_hist.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
        ax_hist.legend(loc='upper right')
    else:
        ax_hist.text(0.5, 0.5, 'No hay trials completados', ha='center', va='center')

    # 2. Importancia de parámetros
    ax_importance = axs[0, 1]
    best_algo = best_trial.params.get('algorithm')
    
    if best_algo is None:
        ax_importance.text(0.5, 0.5, 'No se pudo identificar el algoritmo ganador', ha='center', va='center')
    else:
        # Usar la función de importancia pasando el eje existente
        plot_hyperparameter_importance(study, ax=ax_importance)
        ax_importance.set_title(f'Importancia de Hiperparámetros\n({best_algo})', weight='bold')

    # 3. Mejores parámetros en formato tabla
    ax_params = axs[1, 0]
    ax_params.axis('off')
    try:
        best_params = study.best_params
        rows = []
        if 'algorithm' in best_params:
            rows.append(['algorithm', f"{best_algorithm}"])
        rows.extend([[k, f"{v}"] for k, v in best_params.items() if k != 'algorithm'])
        table = ax_params.table(cellText=rows, colLabels=['Parámetro', 'Valor'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.4)
        ax_params.set_title('Mejores Parámetros', weight='bold', pad=20)
    except Exception:
        ax_params.text(0.5, 0.5, 'No se pudieron obtener los mejores parámetros', ha='center', va='center')

    # 4. Distribución de valores
    ax_dist = axs[1, 1]
    if not completed.empty:
        sns.histplot(data=completed, x='mae', bins=20, kde=True, ax=ax_dist, color='#1f77b4')
        ax_dist.set_title('Distribución del MAE (todos los algoritmos)', weight='bold')
        ax_dist.set_xlabel('MAE')
        ax_dist.set_ylabel('Frecuencia')
        ax_dist.set_yscale('log')
        ax_dist.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    else:
        ax_dist.text(0.5, 0.5, 'Sin datos suficientes para el histograma', ha='center', va='center')

    fig.text(
        0.5,
        0.015,
        'El historial y la distribución combinan todos los algoritmos evaluados. La tabla y los gráficos de '
        f'importancia/relaciones se centran en el mejor modelo encontrado: {best_algorithm}.',
        ha='center',
        fontsize=11
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    return fig

def plot_algorithm_comparison(study):
    """Compara el rendimiento de los algoritmos con métricas claras."""
    trials_df = study.trials_dataframe()
    if 'params_algorithm' not in trials_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No se registró el parámetro "algorithm" en los trials', ha='center', va='center')
        return fig

    completed = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    completed['mae'] = completed['value'].astype(float)
    completed['algorithm'] = completed['params_algorithm']

    if completed.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No hay trials completados para comparar', ha='center', va='center')
        return fig

    summary = (completed.groupby('algorithm')['mae']
               .agg(['median', 'min', 'count'])
               .rename(columns={'median': 'MAE mediano', 'min': 'Mejor MAE', 'count': 'Trials'}))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})

    ax_box, ax_bar = axes
    sns.boxplot(data=completed, x='algorithm', y='mae', ax=ax_box, hue='algorithm', palette='viridis', legend=False)
    sns.stripplot(data=completed, x='algorithm', y='mae', ax=ax_box, color='black', size=5, alpha=0.6, jitter=True)
    ax_box.set_title('Distribución del MAE por Algoritmo', weight='bold')
    ax_box.set_xlabel('Algoritmo')
    ax_box.set_ylabel('MAE (escala log)')
    ax_box.set_yscale('log')
    ax_box.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax_box.axhline(summary['Mejor MAE'].min(), color='red', linestyle='--', linewidth=1.5, label='Mejor MAE global')
    ax_box.legend(loc='upper right')

    summary_sorted = summary.sort_values('Mejor MAE')
    summary_sorted['Algorithm'] = summary_sorted.index
    sns.barplot(data=summary_sorted, y='Algorithm', x='Mejor MAE', ax=ax_bar, palette='mako')
    for i, (mae, count) in enumerate(zip(summary_sorted['Mejor MAE'], summary_sorted['Trials'])):
        ax_bar.text(mae, i, f"{mae:.2f}\n({count} trials)", va='center', ha='left', fontsize=11)
    ax_bar.set_title('Mejor MAE por Algoritmo', weight='bold')
    ax_bar.set_xlabel('Mejor MAE')
    ax_bar.set_ylabel('')
    ax_bar.set_xscale('log')
    ax_bar.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

    plt.tight_layout()
    return fig

def plot_hyperparameter_importance(study, ax=None):
    """Muestra la importancia de los hiperparámetros con anotaciones."""
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    best_trial = study.best_trial
    best_algo = best_trial.params.get('algorithm')

    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df['state'] == 'COMPLETE'].copy()

    if best_algo is None:
        ax.text(0.5, 0.5, 'El parámetro "algorithm" no está disponible en el mejor trial', 
               ha='center', va='center')
        return fig if create_fig else None

    algo_trials = completed[completed['params_algorithm'] == best_algo].copy()
    if len(algo_trials) < 3:
        ax.text(0.5, 0.5, f'No hay suficientes trials para {best_algo}', 
               ha='center', va='center')
        return fig if create_fig else None

    algo_trials['mae'] = algo_trials['value'].astype(float)

    param_cols = [
        col for col in algo_trials.columns
        if col.startswith('params_')
        and col != 'params_algorithm'
        and algo_trials[col].notna().any()
    ]
    
    if not param_cols:
        ax.text(0.5, 0.5, f'{best_algo} no aporta hiperparámetros adicionales', 
               ha='center', va='center')
        return fig if create_fig else None

    def _encode_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int)
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        return series.astype(str).astype('category').cat.codes.astype(float)

    importances = {}
    for col in param_cols:
        encoded = _encode_series(algo_trials[col])
        if encoded.dropna().nunique() <= 1:
            continue
        corr = encoded.corr(algo_trials['mae'], method='spearman')
        if pd.notnull(corr):
            importances[col] = abs(corr)

    if not importances:
        ax.text(0.5, 0.5, 
               f'Hiperparámetros de {best_algo} sin variación suficiente para calcular importancia',
               ha='center', va='center')
        return fig if create_fig else None

    importance_df = (pd.DataFrame({
        'Parámetro': [p.replace('params_', '').replace('_', ' ').title() for p in importances.keys()],
        'Importancia': list(importances.values())
    })
    .loc[lambda df: df['Importancia'] > 0]
    .sort_values('Importancia', ascending=True))

    total = importance_df['Importancia'].sum()
    if total == 0:
        ax.text(0.5, 0.5,
               f'No se detectó relación monotónica clara entre los hiperparámetros de {best_algo} y el MAE',
               ha='center', va='center')
        return fig if create_fig else None

    importance_df['Importancia (%)'] = 100 * importance_df['Importancia'] / total

    # Limpiar el eje antes de dibujar
    ax.clear()
    
    # Crear el gráfico de barras con hue para evitar la advertencia
    sns.barplot(data=importance_df, x='Importancia (%)', y='Parámetro', 
                hue='Parámetro', legend=False, ax=ax, palette='crest')
    
    # Añadir las etiquetas de porcentaje
    for i, pct in enumerate(importance_df['Importancia (%)']):
        ax.text(pct + 1, i, f"{pct:.1f}%", va='center', fontsize=11)

    ax.set_title(f'Importancia de Hiperparámetros para {best_algo}', weight='bold')
    ax.set_xlabel('Contribución relativa (Spearman |ρ|)')
    ax.set_ylabel('Parámetro')
    
    if create_fig:
        plt.tight_layout()
        return fig
    return None

def plot_hyperparameter_relationships(study):
    """Explora la relación entre los hiperparámetros numéricos y el MAE."""
    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df['state'] == 'COMPLETE'].copy()

    best_algo = study.best_trial.params.get('algorithm')
    if best_algo is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No se pudo identificar el algoritmo ganador', ha='center', va='center')
        return fig

    algo_trials = completed[completed['params_algorithm'] == best_algo].copy()
    if len(algo_trials) < 3:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Se necesitan más trials para {best_algo}', ha='center', va='center')
        return fig

    algo_trials['mae'] = algo_trials['value'].astype(float)

    def _encode_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int)
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        return series.astype(str).astype('category').cat.codes.astype(float)

    param_cols = [
        col for col in algo_trials.columns
        if col.startswith('params_')
        and col != 'params_algorithm'
        and algo_trials[col].notna().any()
    ]
    if not param_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'{best_algo} no cuenta con hiperparámetros numéricos', ha='center', va='center')
        return fig

    encoded_df = {}
    for col in param_cols:
        encoded = _encode_series(algo_trials[col])
        if encoded.dropna().nunique() <= 1:
            continue
        encoded_df[col] = encoded

    if not encoded_df:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Hiperparámetros de {best_algo} sin variación suficiente', ha='center', va='center')
        return fig

    encoded_df = pd.DataFrame(encoded_df)
    encoded_df['mae'] = algo_trials['mae'].values

    params = list(encoded_df.columns)
    params.remove('mae')
    n_params = len(params)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), squeeze=False)

    for idx, param in enumerate(params):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        sns.scatterplot(data=encoded_df, x=param, y='mae', ax=ax, alpha=0.8, s=60)
        sns.regplot(data=encoded_df, x=param, y='mae', scatter=False, ax=ax, color='black', line_kws={'lw': 2})
        ax.set_title(f'{param.replace("params_", "").replace("_", " ").title()} vs MAE', weight='bold')
        ax.set_xlabel(param.replace('params_', '').replace('_', ' ').title())
        ax.set_ylabel('MAE (log)')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))

    for idx in range(n_params, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis('off')

    fig.suptitle(f'Relaciones entre Hiperparámetros y MAE para {best_algo}', fontsize=20, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def save_plots(study, output_dir='visualizations'):
    """Guarda todas las visualizaciones en archivos."""
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar y guardar gráficos
    fig1 = plot_optimization_summary(study)
    fig1.savefig(f'{output_dir}/optimization_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = plot_algorithm_comparison(study)
    fig2.savefig(f'{output_dir}/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = plot_hyperparameter_importance(study)
    fig3.savefig(f'{output_dir}/hyperparameter_importance.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    try:
        fig4 = plot_hyperparameter_relationships(study)
        fig4.savefig(f'{output_dir}/hyperparameter_relationships.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
    except Exception as e:
        print(f"No se pudo generar el gráfico de relaciones: {e}")

def show_plots_interactive(study):
    """
    Muestra los gráficos de forma interactiva.
    """
    # 1. Mostrar resumen de optimización
    try:
        fig = plot_optimization_summary(study)
        fig.suptitle('Resumen de la Optimización (Interactivo)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    except Exception as e:
        print(f"Error mostrando el resumen de optimización: {e}")
    
    # 2. Mostrar gráficos individuales de Optuna
    try:
        # Importancia de parámetros
        print("\nMostrando gráficos individuales de Optuna...")
        
        # Importancia de parámetros
        try:
            print("- Mostrando importancia de parámetros...")
            fig = plot_param_importances(study)
            fig.show()
        except Exception as e:
            print(f"  No se pudo mostrar la importancia de parámetros: {e}")
        
        # Coordenadas paralelas (si hay suficientes trials)
        if len(study.trials) > 5:
            try:
                print("- Mostrando coordenadas paralelas...")
                fig = plot_parallel_coordinate(study)
                fig.show()
            except Exception as e:
                print(f"  No se pudo mostrar coordenadas paralelas: {e}")
        
        # Slice plot
        try:
            print("- Mostrando slice plot...")
            fig = plot_slice(study)
            fig.show()
        except Exception as e:
            print(f"  No se pudo mostrar el slice plot: {e}")
            
    except Exception as e:
        print(f"Error mostrando gráficos de Optuna: {e}")
    
    # 3. Mostrar comparación de algoritmos
    try:
        print("\nMostrando comparación de algoritmos...")
        fig = plot_algorithm_comparison(study)
        if fig is not None:
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"No se pudo mostrar la comparación de algoritmos: {e}")

if __name__ == "__main__":
    # Cargar el estudio
    try:
        # Usar ruta relativa desde donde se ejecuta el script
        db_path = os.path.join(os.path.dirname(__file__), 'optuna_storage/aircraft_forecasting.db')
        storage_url = f"sqlite:///{db_path}"
        print(f"Intentando cargar estudio desde: {storage_url}")
        study = load_study(study_name="aircraft_forecasting_study", storage=storage_url)
        
        # Mostrar información básica
        print(f"Número de trials: {len(study.trials)}")
        print(f"Mejor valor (MAE): {study.best_value:.4f}")
        print("Mejores parámetros:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Preguntar al usuario qué acción realizar
        while True:
            print("\nOpciones de visualización:")
            print("1. Mostrar gráficos interactivos")
            print("2. Guardar gráficos en archivos")
            print("3. Ambas opciones")
            print("4. Salir")
            
            choice = input("Seleccione una opción (1-4): ")
            
            if choice == '1':
                show_plots_interactive(study)
            elif choice == '2':
                save_plots(study)
                print("\nGráficos guardados en el directorio 'visualizations/'")
            elif choice == '3':
                show_plots_interactive(study)
                save_plots(study)
                print("\nGráficos guardados en el directorio 'visualizations/'")
            elif choice == '4':
                break
            else:
                print("Opción no válida. Intente nuevamente.")
        
    except Exception as e:
        print(f"Error al cargar el estudio: {e}")
        print("Asegúrate de que el estudio existe y está accesible.")
