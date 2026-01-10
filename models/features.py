"""
Ingeniería de características para modelos de forecasting de aeronaves.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from .config import ModelConfig


class AircraftFeatureEngineer:
    """
    Ingeniero de características para series temporales de aeronaves.

    Crea features temporales, lags, estadísticas móviles y otras
    características útiles para forecasting.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_features(self, df: pd.DataFrame,
                       feature_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Crea todas las características configuradas.

        Args:
            df: DataFrame con datos temporales
            feature_config: Configuración de features (usa config por defecto si None)

        Returns:
            DataFrame con features añadidas
        """
        if feature_config is None:
            feature_config = self.config.feature_config

        self.logger.info("Iniciando creación de características")

        df_featured = df.copy()

        # Features temporales
        if feature_config.get('temporal_features', True):
            df_featured = self._add_temporal_features(df_featured)

        # Features de lag
        if feature_config.get('daily_lag_features') or feature_config.get('hourly_lag_features'):
            # Determinar lags según la frecuencia de los datos
            if hasattr(df.index, 'freq') and str(df.index.freq) == 'h':
                lags = feature_config.get('hourly_lag_features', [1, 24, 168, 720])
            else:
                lags = feature_config.get('daily_lag_features', [1, 7, 14, 30])
            
            df_featured = self._add_lag_features(df_featured, lags)

        # Features móviles
        if feature_config.get('daily_rolling_features') or feature_config.get('hourly_rolling_features'):
            # Determinar ventanas según la frecuencia de los datos
            if hasattr(df.index, 'freq') and str(df.index.freq) == 'h':
                windows = feature_config.get('hourly_rolling_features', [24, 168, 720])
            else:
                windows = feature_config.get('daily_rolling_features', [7, 14, 30])
            
            df_featured = self._add_rolling_features(df_featured, windows)

        # Features estacionales
        if feature_config.get('seasonal_features', True):
            df_featured = self._add_seasonal_features(df_featured)

        # Features de días festivos
        if feature_config.get('holiday_features', True):
            df_featured = self._add_holiday_features(df_featured)

        # Ajuste COVID (si aplica)
        if feature_config.get('covid_adjustment', True):
            df_featured = self._add_covid_adjustment(df_featured)

        # Features meteorológicas (si están disponibles)
        if hasattr(self, "_has_weather_data" ) and self._has_weather_data(df_featured):
            df_featured = self._add_weather_features(df_featured)

        # Features de eventos noticiosos (si están disponibles)
        if hasattr(self, "_has_news_data") and self._has_news_data(df_featured):
            df_featured = self._add_news_event_features(df_featured)

        # Limpiar NaN generados por lags/rolling
        df_featured = df_featured.bfill().ffill()

        self.logger.info(f"Características creadas: {len(df_featured.columns)} columnas totales")

        return df_featured

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características temporales básicas.
        """
        # Día de la semana (0=Lunes, 6=Domingo)
        df['day_of_week'] = df.index.dayofweek

        # Mes del año
        df['month'] = df.index.month

        # Día del mes
        df['day'] = df.index.day

        # Hora del día (solo para datos horarios)
        if hasattr(df.index, 'hour') and str(df.index.freq) == 'h':
            df['hour'] = df.index.hour
            # Features cíclicas para hora
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Trimestre
        df['quarter'] = df.index.quarter

        # Semana del año
        df['week_of_year'] = df.index.isocalendar().week

        # Features cíclicas para día de la semana
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Features cíclicas para mes
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Es fin de semana
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Es hora pico (solo para datos horarios)
        if hasattr(df.index, 'hour') and str(df.index.freq) == 'h':
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
            df['is_rush_hour'] = df['is_rush_hour'].astype(int)

        return df

    def _add_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """
        Añade características de lag (valores anteriores).

        Args:
            df: DataFrame con datos
            lags: Lista de lags a crear (ej: [1, 7, 14])
        """
        target_col = self.config.target_column

        if target_col not in df.columns:
            self.logger.warning(f"Columna target '{target_col}' no encontrada para lags")
            return df

        for lag in lags:
            lag_col = f'{target_col}_lag_{lag}'
            df[lag_col] = df[target_col].shift(lag)

        self.logger.info(f"Features de lag añadidas: {lags}")

        return df

    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Añade características móviles (rolling statistics).

        Args:
            df: DataFrame con datos
            windows: Lista de ventanas para estadísticas móviles
        """
        target_col = self.config.target_column

        if target_col not in df.columns:
            self.logger.warning(f"Columna target '{target_col}' no encontrada para rolling")
            return df

        for window in windows:
            # Media móvil
            df[f'{target_col}_rolling_mean_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )

            # Desviación estándar móvil
            df[f'{target_col}_rolling_std_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )

            # Mínimo móvil
            df[f'{target_col}_rolling_min_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).min()
            )

            # Máximo móvil
            df[f'{target_col}_rolling_max_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).max()
            )

        self.logger.info(f"Features móviles añadidas: ventanas {windows}")

        return df

    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características estacionales.
        """
        # Codificación sinusoidal del día de la semana
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Codificación sinusoidal del mes
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # Tendencia temporal (días desde el inicio)
        df['trend'] = (df.index - df.index.min()).days

        # Estacionalidad semanal (alta/baja según día)
        weekly_pattern = {
            0: 0.8,  # Lunes - tráfico moderado
            1: 0.9,  # Martes
            2: 1.0,  # Miércoles - pico
            3: 0.95, # Jueves
            4: 0.85, # Viernes
            5: 0.6,  # Sábado - bajo
            6: 0.5   # Domingo - bajo
        }
        df['weekly_seasonality'] = df.index.dayofweek.map(weekly_pattern)

        self.logger.info("Features estacionales añadidas")

        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características relacionadas con días festivos.
        """
        # Lista de fechas festivas cubanas aproximadas
        cuban_holidays = [
            # 2023
            '2023-01-01',  # Año Nuevo
            '2023-05-01',  # Día del Trabajo
            '2023-07-25',  # Día de la Rebeldía Nacional
            '2023-07-26',  # Día de la Rebeldía Nacional
            '2023-10-10',  # Independencia
            '2023-12-25',  # Navidad
            # 2024
            '2024-01-01',  # Año Nuevo
            '2024-05-01',  # Día del Trabajo
            '2024-07-25',  # Día de la Rebeldía Nacional
            '2024-07-26',  # Día de la Rebeldía Nacional
            '2024-10-10',  # Independencia
            '2024-12-25',  # Navidad
            # 2025
            '2025-01-01',  # Año Nuevo
            '2025-05-01',  # Día del Trabajo
            '2025-07-25',  # Día de la Rebeldía Nacional
            '2025-07-26',  # Día de la Rebeldía Nacional
            '2025-10-10',  # Independencia
            '2025-12-25',  # Navidad
        ]

        # Convertir a datetime
        holiday_dates = pd.to_datetime(cuban_holidays)

        # Crear feature de día festivo
        df['is_holiday'] = df.index.isin(holiday_dates).astype(int)

        # Crear feature de día anterior a festivo
        df['is_day_before_holiday'] = (
            (df.index + pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
        )

        # Crear feature de día siguiente a festivo
        df['is_day_after_holiday'] = (
            (df.index - pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
        )

        self.logger.info("Features de días festivos añadidas")

        return df

    def _add_covid_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade ajuste por impacto del COVID-19 en el tráfico aéreo.
        """
        # Períodos de COVID con restricciones de vuelo
        covid_periods = [
            ('2020-03-01', '2020-12-31'),  # COVID inicial
            ('2021-01-01', '2021-12-31'),  # Continuación restricciones
        ]

        df['covid_impact'] = 0

        for start, end in covid_periods:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)

            # Dentro del período COVID
            mask = (df.index >= start_date) & (df.index <= end_date)
            df.loc[mask, 'covid_impact'] = 1

            # Período de recuperación (6 meses después)
            recovery_end = end_date + pd.DateOffset(months=6)
            recovery_mask = (df.index > end_date) & (df.index <= recovery_end)
            df.loc[recovery_mask, 'covid_impact'] = 0.5

        # Nota: Los datos reales empiezan en 2022, así que este ajuste
        # es más relevante para datos históricos si se extienden
        covid_count = (df['covid_impact'] > 0).sum()
        if covid_count > 0:
            self.logger.info(f"Ajuste COVID aplicado a {covid_count} registros")
        else:
            self.logger.info("Ajuste COVID: no aplica a este rango de fechas")

        return df

    def create_lagged_target(self, df: pd.DataFrame,
                           forecast_horizon: int = 1) -> pd.DataFrame:
        """
        Crea la variable target con lag para forecasting.

        Args:
            df: DataFrame con datos
            forecast_horizon: Horizonte de predicción (ej: 1 para 1 día adelante)

        Returns:
            DataFrame con target lagged
        """
        target_col = self.config.target_column

        if target_col not in df.columns:
            raise ValueError(f"Columna target '{target_col}' no encontrada")

        # Crear target lagged (lo que queremos predecir)
        df[f'{target_col}_target_h{forecast_horizon}'] = (
            df[target_col].shift(-forecast_horizon)
        )

        # Remover filas con NaN en el target
        df = df.dropna(subset=[f'{target_col}_target_h{forecast_horizon}'])

        self.logger.info(f"Target creado con horizonte {forecast_horizon}: {len(df)} registros válidos")

        return df

    def select_features_for_model(self, df: pd.DataFrame,
                                model_type: str = 'regression') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Selecciona y prepara features para un tipo específico de modelo.

        Args:
            model_type: Tipo de modelo ('regression', 'lstm', 'prophet')

        Returns:
            Tupla (X, y) con features y target
        """
        target_col = self.config.target_column

        # Encontrar columna target (puede tener sufijo de horizonte)
        target_columns = [col for col in df.columns if col.startswith(f'{target_col}_target')]
        if not target_columns:
            # Si no hay target lagged, usar el target original
            y = df[target_col]
        else:
            # Usar el primer target lagged encontrado
            y = df[target_columns[0]]

        # Seleccionar features (excluir targets y columnas no numéricas problemáticas)
        exclude_cols = (
            [target_col] +
            target_columns +
            ['created']  # Columna de timestamp de creación si existe
        )

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Solo columnas numéricas
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]

        self.logger.info(f"Features seleccionadas para {model_type}: {len(X.columns)} features, {len(y)} targets")

        return X, y

    def get_feature_importance_report(self, df: pd.DataFrame) -> Dict:
        """
        Genera un reporte de importancia de características.

        Args:
            df: DataFrame con features

        Returns:
            Diccionario con estadísticas de features
        """
        target_col = self.config.target_column

        if target_col not in df.columns:
            return {}

        # Correlaciones con target
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)

        # Features más correlacionadas (excluyendo el target mismo)
        top_features = correlations.drop(target_col).head(10)

        report = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'top_correlated_features': top_features.to_dict(),
            'feature_stats': {
                'mean_correlation': correlations.mean(),
                'max_correlation': correlations.max(),
                'features_with_high_corr': (correlations > 0.7).sum()
            }
        }

        return report