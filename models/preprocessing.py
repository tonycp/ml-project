"""
Preprocesamiento de datos para modelos de forecasting de aeronaves.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

from .config import ModelConfig


class AircraftDataPreprocessor:
    """
    Preprocesador de datos para series temporales de aeronaves.

    Maneja limpieza de datos, tratamiento de outliers, interpolación
    de valores faltantes y preparación para modelado.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def preprocess_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa datos diarios de ATC.

        Args:
            df: DataFrame con datos diarios

        Returns:
            DataFrame preprocesado
        """
        self.logger.info("Iniciando preprocesamiento de datos diarios")

        # Crear copia para no modificar original
        df_processed = df.copy()

        # 1. Verificar y asegurar frecuencia diaria
        df_processed = self._ensure_daily_frequency(df_processed)

        # 2. Tratar valores faltantes
        df_processed = self._handle_missing_values(df_processed)

        # 3. Detectar y tratar outliers
        df_processed = self._handle_outliers(df_processed)

        # 4. Validar integridad de datos
        df_processed = self._validate_data_integrity(df_processed)

        # 5. Añadir metadatos de procesamiento
        df_processed.attrs['preprocessing_info'] = {
            'original_records': len(df),
            'processed_records': len(df_processed),
            'missing_values_handled': True,
            'outliers_handled': True,
            'frequency': 'D'
        }

        self.logger.info(f"Preprocesamiento completado: {len(df_processed)} registros")

        return df_processed

    def preprocess_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa datos horarios ATFM.

        Args:
            df: DataFrame con datos horarios

        Returns:
            DataFrame preprocesado
        """
        self.logger.info("Iniciando preprocesamiento de datos horarios")

        df_processed = df.copy()

        # 1. Verificar y asegurar frecuencia horaria
        df_processed = self._ensure_hourly_frequency(df_processed)

        # 2. Tratar valores faltantes
        df_processed = self._handle_missing_values(df_processed)

        # 3. Suavizar datos horarios (pueden ser muy variables)
        df_processed = self._smooth_hourly_data(df_processed)

        # 4. Validar integridad
        df_processed = self._validate_data_integrity(df_processed)

        df_processed.attrs['preprocessing_info'] = {
            'original_records': len(df),
            'processed_records': len(df_processed),
            'missing_values_handled': True,
            'smoothing_applied': True,
            'frequency': 'H'
        }

        self.logger.info(f"Preprocesamiento horario completado: {len(df_processed)} registros")

        return df_processed

    def _ensure_daily_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Asegura que los datos tengan frecuencia diaria completa.
        Rellena días faltantes con interpolación.
        """
        # Reindexar para asegurar frecuencia diaria
        date_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='D'
        )

        df_reindexed = df.reindex(date_range)

        # Interpolar valores faltantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_reindexed[numeric_cols] = df_reindexed[numeric_cols].interpolate(method='linear')

        # Para valores faltantes al inicio/final, usar forward/backward fill
        df_reindexed = df_reindexed.fillna(method='ffill').fillna(method='bfill')

        self.logger.info(f"Frecuencia diaria asegurada: {len(df_reindexed)} días (original: {len(df)})")

        return df_reindexed

    def _ensure_hourly_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Asegura que los datos tengan frecuencia horaria completa.
        """
        date_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='h'
        )

        df_reindexed = df.reindex(date_range)

        # Interpolar valores numéricos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_reindexed[numeric_cols] = df_reindexed[numeric_cols].interpolate(method='linear')

        # Fill missing values
        df_reindexed = df_reindexed.ffill().bfill()

        self.logger.info(f"Frecuencia horaria asegurada: {len(df_reindexed)} horas (original: {len(df)})")

        return df_reindexed

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja valores faltantes en el dataset.
        """
        missing_before = df.isnull().sum().sum()

        # Para series temporales, usar interpolación lineal
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

        # Para valores al inicio/final, usar forward/backward fill
        df = df.ffill().bfill()

        missing_after = df.isnull().sum().sum()

        if missing_before > 0:
            self.logger.info(f"Valores faltantes manejados: {missing_before} -> {missing_after}")

        return df

    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Detecta y trata outliers en la serie temporal.

        Args:
            df: DataFrame con datos
            method: Método para detectar outliers ('iqr', 'zscore', 'rolling')
        """
        target_col = self.config.target_column

        if target_col not in df.columns:
            return df

        original_values = df[target_col].copy()

        if method == 'iqr':
            # Método IQR (Interquartile Range)
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Reemplazar outliers con límites
            df[target_col] = np.clip(df[target_col], lower_bound, upper_bound)

        elif method == 'zscore':
            # Método Z-Score
            z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
            threshold = 3

            # Reemplazar outliers con mediana
            median_val = df[target_col].median()
            df.loc[z_scores > threshold, target_col] = median_val

        elif method == 'rolling':
            # Método basado en media móvil
            rolling_mean = df[target_col].rolling(window=30, center=True).mean()
            rolling_std = df[target_col].rolling(window=30, center=True).std()

            # Calcular límites dinámicos
            lower_bound = rolling_mean - 2 * rolling_std
            upper_bound = rolling_mean + 2 * rolling_std

            # Aplicar límites
            df[target_col] = np.where(
                df[target_col] < lower_bound,
                lower_bound,
                np.where(df[target_col] > upper_bound, upper_bound, df[target_col])
            )

        outliers_handled = (original_values != df[target_col]).sum()
        if outliers_handled > 0:
            self.logger.info(f"Outliers manejados ({method}): {outliers_handled} valores ajustados")

        return df

    def _smooth_hourly_data(self, df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """
        Suaviza datos horarios para reducir variabilidad.
        """
        target_col = self.config.target_column

        if target_col in df.columns:
            # Aplicar media móvil para suavizar
            df[f'{target_col}_smoothed'] = df[target_col].rolling(
                window=window, center=True, min_periods=1
            ).mean()

            # Reemplazar valores extremos con suavizados
            original = df[target_col]
            smoothed = df[f'{target_col}_smoothed']

            # Solo reemplazar si la diferencia es significativa
            diff_ratio = np.abs(original - smoothed) / (smoothed + 1)  # +1 para evitar división por cero
            replace_mask = diff_ratio > 0.5  # 50% de diferencia

            df.loc[replace_mask, target_col] = smoothed[replace_mask]

            # Remover columna temporal
            df = df.drop(columns=[f'{target_col}_smoothed'])

            smoothed_count = replace_mask.sum()
            if smoothed_count > 0:
                self.logger.info(f"Datos suavizados: {smoothed_count} valores ajustados")

        return df

    def _validate_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida la integridad de los datos procesados.
        """
        issues = []

        # Verificar valores negativos en target
        target_col = self.config.target_column
        if target_col in df.columns:
            negative_count = (df[target_col] < 0).sum()
            if negative_count > 0:
                issues.append(f"{negative_count} valores negativos en {target_col}")
                # Corregir valores negativos
                df[target_col] = df[target_col].clip(lower=0)

        # Verificar valores faltantes
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"{missing_count} valores faltantes restantes")

        # Verificar frecuencia temporal
        if not df.index.is_monotonic_increasing:
            issues.append("Índice temporal no está ordenado")
            df = df.sort_index()

        # Verificar duplicados en índice
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} índices duplicados")
            df = df[~df.index.duplicated(keep='first')]

        if issues:
            self.logger.warning(f"Issues encontrados en validación: {issues}")
        else:
            self.logger.info("Validación de integridad: OK")

        return df

    def detect_anomalies(self, df: pd.DataFrame, method: str = 'residual') -> pd.Series:
        """
        Detecta anomalías en la serie temporal.

        Args:
            df: DataFrame con datos
            method: Método de detección ('residual', 'isolation_forest', 'prophet')

        Returns:
            Serie booleana indicando anomalías
        """
        target_col = self.config.target_column

        if target_col not in df.columns:
            return pd.Series(False, index=df.index)

        anomalies = pd.Series(False, index=df.index)

        if method == 'residual':
            # Método simple: residuos de media móvil
            rolling_mean = df[target_col].rolling(window=7, center=True).mean()
            rolling_std = df[target_col].rolling(window=7, center=True).std()

            residuals = np.abs(df[target_col] - rolling_mean)
            threshold = 2 * rolling_std

            anomalies = residuals > threshold.fillna(residuals.std())

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest

                # Crear features simples
                features = df[[target_col]].copy()
                features['day_of_week'] = features.index.dayofweek
                features['month'] = features.index.month

                # Entrenar modelo
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                predictions = iso_forest.fit_predict(features.dropna())

                # -1 indica anomalía
                anomalies = pd.Series(predictions == -1, index=features.index)

            except ImportError:
                self.logger.warning("scikit-learn no disponible para Isolation Forest")
                return pd.Series(False, index=df.index)

        self.logger.info(f"Anomalías detectadas ({method}): {anomalies.sum()}")

        return anomalies

    def get_preprocessing_report(self, original_df: pd.DataFrame,
                               processed_df: pd.DataFrame) -> Dict:
        """
        Genera un reporte de preprocesamiento.

        Args:
            original_df: DataFrame original
            processed_df: DataFrame procesado

        Returns:
            Diccionario con estadísticas de preprocesamiento
        """
        target_col = self.config.target_column

        report = {
            'original_records': len(original_df),
            'processed_records': len(processed_df),
            'missing_values_original': original_df.isnull().sum().sum(),
            'missing_values_processed': processed_df.isnull().sum().sum(),
            'date_range': {
                'original': f"{original_df.index.min()} to {original_df.index.max()}",
                'processed': f"{processed_df.index.min()} to {processed_df.index.max()}"
            }
        }

        if target_col in original_df.columns and target_col in processed_df.columns:
            report['target_stats'] = {
                'original': {
                    'mean': original_df[target_col].mean(),
                    'std': original_df[target_col].std(),
                    'min': original_df[target_col].min(),
                    'max': original_df[target_col].max()
                },
                'processed': {
                    'mean': processed_df[target_col].mean(),
                    'std': processed_df[target_col].std(),
                    'min': processed_df[target_col].min(),
                    'max': processed_df[target_col].max()
                }
            }

        return report