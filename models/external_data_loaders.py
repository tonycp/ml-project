"""
Loaders para datos externos: meteorológicos y de noticias.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

from .config import ModelConfig
from .data_loader import ATCAircraftDataLoader

class WeatherDataLoader:
    """
    Cargador de datos meteorológicos para integración con forecasting de aeronaves.

    Los datos meteorológicos pueden afectar significativamente el tráfico aéreo:
    - Viento fuerte: Reduce capacidad de pistas
    - Lluvia/Tormentas: Afecta visibilidad y operaciones
    - Temperatura extrema: Impacta en rendimiento de aeronaves
    - Nubes bajas: Reduce capacidad de aproximación
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuración de datos meteorológicos
        self.weather_config = config.external_data.get('weather', {
            'data_dir': Path('data/weather'),
            'stations': ['HAVANA_INTERNATIONAL'],  # Aeropuerto principal de Cuba
            'variables': [
                'temperature', 'humidity', 'wind_speed', 'wind_direction',
                'precipitation', 'visibility', 'cloud_cover', 'pressure'
            ]
        })

        # Ruta al archivo de datos meteorológicos reales
        self.meteorological_file = config.data_dir / config.meteorological_data_file

    def load_weather_data(self, start_date: Optional[str] = None,
                         end_date: Optional[str] = None, use_median = False) -> pd.DataFrame:
        """
        Carga datos meteorológicos históricos desde el archivo real.

        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)

        Returns:
            DataFrame con datos meteorológicos diarios
        """
        self.logger.info(f"Cargando datos meteorológicos desde: {self.meteorological_file}")

        if not self.meteorological_file.exists():
            self.logger.warning(f"Archivo meteorológico no encontrado: {self.meteorological_file}")
            self.logger.info("Generando datos sintéticos como fallback...")
            return self._generate_synthetic_weather_data(start_date, end_date)

        try:
            # Cargar datos reales
            df = pd.read_csv(self.meteorological_file)
            
            # Procesar fechas
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            # Mapear columnas del archivo real a nombres estándar (usando las columnas _mean)
            # Se podría configurar para usar _median si se prefiere
            suffix = '_median' if use_median else '_mean'
            
            column_mapping = {
                f'ff_Velocidad del Viento{suffix}': 'wind_speed',
                f'dd_Dirección del Viento{suffix}': 'wind_direction', 
                f'T_Temperatura del Aire{suffix}': 'temperature',
                f'HR_Humedad Relativa del Aire{suffix}': 'humidity',
                f'P_Presión Atmosférica{suffix}': 'pressure',
                f'R_Precipitación Acumulada{suffix}': 'precipitation',
                f'S_Radiación Solar Global{suffix}': 'solar_radiation'
            }
            
            # Verificar que todas las columnas existan
            missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Columnas faltantes con {suffix}: {missing_columns}")
                # Intentar con el otro sufijo
                fallback_suffix = '_mean' if use_median else '_median'
                fallback_mapping = {
                    f'ff_Velocidad del Viento{fallback_suffix}': 'wind_speed',
                    f'dd_Dirección del Viento{fallback_suffix}': 'wind_direction', 
                    f'T_Temperatura del Aire{fallback_suffix}': 'temperature',
                    f'HR_Humedad Relativa del Aire{fallback_suffix}': 'humidity',
                    f'P_Presión Atmosférica{fallback_suffix}': 'pressure',
                    f'R_Precipitación Acumulada{fallback_suffix}': 'precipitation',
                    f'S_Radiación Solar Global{fallback_suffix}': 'solar_radiation'
                }
                missing_fallback = [col for col in fallback_mapping.keys() if col not in df.columns]
                if missing_fallback:
                    raise ValueError(f"Columnas faltantes en el archivo: {missing_columns} y {missing_fallback}")
                column_mapping = fallback_mapping
                self.logger.info(f"Usando columnas {fallback_suffix} como fallback")
            
            # Crear DataFrame con nombres estándar
            weather_df = df[list(column_mapping.keys())].copy()
            weather_df.columns = list(column_mapping.values())
            
            # Ajustar unidades y escalas
            weather_df['wind_speed'] = weather_df['wind_speed']  # ya está en m/s
            weather_df['temperature'] = weather_df['temperature']  # ya está en °C
            weather_df['humidity'] = weather_df['humidity']  # ya está en %
            weather_df['pressure'] = weather_df['pressure']  # mantener en hPa (unidades estándar)
            weather_df['precipitation'] = weather_df['precipitation']  # ya está en mm
            
            # Generar variables adicionales basadas en los datos reales
            weather_df = self._add_derived_weather_variables(weather_df)
            
            # Filtrar por rango de fechas si se especifica
            if start_date:
                weather_df = weather_df[weather_df.index >= start_date]
            if end_date:
                weather_df = weather_df[weather_df.index <= end_date]

            self.logger.info(f"Datos meteorológicos cargados: {len(weather_df)} registros")
            return weather_df

        except Exception as e:
            self.logger.error(f"Error cargando datos meteorológicos: {e}")
            self.logger.info("Generando datos sintéticos como fallback...")
            return self._generate_synthetic_weather_data(start_date, end_date)

    def _generate_synthetic_weather_data(self, start_date: Optional[str] = None,
                                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Genera datos sintéticos como fallback cuando no hay datos reales.
        """
        if start_date and end_date:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            date_range = pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')

        np.random.seed(42)
        weather_data = []

        for date in date_range:
            base_temp = 26 + 3 * np.sin(2 * np.pi * date.dayofyear / 365)
            temperature = base_temp + np.random.normal(0, 2)
            humidity = 75 + 10 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 5)
            wind_speed = 15 + 5 * np.random.normal(0, 1)
            wind_direction = np.random.uniform(0, 360)
            rainfall_season = 1 if 5 <= date.month <= 10 else 0.3
            precipitation = np.random.exponential(2) * rainfall_season if np.random.random() < 0.3 else 0
            pressure = 1015 + np.random.normal(0, 3)

            weather_data.append({
                'date': date,
                'temperature': max(15, min(35, temperature)),
                'humidity': max(30, min(95, humidity)),
                'wind_speed': max(0, wind_speed),
                'wind_direction': wind_direction,
                'precipitation': max(0, precipitation),
                'pressure': pressure,
                'solar_radiation': 200 + np.random.normal(0, 50)
            })

        df = pd.DataFrame(weather_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        return self._add_derived_weather_variables(df)

    def _add_derived_weather_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega variables meteorológicas derivadas basadas en los datos reales.
        """

        # Indicadores de condiciones extremas basadas en umbrales reales
        df['high_winds'] = (df['wind_speed'] > 8).astype(int)  # > 8 m/s = ~30 km/h
        df['heavy_rain'] = (df['precipitation'] > 20).astype(int)  # > 20mm
        df['extreme_heat'] = (df['temperature'] > 32).astype(int)  # > 32°C
        df['extreme_cold'] = (df['temperature'] < 18).astype(int)  # < 18°C
        
        # Condiciones de tormenta basadas en combinación de factores
        df['is_storm'] = ((df['high_winds'] == 1) | (df['heavy_rain'] == 1)).astype(int)
        
        # Índice de confort meteorológico (simplificado)
        df['comfort_index'] = (
            df['temperature'] * 0.4 + 
            df['humidity'] * 0.3 + 
            df['wind_speed'] * 0.2 + 
            df['precipitation'] * 0.1
        )
        
        return df

    def _generate_extreme_weather(self, date: pd.Timestamp) -> Dict[str, int]:
        """
        Genera indicadores de condiciones meteorológicas extremas.
        Basado en patrones históricos de huracanes y tormentas en Cuba.
        """
        extreme_events = {
            'is_storm': 0,
            'is_hurricane': 0,
            'high_winds': 0,  # > 30 km/h
            'heavy_rain': 0,  # > 20mm
            'low_visibility': 0,  # < 5km
            'extreme_heat': 0,  # > 32°C
            'extreme_cold': 0   # < 18°C
        }

        # Temporada de huracanes (junio-noviembre)
        if 6 <= date.month <= 11:
            # Probabilidad baja de tormenta/huracán
            if np.random.random() < 0.02:  # 2% de probabilidad
                extreme_events['is_storm'] = 1
            if np.random.random() < 0.005:  # 0.5% de probabilidad
                extreme_events['is_hurricane'] = 1

        # Vientos fuertes (más común en invierno)
        if np.random.random() < 0.05:  # 5% de probabilidad
            extreme_events['high_winds'] = 1

        # Lluvias intensas (temporada de lluvias)
        if 5 <= date.month <= 10 and np.random.random() < 0.08:  # 8% en temporada
            extreme_events['heavy_rain'] = 1

        # Visibilidad reducida (con lluvia o neblina)
        if np.random.random() < 0.03:  # 3% de probabilidad
            extreme_events['low_visibility'] = 1

        # Temperaturas extremas
        if np.random.random() < 0.02:  # 2% de probabilidad
            extreme_events['extreme_heat'] = 1
        if np.random.random() < 0.01:  # 1% de probabilidad
            extreme_events['extreme_cold'] = 1

        return extreme_events

    def get_weather_features_for_date(self, target_date: pd.Timestamp) -> Dict[str, float]:
        """
        Obtiene features meteorológicos para una fecha específica desde datos reales.

        Args:
            target_date: Fecha objetivo

        Returns:
            Diccionario con features meteorológicos
        """
        try:
            # Cargar datos reales
            weather_df = self.load_weather_data(
                start_date=target_date.strftime('%Y-%m-%d'),
                end_date=target_date.strftime('%Y-%m-%d')
            )
            
            if weather_df.empty:
                # Fallback a valores promedio si no hay datos para la fecha
                return self._get_default_weather_features(target_date)
            
            # Obtener la fila correspondiente a la fecha objetivo
            if target_date in weather_df.index:
                row = weather_df.loc[target_date]
                return {
                    'temperature': float(row['temperature']),
                    'humidity': float(row['humidity']),
                    'wind_speed': float(row['wind_speed']),
                    'wind_direction': float(row['wind_direction']),
                    'precipitation': float(row['precipitation']),
                    'pressure': float(row['pressure']),
                    'visibility': float(row['visibility']),
                    'cloud_cover': float(row['cloud_cover']),
                    'solar_radiation': float(row.get('solar_radiation', 200)),
                    'high_winds': int(row['high_winds']),
                    'heavy_rain': int(row['heavy_rain']),
                    'low_visibility': int(row['low_visibility']),
                    'extreme_heat': int(row['extreme_heat']),
                    'extreme_cold': int(row['extreme_cold']),
                    'is_storm': int(row['is_storm']),
                    'is_bad_weather': int(row['is_storm'] or row['heavy_rain'] or row['low_visibility'])
                }
            else:
                return self._get_default_weather_features(target_date)
                
        except Exception as e:
            self.logger.error(f"Error obteniendo features para fecha {target_date}: {e}")
            return self._get_default_weather_features(target_date)

    def _get_default_weather_features(self, target_date: pd.Timestamp) -> Dict[str, float]:
        """
        Retorna valores por defecto basados en patrones climáticos de Cuba.
        """
        # Temperatura promedio por mes en La Habana
        monthly_temps = {
            1: 22, 2: 23, 3: 25, 4: 27, 5: 28, 6: 29, 7: 30, 8: 30, 9: 29, 10: 27, 11: 25, 12: 23
        }

        base_temp = monthly_temps.get(target_date.month, 25)

        return {
            'temperature': base_temp + np.random.normal(0, 2),
            'humidity': 70 + np.random.normal(0, 10),
            'wind_speed': 15 + np.random.normal(0, 5),
            'wind_direction': np.random.uniform(0, 360),
            'precipitation': np.random.exponential(1) if np.random.random() < 0.2 else 0,
            'pressure': 1015 + np.random.normal(0, 3),
            'visibility': 10 + np.random.normal(0, 2),
            'cloud_cover': 50 + np.random.normal(0, 20),
            'solar_radiation': 200 + np.random.normal(0, 50),
            'high_winds': 1 if np.random.random() < 0.05 else 0,
            'heavy_rain': 1 if np.random.random() < 0.08 else 0,
            'low_visibility': 1 if np.random.random() < 0.03 else 0,
            'extreme_heat': 1 if np.random.random() < 0.02 else 0,
            'extreme_cold': 1 if np.random.random() < 0.01 else 0,
            'is_storm': 1 if np.random.random() < 0.02 else 0,
            'is_bad_weather': 1 if np.random.random() < 0.1 else 0
        }


class NewsDataLoader:
    """
    Cargador de datos de noticias desde eventos.json.

    Los eventos noticiosos pueden afectar el tráfico aéreo:
    - Accidentes/Incidentes: Cierran espacios aéreos temporalmente
    - Eventos masivos: Aumentan demanda de vuelos
    - Crisis políticas: Afectan operaciones internacionales
    - Alertas de seguridad: Modifican rutas y procedimientos
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Ruta al archivo de eventos
        self.events_file = self.config.data_dir / self.config.news_file

    def load_news_events(self, start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        feature_type: str = 'aggregated') -> pd.DataFrame:
        """
        Carga eventos de noticias desde eventos.json y genera features.

        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            feature_type: 'aggregated' (Opción A) o 'one_hot' (Opción B)

        Returns:
            DataFrame con features de noticias por fecha
        """
        self.logger.info(f"Cargando datos de eventos desde: {self.events_file}")

        if not self.events_file.exists():
            self.logger.warning(f"Archivo de eventos no encontrado: {self.events_file}")
            return pd.DataFrame()

        try:
            # Cargar datos JSON
            with open(self.events_file, 'r', encoding='utf-8') as f:
                events_data = json.load(f)

            if not events_data:
                self.logger.warning("Archivo de eventos vacío")
                return pd.DataFrame()

            # Convertir a DataFrame
            df_events = pd.DataFrame(events_data)
            
            # Procesar fechas
            df_events['fecha'] = pd.to_datetime(df_events['fecha'])
            df_events = df_events.set_index('fecha').sort_index()

            # Filtrar por rango de fechas si se especifica
            if start_date:
                df_events = df_events[df_events.index >= start_date]
            if end_date:
                df_events = df_events[df_events.index <= end_date]

            # Generar features según el tipo solicitado
            if feature_type == 'aggregated':
                return self._create_aggregated_features(df_events)
            elif feature_type == 'one_hot':
                return self._create_one_hot_features(df_events)
            else:
                raise ValueError(f"feature_type debe ser 'aggregated' o 'one_hot', no '{feature_type}'")

        except Exception as e:
            self.logger.error(f"Error cargando eventos: {e}")
            return pd.DataFrame()

    def _create_aggregated_features(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features agregadas de noticias (Opción A).
        
        Genera features numéricas resumidas por fecha.
        """
        # Agrupar por fecha y contar eventos
        daily_features = []
        
        # Obtener rango completo de fechas para asegurar continuidad
        min_date = df_events.index.min()
        max_date = df_events.index.max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        for date in date_range:
            # Eventos del día
            day_events = df_events[df_events.index.date == date.date()]
            
            if day_events.empty:
                # Día sin eventos
                daily_features.append({
                    'fecha': date,
                    'news_count_total': 0,
                    'news_positive_count': 0,
                    'news_negative_count': 0,
                    'news_neutral_count': 0,
                    'news_sentimiento_avg': 0.0,
                    'news_confianza_avg': 0.0,
                    'news_has_cultural': 0,
                    'news_has_deportivo': 0,
                    'news_has_meteorologico': 0,
                    'news_has_politico': 0,
                    'news_has_economico': 0,
                    'news_has_social': 0,
                    'news_has_incidente': 0,
                    'news_has_regulacion': 0
                })
            else:
                # Calcular features del día
                total_count = len(day_events)
                positive_count = (day_events['sentimiento'] == 'positive').sum()
                negative_count = (day_events['sentimiento'] == 'negative').sum()
                neutral_count = (day_events['sentimiento'] == 'neutral').sum()
                
                # Sentimiento promedio (positive=1, neutral=0, negative=-1)
                sentimiento_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                sentimiento_avg = day_events['sentimiento'].map(sentimiento_map).mean()
                
                # Confianza promedio
                confianza_avg = day_events['confianza_sentimiento'].mean()
                
                # Tipos de eventos (binarios)
                tipos_presentes = set(day_events['tipo'].values)
                
                daily_features.append({
                    'fecha': date,
                    'news_count_total': total_count,
                    'news_positive_count': positive_count,
                    'news_negative_count': negative_count,
                    'news_neutral_count': neutral_count,
                    'news_sentimiento_avg': sentimiento_avg,
                    'news_confianza_avg': confianza_avg,
                    'news_has_cultural': 1 if 'CULTURAL' in tipos_presentes else 0,
                    'news_has_deportivo': 1 if 'DEPORTIVO' in tipos_presentes else 0,
                    'news_has_meteorologico': 1 if 'METEOROLOGICO' in tipos_presentes else 0,
                    'news_has_politico': 1 if 'POLITICO' in tipos_presentes else 0,
                    'news_has_economico': 1 if 'ECONOMICO' in tipos_presentes else 0,
                    'news_has_social': 1 if 'SOCIAL' in tipos_presentes else 0,
                    'news_has_incidente': 1 if 'INCIDENTE' in tipos_presentes else 0,
                    'news_has_regulacion': 1 if 'REGULACION' in tipos_presentes else 0
                })
        
        # Crear DataFrame
        df_features = pd.DataFrame(daily_features)
        df_features['fecha'] = pd.to_datetime(df_features['fecha'])
        df_features = df_features.set_index('fecha').sort_index()
        
        self.logger.info(f"Features agregadas creadas: {len(df_features)} registros, {len(df_features.columns)} columnas")
        return df_features

    def _create_one_hot_features(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Crea conteo de noticias por tipo y sentimiento (Opción B).
        
        Genera columnas con el número de noticias de cada tipo y sentimiento.
        """
        # Obtener todos los tipos y sentimientos únicos
        all_tipos = df_events['tipo'].unique()
        all_sentimientos = df_events['sentimiento'].unique()
        
        # Obtener rango completo de fechas
        min_date = df_events.index.min()
        max_date = df_events.index.max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        daily_features = []
        
        for date in date_range:
            # Eventos del día
            day_events = df_events[df_events.index.date == date.date()]
            
            # Inicializar diccionario de features
            row_features = {'fecha': date}
            
            # Conteo por tipo (no binario)
            for tipo in all_tipos:
                if not day_events.empty:
                    count = (day_events['tipo'] == tipo).sum()
                    row_features[f'news_tipo_{tipo}'] = count
                else:
                    row_features[f'news_tipo_{tipo}'] = 0
            
            # Conteo por sentimiento (no binario)
            for sentimiento in all_sentimientos:
                if not day_events.empty:
                    count = (day_events['sentimiento'] == sentimiento).sum()
                    row_features[f'news_sentimiento_{sentimiento}'] = count
                else:
                    row_features[f'news_sentimiento_{sentimiento}'] = 0
            
            daily_features.append(row_features)
        
        # Crear DataFrame
        df_features = pd.DataFrame(daily_features)
        df_features['fecha'] = pd.to_datetime(df_features['fecha'])
        df_features = df_features.set_index('fecha').sort_index()
        
        self.logger.info(f"Conteo de noticias por tipo creado: {len(df_features)} registros, {len(df_features.columns)} columnas")
        return df_features

    def get_news_features_for_date(self, target_date: pd.Timestamp) -> Dict[str, float]:
        """
        Obtiene features de noticias para una fecha específica.

        Args:
            target_date: Fecha objetivo

        Returns:
            Diccionario con features de eventos noticiosos
        """
        # En producción, esto consultaría la base de datos de eventos
        # Por ahora, devolver valores basados en probabilidad

        return {
            'has_recent_accident': 1 if np.random.random() < 0.01 else 0,
            'has_storm_alert': 1 if np.random.random() < 0.05 else 0,
            'has_political_event': 1 if np.random.random() < 0.02 else 0,
            'news_event_impact': np.random.uniform(0, 1) if np.random.random() < 0.1 else 0
        }


class MultiModalDataLoader:
    """
    Cargador multimodal que combina datos ATC/ATFM con datos externos.

    Integra:
    - Datos de aeronaves (ATC/ATFM)
    - Datos meteorológicos
    - Datos de eventos noticiosos
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Loaders individuales
        self.atc_loader = ATCAircraftDataLoader(config)
        self.weather_loader = WeatherDataLoader(config)
        self.news_loader = NewsDataLoader(config)

    def load_multimodal_data(self, data_type: str = 'daily_atc',
                            use_one_hot: bool = True,
                            include_weather: bool = True,
                            include_news: bool = True) -> pd.DataFrame:
        """
        Carga datos multimodales combinados.

        Args:
            data_type: Tipo de datos ATC ('daily_atc', 'hourly_atfm', 'monthly_route')
            include_weather: Incluir datos meteorológicos
            include_news: Incluir datos de noticias

        Returns:
            DataFrame con datos combinados
        """
        self.logger.info(f"Cargando datos multimodales: {data_type}")

        # Cargar datos ATC/ATFM base
        base_df = self.atc_loader.get_training_data(data_type)

        if use_one_hot and data_type == 'daily_atc':
            df_acids = self.atc_loader.load_daily_acids_data(use_one_hot=True)
            base_df = base_df.merge(df_acids, left_index=True, right_index=True, how='left')

        # Fusionar datos meteorológicos
        if include_weather:
            weather_df = self.weather_loader.load_weather_data(
                start_date=base_df.index.min().strftime('%Y-%m-%d'),
                end_date=base_df.index.max().strftime('%Y-%m-%d')
            )

            # Unir datos (left join para mantener todas las fechas ATC)
            base_df = base_df.merge(
                weather_df,
                left_index=True,
                right_index=True,
                how='left'
            )

            # Rellenar valores faltantes con interpolación
            numeric_cols = weather_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in base_df.columns:
                    base_df[col] = base_df[col].interpolate(method='linear')
                    base_df[col] = base_df[col].fillna(base_df[col].mean())

            self.logger.info("Datos meteorológicos integrados")

        # Fusionar datos de noticias
        if include_news:
            news_df = self.news_loader.load_news_events(

                start_date=base_df.index.min().strftime('%Y-%m-%d'),
                end_date=base_df.index.max().strftime('%Y-%m-%d')
            )

            # Unir datos
            base_df = base_df.merge(
                news_df,
                left_index=True,
                right_index=True,
                how='left'
            )

            # Rellenar valores faltantes (eventos son esporádicos)
            event_cols = news_df.select_dtypes(include=[np.number]).columns
            for col in event_cols:
                if col in base_df.columns:
                    base_df[col] = base_df[col].fillna(0)

            self.logger.info("Datos de noticias integrados")

        self.logger.info(f"Datos multimodales preparados: {len(base_df)} registros, {len(base_df.columns)} columnas")

        return base_df

    def get_multimodal_features_for_forecast(self, target_date: pd.Timestamp) -> Dict[str, float]:
        """
        Obtiene features multimodales para forecasting futuro.

        Args:
            target_date: Fecha objetivo para forecast

        Returns:
            Diccionario con todas las features disponibles
        """
        features = {}

        # Features meteorológicos
        weather_features = self.weather_loader.get_weather_features_for_date(target_date)
        features.update(weather_features)

        # Features de noticias
        news_features = self.news_loader.get_news_features_for_date(target_date)
        features.update(news_features)

        return features

    def get_data_info(self) -> Dict:
        """
        Obtiene información sobre las fuentes de datos disponibles.
        """
        info = {
            'atc_data': self.atc_loader.get_data_info(),
            'weather_data': {
                'available': True,
                'variables': self.weather_loader.weather_config['variables'],
                'stations': self.weather_loader.weather_config['stations']
            },
            'news_data': {
                'available': True,
                'sources': self.news_loader.news_config['sources'],
                'event_types': self.news_loader.news_config['event_types']
            }
        }

        return info