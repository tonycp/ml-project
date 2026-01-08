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

    def load_weather_data(self, start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carga datos meteorológicos históricos.

        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)

        Returns:
            DataFrame con datos meteorológicos diarios
        """
        self.logger.info("Cargando datos meteorológicos...")

        # Por ahora, crear datos sintéticos representativos del clima cubano
        # En producción, esto se conectaría a APIs como OpenWeatherMap, NOAA, etc.

        if start_date and end_date:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            # Usar rango por defecto
            date_range = pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')

        # Generar datos sintéticos representativos del clima cubano
        np.random.seed(42)  # Para reproducibilidad

        weather_data = []

        for date in date_range:
            # Temperatura: 20-32°C (más cálido en verano)
            base_temp = 26 + 3 * np.sin(2 * np.pi * date.dayofyear / 365)
            temperature = base_temp + np.random.normal(0, 2)

            # Humedad: 60-90%
            humidity = 75 + 10 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 5)

            # Velocidad del viento: 5-25 km/h
            wind_speed = 15 + 5 * np.random.normal(0, 1)

            # Dirección del viento (grados)
            wind_direction = np.random.uniform(0, 360)

            # Precipitación: 0-50mm/día (más en temporada de lluvias)
            rainfall_season = 1 if 5 <= date.month <= 10 else 0.3  # Mayo-Octubre: temporada de lluvias
            precipitation = np.random.exponential(2) * rainfall_season if np.random.random() < 0.3 else 0

            # Visibilidad: 5-15 km (reducida con lluvia)
            base_visibility = 12
            visibility = base_visibility * (0.5 if precipitation > 10 else 1) + np.random.normal(0, 1)

            # Cobertura de nubes: 0-100%
            cloud_cover = 40 + 30 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 10)

            # Presión atmosférica: 1010-1020 hPa
            pressure = 1015 + np.random.normal(0, 3)

            # Condiciones extremas (basado en datos históricos cubanos)
            extreme_weather = self._generate_extreme_weather(date)

            weather_data.append({
                'date': date,
                'temperature': max(15, min(35, temperature)),  # Límites realistas
                'humidity': max(30, min(95, humidity)),
                'wind_speed': max(0, wind_speed),
                'wind_direction': wind_direction,
                'precipitation': max(0, precipitation),
                'visibility': max(1, visibility),
                'cloud_cover': max(0, min(100, cloud_cover)),
                'pressure': pressure,
                **extreme_weather
            })

        df = pd.DataFrame(weather_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        self.logger.info(f"Datos meteorológicos cargados: {len(df)} registros")

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
        Obtiene features meteorológicos para una fecha específica.

        Args:
            target_date: Fecha objetivo

        Returns:
            Diccionario con features meteorológicos
        """
        # En producción, esto consultaría la base de datos/API
        # Por ahora, devolver valores representativos

        # Temperatura promedio por mes en La Habana
        monthly_temps = {
            1: 22, 2: 23, 3: 25, 4: 27, 5: 28, 6: 29, 7: 30, 8: 30, 9: 29, 10: 27, 11: 25, 12: 23
        }

        base_temp = monthly_temps.get(target_date.month, 25)

        return {
            'temperature': base_temp + np.random.normal(0, 2),
            'humidity': 70 + np.random.normal(0, 10),
            'wind_speed': 15 + np.random.normal(0, 5),
            'precipitation': np.random.exponential(1) if np.random.random() < 0.2 else 0,
            'visibility': 10 + np.random.normal(0, 2),
            'is_bad_weather': 1 if np.random.random() < 0.1 else 0  # 10% de días con mal tiempo
        }


class NewsDataLoader:
    """
    Cargador de datos de noticias usando el sistema Event_extractor.

    Los eventos noticiosos pueden afectar el tráfico aéreo:
    - Accidentes/Incidentes: Cierran espacios aéreos temporalmente
    - Eventos masivos: Aumentan demanda de vuelos
    - Crisis políticas: Afectan operaciones internacionales
    - Alertas de seguridad: Modifican rutas y procedimientos
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuración de datos de noticias
        self.news_config = config.external_data.get('news', {
            'data_dir': Path('data/news'),
            'sources': ['granma', 'juventud_rebelde', 'cuba_debate'],
            'event_types': ['ACCIDENTE', 'METEOROLOGICO', 'POLITICO', 'SOCIAL', 'INCIDENTE']
        })

    def load_news_events(self, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Carga eventos de noticias que pueden afectar el tráfico aéreo.

        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            DataFrame con eventos noticiosos por fecha
        """
        self.logger.info("Cargando datos de eventos noticiosos...")

        try:
            # Importar el sistema Event_extractor
            import sys
            sys.path.append(str(Path(__file__).parent.parent))

            from Event_extractor import EventExtractionPipeline, NewsContent, NewsMetadata

            # Crear pipeline de extracción
            pipeline = EventExtractionPipeline()

        except ImportError:
            self.logger.warning("Event_extractor no disponible, usando datos sintéticos")
            return self._generate_synthetic_news_events(start_date, end_date)

        # En producción, aquí cargaríamos noticias reales desde archivos/APIs
        # Por ahora, generar eventos sintéticos representativos

        return self._generate_synthetic_news_events(start_date, end_date)

    def _generate_synthetic_news_events(self, start_date: Optional[str] = None,
                                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Genera eventos noticiosos sintéticos representativos de Cuba.
        """
        if start_date and end_date:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            date_range = pd.date_range(start='2022-01-01', end='2025-12-31', freq='D')

        events_data = []

        for date in date_range:
            # Eventos diarios (baja probabilidad)
            daily_events = {
                'accident_count': 0,
                'storm_alert_count': 0,
                'political_event_count': 0,
                'social_event_count': 0,
                'incident_count': 0,
                'international_event_count': 0,
                'has_major_event': 0,
                'event_impact_score': 0.0
            }

            # Accidentes/Incidentes (muy raros)
            if np.random.random() < 0.005:  # 0.5% de probabilidad
                daily_events['accident_count'] = 1
                daily_events['incident_count'] = 1
                daily_events['has_major_event'] = 1
                daily_events['event_impact_score'] = 0.8

            # Alertas meteorológicas (más comunes en temporada de lluvias)
            rain_season = 1 if 5 <= date.month <= 10 else 0
            storm_prob = 0.02 * rain_season + 0.005  # 2% en temporada, 0.5% fuera
            if np.random.random() < storm_prob:
                daily_events['storm_alert_count'] = 1
                daily_events['has_major_event'] = 1
                daily_events['event_impact_score'] = 0.6

            # Eventos políticos (elecciones, visitas importantes)
            if np.random.random() < 0.01:  # 1% de probabilidad
                daily_events['political_event_count'] = 1
                daily_events['international_event_count'] = 1
                daily_events['event_impact_score'] = 0.4

            # Eventos sociales/masivos
            if np.random.random() < 0.015:  # 1.5% de probabilidad
                daily_events['social_event_count'] = 1
                daily_events['event_impact_score'] = 0.3

            events_data.append({
                'date': date,
                **daily_events
            })

        df = pd.DataFrame(events_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        self.logger.info(f"Eventos noticiosos generados: {len(df)} registros")

        return df

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