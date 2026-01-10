"""
Cargador de datos para archivos ATC/ATFM.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import logging
import ast
from datetime import datetime, timedelta

import re
from .config import ModelConfig


class ATCAircraftDataLoader:
    """
    Cargador de datos de aeronaves desde archivos ATC/ATFM.

    Maneja la carga y procesamiento inicial de diferentes tipos de archivos:
    - Resúmenes diarios ATC (atc_dayatcopsummary)
    - ACIDs diarios ATC (atc_daylyacids)
    - Vuelos horarios ATFM (atfm_hourlyaoigroupflights)
    - Vuelos mensuales por ruta (atfm_monthrouteflights)
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Crear directorios si no existen
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def load_daily_atc_data(self) -> pd.DataFrame:
        """
        Carga datos diarios de operaciones ATC.

        Returns:
            DataFrame con datos diarios de total de aeronaves
        """
        file_path = self.config.get_data_path(self.config.atc_daily_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        self.logger.info(f"Cargando datos diarios ATC desde: {file_path}")

        # Leer CSV
        df = pd.read_csv(file_path)

        # Convertir columna de fecha y eliminar timezone
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column]).dt.tz_localize(None)

        # Establecer índice de fecha
        df = df.set_index(self.config.date_column).sort_index()

        # Filtrar columnas relevantes
        relevant_cols = [
            'arrivals', 'departures', 'overflights', 'nationals',
            'unknown', 'total', 'fpp'
        ]
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols]

        # Verificar que la columna target existe
        if self.config.target_column not in df.columns:
            # Calcular total si no existe
            if all(col in df.columns for col in ['arrivals', 'departures', 'overflights']):
                df[self.config.target_column] = (
                    df['arrivals'] + df['departures'] + df['overflights']
                )
                self.logger.info("Columna 'total' calculada como suma de arrivals + departures + overflights")
            else:
                raise ValueError(f"Columna target '{self.config.target_column}' no encontrada y no se puede calcular")

        self.logger.info(f"Datos diarios cargados: {len(df)} registros, columnas: {list(df.columns)}")

        return df

    def load_hourly_atc_data(self) -> pd.DataFrame:
        """
        Carga datos horarios de operaciones ATC.

        Returns:
            DataFrame con datos horarios de total de aeronaves
        """
        file_path = self.config.get_data_path(self.config.atc_hourly_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        self.logger.info(f"Cargando datos horarios ATC desde: {file_path}")

        # Leer CSV
        df = pd.read_csv(file_path)

        # Convertir columna de fecha y eliminar timezone
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column]).dt.tz_localize(None)

        # Establecer índice de fecha
        df = df.set_index(self.config.date_column).sort_index()

        # Filtrar columnas relevantes
        relevant_cols = [
            'arrivals', 'departures', 'overflights', 'nationals',
            'unknown', 'total', 'fpp'
        ]
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols]

        # Verificar que la columna target existe
        if self.config.target_column not in df.columns:
            # Calcular total si no existe
            if all(col in df.columns for col in ['arrivals', 'departures', 'overflights']):
                df[self.config.target_column] = (
                    df['arrivals'] + df['departures'] + df['overflights']
                )
                self.logger.info("Columna 'total' calculada como suma de arrivals + departures + overflights")
            else:
                raise ValueError(f"Columna target '{self.config.target_column}' no encontrada y no se puede calcular")

        self.logger.info(f"Datos horarios cargados: {len(df)} registros, columnas: {list(df.columns)}")

        return df

    def load_daily_acids_data(self, use_one_hot: bool = False) -> pd.DataFrame:
        """
        Carga y transforma los datos de `daily_acids` en features numéricas.

        Args:
            use_one_hot: Si True, crea one-hot encoding de aerolíneas

        Returns:
            pd.DataFrame: Un DataFrame con la fecha como índice y las nuevas
                          features de aerolíneas.
        """
        file_path = self.config.get_data_path(self.config.atc_daily_acids_file)
        if not file_path.exists():
            self.logger.warning(f"Archivo de ACIDs no encontrado en: {file_path}. Saltando carga.")
            return pd.DataFrame()

        self.logger.info(f"Cargando datos diarios ACIDs desde: {file_path}")
        df_raw = pd.read_csv(file_path)

        if df_raw.empty:
            self.logger.warning("El archivo de ACIDs está vacío.")
            return pd.DataFrame()

        # Transformar los datos crudos en features
        if use_one_hot:
            df_features = self._create_airline_one_hot_features(df_raw)
        else:
            df_features = self._create_airline_features(df_raw)

        self.logger.info(f"Features de aerolíneas creadas: {len(df_features)} registros, {len(df_features.columns)} columnas.")
        return df_features

    def _extract_airline_code(self, flight_id: str) -> str:
        """Extrae el código de aerolínea de un ID de vuelo."""
        if not isinstance(flight_id, str):
            return 'UNKNOWN'
        # Vuelos comerciales (e.g., AAL, BAW, UAL)
        match = re.match(r'^([A-Z]{3})', flight_id)
        if match:
            return match.group(1)
        # Vuelos privados/generales (e.g., N123AB)
        if re.match(r'^[NCT]|^[A-Z]{1,2}-', flight_id):
            return 'PRIVATE'
        return 'UNKNOWN'

    def _safe_parse_list(self, list_str: str) -> list:
        """Parsea de forma segura un string que representa una lista."""
        if pd.isna(list_str) or not isinstance(list_str, str) or list_str == '':
            return []
        try:
            # Usar ast.literal_eval como en tu código
            return ast.literal_eval(list_str)
        except Exception:
            # Fallback: limpiar y parsear manualmente
            cleaned = list_str.strip('[]').replace('"', '').replace("'", "")
            if cleaned:
                return [x.strip() for x in cleaned.split(',') if x.strip()]
            return []

    def _create_airline_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Crea features numéricas a partir de la composición de aerolíneas."""
        major_us = {'AAL', 'UAL', 'DAL', 'SWA', 'NKS'}
        major_eu = {'IBE', 'AFR', 'BAW', 'KLM', 'DLH', 'CFG'}
        major_cargo = {'UPS', 'FDX', 'GTI', 'ABX'}

        feature_rows = []
        for _, row in df_raw.iterrows():
            acids_list = self._safe_parse_list(row.get('acids_list', '[]'))
            if not acids_list:
                continue

            airlines = [self._extract_airline_code(fid) for fid in acids_list]
            total_flights = len(airlines)
            
            airline_counts = pd.Series(airlines).value_counts()
            unique_airlines = set(airline_counts.index)

            # Calcular conteos y porcentajes
            us_flights = sum(airline_counts.get(c, 0) for c in major_us)
            eu_flights = sum(airline_counts.get(c, 0) for c in major_eu)
            cargo_flights = sum(airline_counts.get(c, 0) for c in major_cargo)
            private_flights = airline_counts.get('PRIVATE', 0)

            feature_rows.append({
                'time': row[self.config.date_column],
                'num_unique_airlines': len(unique_airlines),
                'pct_us_major': us_flights / total_flights,
                'pct_eu_major': eu_flights / total_flights,
                'pct_cargo_major': cargo_flights / total_flights,
                'pct_private': private_flights / total_flights,
                'top_airline_pct': airline_counts.iloc[0] / total_flights if not airline_counts.empty else 0
            })

        if not feature_rows:
            return pd.DataFrame()

        df_features = pd.DataFrame(feature_rows)
        df_features[self.config.date_column] = pd.to_datetime(df_features[self.config.date_column])
        df_features = df_features.set_index(self.config.date_column).sort_index()
        
        return df_features
    
    def _create_airline_one_hot_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Crea conteo de vuelos por aerolínea a partir de la columna acids_list.
        
        Cuenta cuántos vuelos tiene cada aerolínea cada día.
        """
        # Usar tu lógica para obtener el conjunto de todas las aerolíneas
        all_airlines = set()
        for acids_string in df_raw['acids_list']:
            acids_list = self._safe_parse_list(acids_string)
            airlines_list = [self._extract_airline_code(acid) for acid in acids_list]
            airlines_list = [air for air in airlines_list if len(air) == 3] 
            all_airlines.update(airlines_list)
        
        # Convertir a lista ordenada para consistencia
        all_airlines = sorted(list(all_airlines))
        
        self.logger.info(f"Creando conteo de vuelos para {len(all_airlines)} aerolíneas únicas")
        
        # Crear DataFrame con conteo de vuelos por aerolínea
        feature_rows = []
        for _, row in df_raw.iterrows():
            acids_list = self._safe_parse_list(row.get('acids_list', '[]'))
            airlines_list = [self._extract_airline_code(acid) for acid in acids_list]
            airlines_list = [air for air in airlines_list if len(air) == 3]
            
            # Contar vuelos por aerolínea
            airline_counts = pd.Series(airlines_list).value_counts()
            
            # Crear diccionario con conteo de vuelos
            row_features = {
                'time': row[self.config.date_column],
                **{f'airline_{airline}': airline_counts.get(airline, 0) 
                  for airline in all_airlines}
            }
            feature_rows.append(row_features)
        
        if not feature_rows:
            return pd.DataFrame()
        
        df_features = pd.DataFrame(feature_rows)
        df_features[self.config.date_column] = pd.to_datetime(df_features[self.config.date_column])
        df_features = df_features.set_index(self.config.date_column).sort_index()
        
        return df_features

    def load_hourly_acids_data(self, use_one_hot: bool = False) -> pd.DataFrame:
        """
        Carga y transforma los datos de `hourly_acids` en features numéricas.

        Args:
            use_one_hot: Si True, crea one-hot encoding de aerolíneas

        Returns:
            pd.DataFrame: Un DataFrame con la fecha como índice y las nuevas
                          features de aerolíneas.
        """
        file_path = self.config.get_data_path(self.config.atc_hourly_acids_file)
        if not file_path.exists():
            self.logger.warning(f"Archivo de ACIDs horarios no encontrado en: {file_path}. Saltando carga.")
            return pd.DataFrame()

        self.logger.info(f"Cargando datos horarios ACIDs desde: {file_path}")
        df_raw = pd.read_csv(file_path)

        if df_raw.empty:
            self.logger.warning("El archivo de ACIDs horarios está vacío.")
            return pd.DataFrame()

        # Transformar los datos crudos en features
        if use_one_hot:
            df_features = self._create_hourly_airline_one_hot_features(df_raw)
        else:
            df_features = self._create_hourly_airline_features(df_raw)

        self.logger.info(f"Features de aerolíneas horarias creadas: {len(df_features)} registros, {len(df_features.columns)} columnas.")
        return df_features

    def _create_hourly_airline_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Crea features numéricas a partir de la composición de aerolíneas horarias."""
        major_us = {'AAL', 'UAL', 'DAL', 'SWA', 'NKS'}
        major_eu = {'IBE', 'AFR', 'BAW', 'KLM', 'DLH', 'CFG'}
        major_cargo = {'UPS', 'FDX', 'GTI', 'ABX'}

        feature_rows = []
        for _, row in df_raw.iterrows():
            flight_ids_list = self._safe_parse_list(row.get('flight_ids', '[]'))
            if not flight_ids_list:
                continue

            airlines = [self._extract_airline_code(fid) for fid in flight_ids_list]
            total_flights = len(airlines)
            
            airline_counts = pd.Series(airlines).value_counts()
            unique_airlines = set(airline_counts.index)

            # Calcular conteos y porcentajes
            us_flights = sum(airline_counts.get(c, 0) for c in major_us)
            eu_flights = sum(airline_counts.get(c, 0) for c in major_eu)
            cargo_flights = sum(airline_counts.get(c, 0) for c in major_cargo)
            private_flights = airline_counts.get('PRIVATE', 0)

            # Crear timestamp combinando date y hour
            date_str = row['date']
            hour = int(row['hour'])
            timestamp = pd.to_datetime(f"{date_str} {hour:02d}:00:00")

            feature_rows.append({
                'time': timestamp,
                'num_unique_airlines': len(unique_airlines),
                'pct_us_major': us_flights / total_flights,
                'pct_eu_major': eu_flights / total_flights,
                'pct_cargo_major': cargo_flights / total_flights,
                'pct_private': private_flights / total_flights,
                'top_airline_pct': airline_counts.iloc[0] / total_flights if not airline_counts.empty else 0
            })

        if not feature_rows:
            return pd.DataFrame()

        df_features = pd.DataFrame(feature_rows)
        df_features[self.config.date_column] = pd.to_datetime(df_features[self.config.date_column])
        df_features = df_features.set_index(self.config.date_column).sort_index()
        
        return df_features
    
    def _create_hourly_airline_one_hot_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Crea conteo de vuelos por aerolínea a partir de la columna flight_ids.
        
        Cuenta cuántos vuelos tiene cada aerolínea cada hora.
        """
        # Obtener el conjunto de todas las aerolíneas
        all_airlines = set()
        for flight_ids_string in df_raw['flight_ids']:
            flight_ids_list = self._safe_parse_list(flight_ids_string)
            airlines_list = [self._extract_airline_code(fid) for fid in flight_ids_list]
            airlines_list = [air for air in airlines_list if len(air) == 3] 
            all_airlines.update(airlines_list)
        
        # Convertir a lista ordenada para consistencia
        all_airlines = sorted(list(all_airlines))
        
        self.logger.info(f"Creando conteo de vuelos horarios para {len(all_airlines)} aerolíneas únicas")
        
        # Crear DataFrame con conteo de vuelos por aerolínea
        feature_rows = []
        for _, row in df_raw.iterrows():
            flight_ids_list = self._safe_parse_list(row.get('flight_ids', '[]'))
            airlines_list = [self._extract_airline_code(fid) for fid in flight_ids_list]
            airlines_list = [air for air in airlines_list if len(air) == 3]
            
            # Contar vuelos por aerolínea
            airline_counts = pd.Series(airlines_list).value_counts()
            
            # Crear timestamp combinando date y hour
            date_str = row['date']
            hour = int(row['hour'])
            timestamp = pd.to_datetime(f"{date_str} {hour:02d}:00:00")
            
            # Crear diccionario con conteo de vuelos
            row_features = {
                'time': timestamp,
                **{f'airline_{airline}': airline_counts.get(airline, 0) 
                  for airline in all_airlines}
            }
            feature_rows.append(row_features)
        
        if not feature_rows:
            return pd.DataFrame()
        
        df_features = pd.DataFrame(feature_rows)
        df_features[self.config.date_column] = pd.to_datetime(df_features[self.config.date_column])
        df_features = df_features.set_index(self.config.date_column).sort_index()
        
        return df_features

    def load_hourly_atfm_data(self) -> pd.DataFrame:
        """
        Carga datos horarios ATFM agrupados por área.

        Returns:
            DataFrame con datos horarios de aeronaves por área
        """
        file_path = self.config.get_data_path(self.config.atfm_hourly_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        self.logger.info(f"Cargando datos horarios ATFM desde: {file_path}")

        # Leer CSV
        df = pd.read_csv(file_path)

        # Convertir columna de fecha/hora
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column])

        # Parsear columna 'gufis' que contiene lista de IDs de vuelos
        df['aircraft_count'] = df['gufis'].apply(self._parse_gufis_count)

        # Establecer índice de fecha
        df = df.set_index(self.config.date_column).sort_index()

        # Agrupar por hora y área para obtener totales
        hourly_data = df.groupby([pd.Grouper(freq='H'), 'aoi'])['aircraft_count'].sum().reset_index()

        # Pivot para tener áreas como columnas
        hourly_data = hourly_data.pivot(
            index=self.config.date_column,
            columns='aoi',
            values='aircraft_count'
        ).fillna(0)

        # Calcular total general
        hourly_data[self.config.target_column] = hourly_data.sum(axis=1)

        self.logger.info(f"Datos horarios cargados: {len(hourly_data)} registros, áreas: {list(hourly_data.columns)}")

        return hourly_data

    def load_monthly_route_data(self) -> pd.DataFrame:
        """
        Carga datos mensuales de vuelos por ruta.

        Returns:
            DataFrame con datos mensuales por ruta
        """
        file_path = self.config.get_data_path(self.config.atfm_monthly_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        self.logger.info(f"Cargando datos mensuales por ruta desde: {file_path}")

        # Leer CSV
        df = pd.read_csv(file_path)

        # Convertir columna de fecha y eliminar timezone
        df[self.config.date_column] = pd.to_datetime(df[self.config.date_column]).dt.tz_localize(None)

        # Pivot para tener rutas como columnas
        monthly_data = df.pivot(
            index=self.config.date_column,
            columns='route_name',
            values=self.config.target_column
        ).fillna(0)

        # Calcular total general
        monthly_data[self.config.target_column] = monthly_data.sum(axis=1)

        self.logger.info(f"Datos mensuales cargados: {len(monthly_data)} registros, rutas: {list(monthly_data.columns)}")

        return monthly_data

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Carga todos los datasets disponibles.

        Returns:
            Diccionario con todos los datasets cargados
        """
        data = {}

        try:
            data['daily_atc'] = self.load_daily_atc_data()
        except Exception as e:
            self.logger.warning(f"Error cargando datos diarios ATC: {e}")

        try:
            data['hourly_atc'] = self.load_hourly_atc_data()
        except Exception as e:
            self.logger.warning(f"Error cargando datos horarios ATC: {e}")

        try:
            data['daily_acids'] = self.load_daily_acids_data()
        except Exception as e:
            self.logger.warning(f"Error cargando datos diarios ACIDs: {e}")

        try:
            data['hourly_acids'] = self.load_hourly_acids_data()
        except Exception as e:
            self.logger.warning(f"Error cargando datos horarios ACIDs: {e}")

        try:
            data['hourly_atfm'] = self.load_hourly_atfm_data()
        except Exception as e:
            self.logger.warning(f"Error cargando datos horarios ATFM: {e}")

        try:
            data['monthly_route'] = self.load_monthly_route_data()
        except Exception as e:
            self.logger.warning(f"Error cargando datos mensuales por ruta: {e}")

        return data

    def get_training_data(self,
                         data_type: str = 'daily_atc',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Obtiene datos de entrenamiento para un tipo específico.

        Args:
            data_type: Tipo de datos ('daily_atc', 'hourly_atc', 'daily_acids', 'hourly_acids', 'hourly_atfm', 'monthly_route')
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)

        Returns:
            DataFrame filtrado para entrenamiento
        """
        # Cargar datos según tipo
        if data_type == 'daily_atc':
            df = self.load_daily_atc_data()
        elif data_type == 'hourly_atc':
            df = self.load_hourly_atc_data()
        elif data_type == 'daily_acids':
            df = self.load_daily_acids_data()
        elif data_type == 'hourly_acids':
            df = self.load_hourly_acids_data()
        elif data_type == 'hourly_atfm':
            df = self.load_hourly_atfm_data()
        elif data_type == 'monthly_route':
            df = self.load_monthly_route_data()
        else:
            raise ValueError(f"Tipo de datos desconocido: {data_type}")

        # Aplicar filtros de fecha
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df.index >= start]

        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df.index <= end]

        # Verificar que hay datos suficientes
        if len(df) < 30:  # Mínimo 30 días de datos
            self.logger.warning(f"Pocos datos para entrenamiento: {len(df)} registros")

        self.logger.info(f"Datos de entrenamiento preparados: {len(df)} registros del {df.index.min()} al {df.index.max()}")

        return df

    def split_train_val_test(self,
                           df: pd.DataFrame,
                           train_split: float = None,
                           val_split: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.

        Args:
            df: DataFrame con datos temporales
            train_split: Proporción para entrenamiento
            val_split: Proporción para validación

        Returns:
            Tupla (train_df, val_df, test_df)
        """
        if train_split is None:
            train_split = self.config.train_split
        if val_split is None:
            val_split = self.config.val_split

        n_total = len(df)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

        self.logger.info(f"Split realizado: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return train_df, val_df, test_df

    def _parse_gufis_count(self, gufis_str: str) -> int:
        """
        Parsea la columna 'gufis' que contiene una lista JSON de IDs de vuelos.

        Args:
            gufis_str: String JSON con lista de GUFIs

        Returns:
            Número de aeronaves (longitud de la lista)
        """
        if pd.isna(gufis_str) or gufis_str == '':
            return 0

        try:
            # Parsear JSON string
            gufis_list = json.loads(gufis_str)
            return len(gufis_list)
        except (json.JSONDecodeError, TypeError):
            # Si no es JSON válido, intentar contar elementos separados por comas
            if isinstance(gufis_str, str):
                # Remover corchetes y comillas
                cleaned = gufis_str.strip('[]').replace('"', '').replace("'", '')
                if cleaned:
                    return len([x.strip() for x in cleaned.split(',') if x.strip()])
            return 0

    def get_data_info(self) -> Dict:
        """
        Obtiene información general sobre los datasets disponibles.

        Returns:
            Diccionario con información de los datasets
        """
        info = {}

        try:
            daily_df = self.load_daily_atc_data()
            info['daily_atc'] = {
                'records': len(daily_df),
                'date_range': f"{daily_df.index.min()} to {daily_df.index.max()}",
                'columns': list(daily_df.columns),
                'target_stats': {
                    'mean': daily_df[self.config.target_column].mean(),
                    'std': daily_df[self.config.target_column].std(),
                    'min': daily_df[self.config.target_column].min(),
                    'max': daily_df[self.config.target_column].max()
                }
            }
        except Exception as e:
            info['daily_atc'] = {'error': str(e)}

        try:
            hourly_df = self.load_hourly_atc_data()
            info['hourly_atc'] = {
                'records': len(hourly_df),
                'date_range': f"{hourly_df.index.min()} to {hourly_df.index.max()}",
                'columns': list(hourly_df.columns),
                'target_stats': {
                    'mean': hourly_df[self.config.target_column].mean(),
                    'std': hourly_df[self.config.target_column].std(),
                    'min': hourly_df[self.config.target_column].min(),
                    'max': hourly_df[self.config.target_column].max()
                }
            }
        except Exception as e:
            info['hourly_atc'] = {'error': str(e)}

        try:
            daily_acids_df = self.load_daily_acids_data()
            info['daily_acids'] = {
                'records': len(daily_acids_df),
                'date_range': f"{daily_acids_df.index.min()} to {daily_acids_df.index.max()}",
                'columns': list(daily_acids_df.columns),
                'target_stats': {
                    'mean': daily_acids_df[self.config.target_column].mean(),
                    'std': daily_acids_df[self.config.target_column].std(),
                    'min': daily_acids_df[self.config.target_column].min(),
                    'max': daily_acids_df[self.config.target_column].max()
                }
            }
        except Exception as e:
            info['daily_acids'] = {'error': str(e)}

        try:
            hourly_acids_df = self.load_hourly_acids_data()
            info['hourly_acids'] = {
                'records': len(hourly_acids_df),
                'date_range': f"{hourly_acids_df.index.min()} to {hourly_acids_df.index.max()}",
                'columns': list(hourly_acids_df.columns),
                'target_stats': {
                    'mean': hourly_acids_df[self.config.target_column].mean(),
                    'std': hourly_acids_df[self.config.target_column].std(),
                    'min': hourly_acids_df[self.config.target_column].min(),
                    'max': hourly_acids_df[self.config.target_column].max()
                }
            }
        except Exception as e:
            info['hourly_acids'] = {'error': str(e)}

        try:
            hourly_df = self.load_hourly_atfm_data()
            info['hourly_atfm'] = {
                'records': len(hourly_df),
                'date_range': f"{hourly_df.index.min()} to {hourly_df.index.max()}",
                'areas': [col for col in hourly_df.columns if col != self.config.target_column],
                'target_stats': {
                    'mean': hourly_df[self.config.target_column].mean(),
                    'std': hourly_df[self.config.target_column].std(),
                    'min': hourly_df[self.config.target_column].min(),
                    'max': hourly_df[self.config.target_column].max()
                }
            }
        except Exception as e:
            info['hourly_atfm'] = {'error': str(e)}

        try:
            monthly_df = self.load_monthly_route_data()
            info['monthly_route'] = {
                'records': len(monthly_df),
                'date_range': f"{monthly_df.index.min()} to {monthly_df.index.max()}",
                'routes': [col for col in monthly_df.columns if col != self.config.target_column],
                'target_stats': {
                    'mean': monthly_df[self.config.target_column].mean(),
                    'std': monthly_df[self.config.target_column].std(),
                    'min': monthly_df[self.config.target_column].min(),
                    'max': monthly_df[self.config.target_column].max()
                }
            }
        except Exception as e:
            info['monthly_route'] = {'error': str(e)}

        return info