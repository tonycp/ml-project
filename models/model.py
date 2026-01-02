"""
Modelos de forecasting para predicción de aeronaves.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import pickle
from pathlib import Path
from abc import ABC, abstractmethod

from .config import ModelConfig


class BaseForecaster(ABC):
    """Clase base abstracta para todos los forecasters."""

    def __init__(self, config: ModelConfig, name: str):
        self.config = config
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.is_trained = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseForecaster':
        """Entrena el modelo."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones."""
        pass

    @abstractmethod
    def save(self, filepath: Path) -> None:
        """Guarda el modelo."""
        pass

    @abstractmethod
    def load(self, filepath: Path) -> 'BaseForecaster':
        """Carga el modelo."""
        pass

    def get_model_info(self) -> Dict:
        """Retorna información del modelo."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'config': self.config.__dict__
        }


class ARIMAModel(BaseForecaster):
    """Modelo ARIMA/SARIMA para forecasting de series temporales."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, 'arima')
        self.order = config.models['arima']['order']
        self.seasonal_order = config.models['arima']['seasonal_order']

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ARIMAModel':
        """Entrena modelo ARIMA."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            self.logger.info(f"Entrenando ARIMA con order={self.order}, seasonal_order={self.seasonal_order}")

            # ARIMA solo usa la serie temporal target, ignora otras features
            self.model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            self.results = self.model.fit(disp=False)
            self.is_trained = True

            self.logger.info("ARIMA entrenado exitosamente")

        except ImportError:
            raise ImportError("statsmodels no está instalado. Instale con: pip install statsmodels")
        except Exception as e:
            self.logger.error(f"Error entrenando ARIMA: {e}")
            raise

        return self

    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones con ARIMA."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        try:
            # Predicciones one-step-ahead para datos históricos
            if len(X) > 0:
                predictions = self.results.predict(
                    start=X.index[0],
                    end=X.index[-1],
                    dynamic=False
                )
                return predictions.values
            else:
                # Forecast futuro
                forecast = self.results.forecast(steps=forecast_horizon)
                return forecast.values

        except Exception as e:
            self.logger.error(f"Error en predicción ARIMA: {e}")
            raise

    def save(self, filepath: Path) -> None:
        """Guarda modelo ARIMA."""
        model_data = {
            'model': self.model,
            'results': self.results,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'config': self.config.__dict__
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Modelo ARIMA guardado en {filepath}")

    def load(self, filepath: Path) -> 'ARIMAModel':
        """Carga modelo ARIMA."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.results = model_data['results']
        self.order = model_data['order']
        self.seasonal_order = model_data['seasonal_order']
        self.is_trained = True

        self.logger.info(f"Modelo ARIMA cargado desde {filepath}")

        return self


class ProphetModel(BaseForecaster):
    """Modelo Prophet de Facebook para forecasting."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, 'prophet')
        self.prophet_config = config.models['prophet']

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ProphetModel':
        """Entrena modelo Prophet."""
        try:
            from prophet import Prophet

            self.logger.info("Entrenando Prophet")

            # Preparar datos para Prophet (requiere columnas 'ds' y 'y')
            df_prophet = pd.DataFrame({
                'ds': y.index,
                'y': y.values
            })

            self.model = Prophet(**self.prophet_config)
            self.model.fit(df_prophet)

            self.is_trained = True
            self.logger.info("Prophet entrenado exitosamente")

        except ImportError:
            raise ImportError("prophet no está instalado. Instale con: pip install prophet")
        except Exception as e:
            self.logger.error(f"Error entrenando Prophet: {e}")
            raise

        return self

    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones con Prophet."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        try:
            # Crear dataframe futuro
            if len(X) > 0:
                # Predicciones en datos históricos
                future_dates = pd.DataFrame({'ds': X.index})
            else:
                # Forecast futuro
                last_date = self.model.history['ds'].max()
                future_dates = pd.DataFrame({
                    'ds': pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_horizon,
                        freq='D'
                    )
                })

            forecast = self.model.predict(future_dates)
            return forecast['yhat'].values

        except Exception as e:
            self.logger.error(f"Error en predicción Prophet: {e}")
            raise

    def save(self, filepath: Path) -> None:
        """Guarda modelo Prophet."""
        model_data = {
            'model': self.model,
            'config': self.prophet_config,
            'model_config': self.config.__dict__
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Modelo Prophet guardado en {filepath}")

    def load(self, filepath: Path) -> 'ProphetModel':
        """Carga modelo Prophet."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.prophet_config = model_data['config']
        self.is_trained = True

        self.logger.info(f"Modelo Prophet cargado desde {filepath}")

        return self


class RandomForestModel(BaseForecaster):
    """Modelo Random Forest para forecasting de series temporales."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, 'random_forest')
        self.rf_config = config.models.get('random_forest', {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        })

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """Entrena modelo Random Forest."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            self.logger.info(f"Entrenando Random Forest con {self.rf_config['n_estimators']} árboles")

            self.model = RandomForestRegressor(**self.rf_config)
            self.model.fit(X, y)

            self.is_trained = True
            self.logger.info("Random Forest entrenado exitosamente")

            # Calcular importancia de features
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        except ImportError:
            raise ImportError("scikit-learn no está instalado. Instale con: pip install scikit-learn")
        except Exception as e:
            self.logger.error(f"Error entrenando Random Forest: {e}")
            raise

        return self

    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones con Random Forest."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        try:
            predictions = self.model.predict(X)
            return predictions

        except Exception as e:
            self.logger.error(f"Error en predicción Random Forest: {e}")
            raise

    def save(self, filepath: Path) -> None:
        """Guarda modelo Random Forest."""
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'config': self.rf_config,
            'model_config': self.config.__dict__
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Modelo Random Forest guardado en {filepath}")

    def load(self, filepath: Path) -> 'RandomForestModel':
        """Carga modelo Random Forest."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance', {})
        self.rf_config = model_data['config']
        self.is_trained = True

        self.logger.info(f"Modelo Random Forest cargado desde {filepath}")

        return self


class LSTMModel(BaseForecaster):
    """Modelo LSTM para forecasting de series temporales."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, 'lstm')
        self.lstm_config = config.models['lstm']
        self.sequence_length = self.lstm_config['sequence_length']

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMModel':
        """Entrena modelo LSTM."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from sklearn.preprocessing import MinMaxScaler

            self.logger.info(f"Entrenando LSTM con sequence_length={self.sequence_length}")

            # Escalar datos
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()

            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))

            # Crear secuencias
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled.ravel())

            # Construir modelo
            self.model = Sequential([
                LSTM(64, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2]),
                     return_sequences=True),
                Dropout(self.lstm_config['dropout_rate']),
                LSTM(32, activation='relu'),
                Dropout(self.lstm_config['dropout_rate']),
                Dense(1)
            ])

            self.model.compile(
                optimizer=Adam(learning_rate=self.lstm_config['learning_rate']),
                loss='mse'
            )

            # Entrenar
            self.model.fit(
                X_seq, y_seq,
                epochs=self.lstm_config['epochs'],
                batch_size=self.lstm_config['batch_size'],
                verbose=0
            )

            self.is_trained = True
            self.logger.info("LSTM entrenado exitosamente")

        except ImportError:
            raise ImportError("tensorflow no está instalado. Instale con: pip install tensorflow")
        except Exception as e:
            self.logger.error(f"Error entrenando LSTM: {e}")
            raise

        return self

    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones con LSTM."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        try:
            X_scaled = self.scaler_X.transform(X)
            X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

            predictions_scaled = self.model.predict(X_seq, verbose=0)
            predictions = self.scaler_y.inverse_transform(predictions_scaled)

            return predictions.ravel()

        except Exception as e:
            self.logger.error(f"Error en predicción LSTM: {e}")
            raise

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crea secuencias para LSTM."""
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def save(self, filepath: Path) -> None:
        """Guarda modelo LSTM."""
        # Guardar modelo de TensorFlow
        model_path = filepath.with_suffix('.h5')
        self.model.save(model_path)

        # Guardar scalers y config
        model_data = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'sequence_length': self.sequence_length,
            'config': self.lstm_config,
            'model_config': self.config.__dict__
        }

        pickle_path = filepath.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Modelo LSTM guardado en {model_path} y {pickle_path}")

    def load(self, filepath: Path) -> 'LSTMModel':
        """Carga modelo LSTM."""
        try:
            import tensorflow as tf

            # Cargar modelo de TensorFlow
            model_path = filepath.with_suffix('.h5')
            self.model = tf.keras.models.load_model(model_path)

            # Cargar scalers y config
            pickle_path = filepath.with_suffix('.pkl')
            with open(pickle_path, 'rb') as f:
                model_data = pickle.load(f)

            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.sequence_length = model_data['sequence_length']
            self.lstm_config = model_data['config']
            self.is_trained = True

            self.logger.info(f"Modelo LSTM cargado desde {model_path}")

        except Exception as e:
            self.logger.error(f"Error cargando LSTM: {e}")
            raise

        return self


class EnsembleModel(BaseForecaster):
    """Modelo ensemble que combina múltiples forecasters."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, 'ensemble')
        self.models = {}
        self.weights = config.models['ensemble']['weights']

    def add_model(self, model: BaseForecaster, weight: float = 1.0) -> None:
        """Añade un modelo al ensemble."""
        self.models[model.name] = {'model': model, 'weight': weight}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """Entrena todos los modelos del ensemble."""
        self.logger.info("Entrenando ensemble de modelos")

        for name, model_info in self.models.items():
            self.logger.info(f"Entrenando {name}...")
            model_info['model'].fit(X, y)

        self.is_trained = True
        self.logger.info("Ensemble entrenado exitosamente")

        return self

    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones promediadas por el ensemble."""
        if not self.is_trained:
            raise ValueError("Ensemble no entrenado")

        predictions = []
        weights = []

        for name, model_info in self.models.items():
            try:
                pred = model_info['model'].predict(X, forecast_horizon)
                predictions.append(pred)
                weights.append(model_info['weight'])
            except Exception as e:
                self.logger.warning(f"Error prediciendo con {name}: {e}")
                continue

        if not predictions:
            raise ValueError("Ningún modelo pudo hacer predicciones")

        # Promedio ponderado
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalizar

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def save(self, filepath: Path) -> None:
        """Guarda el ensemble."""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'config': self.config.__dict__
        }

        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)

        self.logger.info(f"Ensemble guardado en {filepath}")

    def load(self, filepath: Path) -> 'EnsembleModel':
        """Carga el ensemble."""
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)

        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.is_trained = True

        self.logger.info(f"Ensemble cargado desde {filepath}")

        return self


class AircraftForecaster:
    """
    Clase principal para forecasting de aeronaves.

    Coordina el uso de diferentes modelos y proporciona
    una interfaz unificada para forecasting.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_model = None

    def add_model(self, model: BaseForecaster) -> None:
        """Añade un modelo al forecaster."""
        self.models[model.name] = model
        self.logger.info(f"Modelo {model.name} añadido")

    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Entrena todos los modelos y retorna métricas.

        Returns:
            Diccionario con resultados de entrenamiento por modelo
        """
        results = {}

        for name, model in self.models.items():
            self.logger.info(f"Entrenando modelo: {name}")

            try:
                # Entrenar modelo
                model.fit(X, y)

                # Evaluar en conjunto de validación (usando cross-validation simple)
                metrics = self._evaluate_model(model, X, y)

                results[name] = {
                    'success': True,
                    'metrics': metrics,
                    'model_info': model.get_model_info()
                }

                self.logger.info(f"Modelo {name} entrenado - MAE: {metrics.get('mae', 'N/A'):.2f}")

            except Exception as e:
                self.logger.error(f"Error entrenando {name}: {e}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }

        # Seleccionar mejor modelo
        self._select_best_model(results)

        return results

    def forecast(self, X: pd.DataFrame = None,
                forecast_horizon: int = 1,
                model_name: str = None) -> np.ndarray:
        """
        Realiza forecasting usando el mejor modelo o uno específico.

        Args:
            X: Features para predicción (None para forecast futuro)
            forecast_horizon: Horizonte de predicción
            model_name: Nombre del modelo a usar (None = mejor modelo)

        Returns:
            Array con predicciones
        """
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Modelo {model_name} no encontrado")
            model = self.models[model_name]
        else:
            if self.best_model is None:
                raise ValueError("No hay modelo entrenado disponible")
            model = self.best_model

        self.logger.info(f"Realizando forecast con {model.name} (horizonte: {forecast_horizon})")

        predictions = model.predict(X, forecast_horizon)

        return predictions

    def save_models(self) -> None:
        """Guarda todos los modelos entrenados."""
        self.config.model_save_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            if model.is_trained:
                filepath = self.config.get_model_path(name)
                model.save(filepath)

        self.logger.info(f"Modelos guardados en {self.config.model_save_dir}")

    def load_models(self) -> None:
        """Carga modelos guardados."""
        for name in self.models.keys():
            filepath = self.config.get_model_path(name)
            if filepath.exists():
                self.models[name].load(filepath)

        self.logger.info("Modelos cargados")

    def _evaluate_model(self, model: BaseForecaster,
                       X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evalúa un modelo usando validación cruzada temporal.

        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=3)
            metrics = {'mae': [], 'rmse': [], 'r2': []}

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Re-entrenar en fold de entrenamiento
                model.fit(X_train, y_train)

                # Predecir en validación
                y_pred = model.predict(X_val)

                # Calcular métricas
                metrics['mae'].append(mean_absolute_error(y_val, y_pred))
                metrics['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
                metrics['r2'].append(r2_score(y_val, y_pred))

            # Promediar métricas
            avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

            return avg_metrics

        except Exception as e:
            self.logger.error(f"Error evaluando modelo: {e}")
            return {}

    def _select_best_model(self, results: Dict[str, Dict]) -> None:
        """Selecciona el mejor modelo basado en MAE."""
        successful_models = {
            name: info for name, info in results.items()
            if info.get('success', False)
        }

        if not successful_models:
            self.logger.warning("Ningún modelo se entrenó exitosamente")
            return

        # Seleccionar por MAE más bajo
        best_name = min(
            successful_models.keys(),
            key=lambda x: successful_models[x]['metrics'].get('mae', float('inf'))
        )

        self.best_model = self.models[best_name]
        self.logger.info(f"Mejor modelo seleccionado: {best_name} (MAE: {successful_models[best_name]['metrics'].get('mae', 'N/A'):.2f})")

    def get_forecast_report(self, predictions: np.ndarray,
                          actual: Optional[np.ndarray] = None,
                          dates: Optional[pd.DatetimeIndex] = None) -> Dict:
        """
        Genera reporte de forecast.

        Args:
            predictions: Predicciones del modelo
            actual: Valores reales (opcional)
            dates: Fechas de las predicciones (opcional)

        Returns:
            Diccionario con reporte de forecast
        """
        report = {
            'predictions_count': len(predictions),
            'predictions_range': {
                'min': float(predictions.min()),
                'max': float(predictions.max()),
                'mean': float(predictions.mean()),
                'std': float(predictions.std())
            }
        }

        if actual is not None and len(actual) == len(predictions):
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            report['metrics'] = {
                'mae': float(mean_absolute_error(actual, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(actual, predictions))),
                'mape': float(np.mean(np.abs((actual - predictions) / actual)) * 100)
            }

        if dates is not None:
            report['date_range'] = {
                'start': str(dates.min()),
                'end': str(dates.max())
            }

        return report