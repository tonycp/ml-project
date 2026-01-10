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
                
                # Determinar frecuencia según los datos de entrenamiento
                if hasattr(self.model.history['ds'], 'freq') and 'H' in str(self.model.history['ds'].freq):
                    freq = 'H'
                else:
                    freq = 'D'
                
                future_dates = pd.DataFrame({
                    'ds': pd.date_range(
                        start=last_date + pd.Timedelta(hours=1) if freq == 'H' else pd.Timedelta(days=1),
                        periods=forecast_horizon,
                        freq=freq
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
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
            from tensorflow.keras.optimizers import Adam
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.exceptions import NotFittedError

            self.logger.info(f"Entrenando LSTM con sequence_length={self.sequence_length}")

            # Validar datos de entrada
            if X.isnull().values.any():
                self.logger.warning("Se detectaron valores NaN en los datos de entrada. Reemplazando con ceros.")
                X = X.fillna(0.0)
            
            if y.isnull().any():
                self.logger.warning("Se detectaron valores NaN en el target. Reemplazando con ceros.")
                y = y.fillna(0.0)

            # Verificar que hay suficientes datos
            if len(X) < self.sequence_length * 2:  # Necesitamos al menos 2 secuencias para entrenamiento
                raise ValueError(f"No hay suficientes datos para entrenar. Se necesitan al menos {self.sequence_length * 2} muestras, pero solo hay {len(X)}")

            # Escalar datos
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()

            try:
                X_scaled = self.scaler_X.fit_transform(X)
                y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
            except Exception as e:
                raise ValueError(f"Error al escalar los datos: {str(e)}")

            # Crear secuencias
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
            
            if len(X_seq) == 0 or len(y_seq) == 0:
                raise ValueError("No se pudieron crear secuencias de entrenamiento. Verifique los datos de entrada.")

            # Construir modelo
            try:
                self.model = Sequential([
                    Input(shape=(X_seq.shape[1], X_seq.shape[2])),
                    LSTM(64, activation='relu', return_sequences=True),
                    Dropout(self.lstm_config['dropout_rate']),
                    LSTM(32, activation='relu'),
                    Dropout(self.lstm_config['dropout_rate']),
                    Dense(1)
                ])

                # Usar learning rate del config o valor por defecto
                learning_rate = float(self.lstm_config.get('learning_rate', 0.001))
                
                self.model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='mse',
                    metrics=['mae']
                )

                # Entrenar con validación
                history = self.model.fit(
                    X_seq, 
                    y_seq,
                    epochs=int(self.lstm_config.get('epochs', 100)),
                    batch_size=int(self.lstm_config.get('batch_size', 32)),
                    validation_split=0.1,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )

                self.is_trained = True
                self.logger.info("LSTM entrenado exitosamente")
                self.logger.info(f"Pérdida final: {history.history['loss'][-1]:.4f}, MAE: {history.history['mae'][-1]:.4f}")
                
                if 'val_loss' in history.history:
                    self.logger.info(f"Pérdida de validación: {history.history['val_loss'][-1]:.4f}, MAE de validación: {history.history['val_mae'][-1]:.4f}")

            except Exception as e:
                self.logger.error(f"Error al construir o entrenar el modelo: {str(e)}")
                raise

        except ImportError as e:
            error_msg = "TensorFlow no está instalado. Instale con: pip install tensorflow"
            self.logger.error(error_msg)
            raise ImportError(error_msg) from e
            
        except Exception as e:
            self.logger.error(f"Error inesperado durante el entrenamiento: {str(e)}")
            self.is_trained = False
            raise

        return self

    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones con LSTM."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        try:
            # Validar entrada
            if X.isnull().values.any():
                self.logger.warning("Se detectaron valores NaN en los datos de entrada. Reemplazando con ceros.")
                X = X.fillna(0.0)
            
            # Verificar que hay suficientes datos
            if len(X) < self.sequence_length:
                self.logger.warning(f"No hay suficientes datos para hacer predicciones. Se necesitan al menos {self.sequence_length} puntos, pero solo hay {len(X)}.")
                return np.zeros(len(X))
            
            # Escalar datos
            try:
                X_scaled = self.scaler_X.transform(X)
            except Exception as e:
                self.logger.error(f"Error al escalar datos: {e}")
                return np.zeros(len(X))
            
            # Crear secuencias para predicción
            try:
                X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
                if len(X_seq) == 0:
                    self.logger.error("No se pudieron crear secuencias para predicción")
                    return np.zeros(len(X))
            except Exception as e:
                self.logger.error(f"Error al crear secuencias: {e}")
                return np.zeros(len(X))
            
            # Hacer predicciones
            try:
                predictions_scaled = self.model.predict(X_seq, verbose=0)
                
                # Verificar predicciones
                if np.isnan(predictions_scaled).any() or np.isinf(predictions_scaled).any():
                    self.logger.warning("Se detectaron valores no válidos en las predicciones escaladas. Reemplazando con ceros.")
                    predictions_scaled = np.nan_to_num(predictions_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Invertir escalado
                try:
                    predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))
                    
                    if np.isnan(predictions).any() or np.isinf(predictions).any():
                        self.logger.warning("Se detectaron valores no válidos después de la inversión de escala. Reemplazando con ceros.")
                        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Crear array de salida
                    full_predictions = np.zeros(len(X))
                    start_idx = len(X) - len(predictions)
                    full_predictions[start_idx:] = predictions.ravel()
                    
                    return full_predictions
                    
                except Exception as e:
                    self.logger.error(f"Error al invertir el escalado: {e}")
                    return np.zeros(len(X))
                    
            except Exception as e:
                self.logger.error(f"Error al hacer predicciones: {e}")
                return np.zeros(len(X))
                
        except Exception as e:
            self.logger.error(f"Error en predicción LSTM: {str(e)}")
            return np.zeros(len(X))

        except Exception as e:
            self.logger.error(f"Error en predicción LSTM: {e}")
            # En caso de error, devolver un array de ceros del tamaño correcto
            return np.zeros(len(X))

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



class XGBoostModel(BaseForecaster):
    """Modelo XGBoost para forecasting de series temporales."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, 'xgboost')
        self.xgb_config = config.models.get('xgboost', {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        })

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostModel':
        """Entrena modelo XGBoost."""
        try:
            import xgboost as xgb

            self.logger.info(f"Entrenando XGBoost con {self.xgb_config['n_estimators']} estimadores")

            self.model = xgb.XGBRegressor(**self.xgb_config)
            self.model.fit(X, y)

            self.is_trained = True
            self.logger.info("XGBoost entrenado exitosamente")

            # Calcular importancia de features
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        except ImportError:
            raise ImportError("xgboost no está instalado. Instale con: pip install xgboost")
        except Exception as e:
            self.logger.error(f"Error entrenando XGBoost: {e}")
            raise

        return self

    def predict(self, X: pd.DataFrame, forecast_horizon: int = 1) -> np.ndarray:
        """Realiza predicciones con XGBoost."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        try:
            predictions = self.model.predict(X)
            return predictions

        except Exception as e:
            self.logger.error(f"Error en predicción XGBoost: {e}")
            raise

    def save(self, filepath: Path) -> None:
        """Guarda modelo XGBoost."""
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'config': self.xgb_config,
            'model_config': self.config.__dict__
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Modelo XGBoost guardado en {filepath}")

    def load(self, filepath: Path) -> 'XGBoostModel':
        """Carga modelo XGBoost."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance', {})
        self.xgb_config = model_data['config']
        self.is_trained = True

        self.logger.info(f"Modelo XGBoost cargado desde {filepath}")

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

    def generate_learning_curve(self, model_name: str, X: pd.DataFrame, y: pd.Series, train_sizes: np.ndarray = None) -> Dict:
        """
        Genera curva de aprendizaje para un modelo específico.
        
        Args:
            model_name: Nombre del modelo a evaluar
            X: Features
            y: Target
            train_sizes: Array de tamaños de entrenamiento (fracciones de 0.1 a 1.0)
            
        Returns:
            Diccionario con datos de la curva de aprendizaje
        """
        from sklearn.model_selection import learning_curve
        from sklearn.metrics import mean_absolute_error, make_scorer
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        
        # Para modelos que no son scikit-learn, necesitamos un enfoque diferente
        if hasattr(model, 'model') and hasattr(model.model, 'fit'):
            # Es un modelo scikit-learn compatible
            try:
                # Crear scorer personalizado para MAE
                mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                
                # Generar curva de aprendizaje
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    model.model, X, y,
                    train_sizes=train_sizes,
                    cv=3,  # 3-fold cross-validation
                    scoring=mae_scorer,
                    n_jobs=-1,
                    random_state=42
                )
                
                # Convertir scores negativos (por el scorer) a positivos
                train_scores = np.abs(train_scores)
                val_scores = np.abs(val_scores)
                
                return {
                    'train_sizes': train_sizes_abs,
                    'train_scores_mean': np.mean(train_scores, axis=1),
                    'train_scores_std': np.std(train_scores, axis=1),
                    'val_scores_mean': np.mean(val_scores, axis=1),
                    'val_scores_std': np.std(val_scores, axis=1)
                }
                
            except Exception as e:
                self.logger.warning(f"No se pudo generar curva de aprendizaje automática para {model_name}: {e}")
        
        # Método manual para modelos no compatibles
        return self._generate_manual_learning_curve(model_name, X, y, train_sizes)
    
    def _generate_manual_learning_curve(self, model_name: str, X: pd.DataFrame, y: pd.Series, train_sizes: np.ndarray) -> Dict:
        """
        Genera curva de aprendizaje manualmente para modelos no scikit-learn.
        """
        from sklearn.model_selection import train_test_split
        
        train_scores = []
        val_scores = []
        train_sizes_abs = []
        
        # Dividir datos en entrenamiento y validación
        X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for train_size in train_sizes:
            # Calcular tamaño absoluto del conjunto de entrenamiento
            n_samples = int(len(X_temp) * train_size)
            train_sizes_abs.append(n_samples)
            
            # Tomar subconjunto de entrenamiento
            X_train = X_temp[:n_samples]
            y_train = y_temp[:n_samples]
            
            try:
                # Crear una nueva instancia del modelo para cada tamaño
                model_class = type(self.models[model_name])
                model_instance = model_class(self.models[model_name].config)
                
                # Entrenar con el subconjunto
                model_instance.fit(X_train, y_train)
                
                # Predecir en entrenamiento y validación
                train_pred = model_instance.predict(X_train)
                val_pred = model_instance.predict(X_val)
                
                # Calcular MAE
                train_mae = mean_absolute_error(y_train, train_pred)
                val_mae = mean_absolute_error(y_val, val_pred)
                
                train_scores.append(train_mae)
                val_scores.append(val_mae)
                
            except Exception as e:
                self.logger.warning(f"Error en curva de aprendizaje para tamaño {train_size}: {e}")
                # Usar valores altos si falla
                train_scores.append(float('inf'))
                val_scores.append(float('inf'))
        
        return {
            'train_sizes': np.array(train_sizes_abs),
            'train_scores_mean': np.array(train_scores),
            'train_scores_std': np.zeros_like(train_scores),  # No hay variabilidad en una sola ejecución
            'val_scores_mean': np.array(val_scores),
            'val_scores_std': np.zeros_like(val_scores)
        }

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