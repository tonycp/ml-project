# aircraft_forecasting_pycaret.ipynb
# =================================
# PyCaret Integration with Custom Models for Aircraft Forecasting

# 1. Import Libraries
# -------------------
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from pycaret.regression import *

# Add parent directory to path
sys.path.append(str(Path().cwd().parent))

# 2. Setup Logging
# ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 3. Import Custom Models
# -----------------------
from models import (
    ModelConfig,
    ATCAircraftDataLoader,
    AircraftDataPreprocessor,
    AircraftFeatureEngineer,
    ARIMAModel,
    ProphetModel,
    RandomForestModel,
    LSTMModel
)

# 4. Configuration
# ----------------
config = ModelConfig()
logger.info("Configuration loaded")

# 5. Load and Prepare Data
# ------------------------
def load_and_prepare_data():
    """Load and preprocess data using existing pipeline."""
    logger.info("Loading and preparing data...")
    
    # Load data
    data_loader = ATCAircraftDataLoader(config)
    df = data_loader.get_training_data('daily_atc')
    
    # Preprocess
    preprocessor = AircraftDataPreprocessor(config)
    df_processed = preprocessor.preprocess_daily_data(df)
    
    # Feature engineering
    feature_engineer = AircraftFeatureEngineer(config)
    df_featured = feature_engineer.create_features(df_processed)
    df_featured = feature_engineer.create_lagged_target(df_featured, forecast_horizon=1)
    
    # Prepare features and target
    X, y = feature_engineer.select_features_for_model(df_featured)
    return X, y, df_featured

X, y, df_featured = load_and_prepare_data()

# 6. PyCaret Setup with Custom Models
# -----------------------------------
def setup_pycaret(X, y):
    """Setup PyCaret environment with custom models."""
    # Combine features and target for PyCaret
    data = X.copy()
    data['target'] = y
    
    # Initialize PyCaret
    exp = setup(
        data=data,
        target='target',
        train_size=0.8,
        fold=5,
        session_id=42,
        log_experiment=True,
        experiment_name='aircraft_forecasting',
        silent=True,
        verbose=False
    )
    return exp

exp = setup_pycaret(X, y)

# 7. Custom Model Wrappers for PyCaret
# ------------------------------------
from pycaret.containers.models.base_model import (
    TimeSeriesModelContainer,
    TimeSeriesModelMixin
)

class CustomARIMA(TimeSeriesModelMixin):
    def __init__(self):
        self.model = ARIMAModel(config)
        self._fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise Exception("Model not fitted yet")
        return self.model.predict(X)

class CustomProphet(TimeSeriesModelMixin):
    def __init__(self):
        self.model = ProphetModel(config)
        self._fitted = False

    def fit(self, X, y):
        # Prophet expects a DataFrame with 'ds' and 'y' columns
        df = X.copy()
        df['y'] = y
        if 'ds' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'ds'})
        self.model.fit(df, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise Exception("Model not fitted yet")
        # Ensure we have the 'ds' column for prediction
        if 'ds' not in X.columns and 'date' in X.columns:
            X = X.rename(columns={'date': 'ds'})
        return self.model.predict(X)

class CustomRandomForest(TimeSeriesModelMixin):
    def __init__(self):
        self.model = RandomForestModel(config)
        self._fitted = False
        self.feature_names = None

    def fit(self, X, y):
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise Exception("Model not fitted yet")
        # Ensure the input has the same features as training data
        if hasattr(self, 'feature_names'):
            X = X[self.feature_names]
        return self.model.predict(X)

# 8. Add Custom Models to PyCaret
# -------------------------------
def add_custom_models():
    """Add custom models to PyCaret."""
    # ARIMA
    add_metric('mae', 'MAE', mean_absolute_error, greater_is_better=False)
    
    # Create and add custom models
    custom_models = {
        'arima': CustomARIMA(),
        'prophet': CustomProphet(),
        'random_forest': CustomRandomForest()
    }
    
    for name, model in custom_models.items():
        try:
            create_model(model, verbose=False)
            logger.info(f"Added custom model: {name}")
        except Exception as e:
            logger.error(f"Error adding {name}: {str(e)}")

add_custom_models()

# 9. Compare Models
# -----------------
best_model = compare_models(
    sort='MAE',
    n_select=3,
    include=['arima', 'lightgbm', 'rf', 'et', 'xgboost']
)

# 10. Create and Tune Best Model
# ------------------------------
tuned_model = tune_model(best_model, optimize='MAE')

# 11. Finalize and Save Model
# ---------------------------
final_model = finalize_model(tuned_model)
save_model(final_model, 'best_aircraft_forecasting_model')

# 12. Make Predictions
# --------------------
predictions = predict_model(final_model, data=X)
print(predictions.head())

# 13. Analyze Model
# -----------------
plot_model(final_model, plot='error')
plot_model(final_model, plot='feature')
evaluate_model(final_model)

# 14. Save Experiment
# -------------------
save_experiment('aircraft_forecasting_experiment')