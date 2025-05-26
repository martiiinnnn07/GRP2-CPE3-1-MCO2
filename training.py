import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import os
import json
import logging

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('training.log')  # Save logs to file
    ]
)
logger = logging.getLogger(__name__)

# Custom callback to track loss per epoch
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

# Load data
def load_data():
    logger.info("Attempting to load cleaned_dataset.parquet")
    try:
        df = pd.read_parquet('datasets/cleaned_dataset.parquet', engine='pyarrow')
        logger.info(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        if df.empty:
            logger.error("Dataset is empty")
            return None
        return df
    except FileNotFoundError:
        logger.error("cleaned_dataset.parquet not found in datasets/")
        return None
    except Exception as e:
        logger.error(f"Error loading cleaned_dataset.parquet: {str(e)}")
        return None

# Prepare univariate data with train-validation split
def prepare_univariate_data(data, param, window_size=7, val_split=0.2):
    logger.info(f"Preparing univariate data for {param}")
    try:
        scaler = MinMaxScaler()
        values = data[param].dropna().values.reshape(-1, 1)
        logger.info(f"Parameter {param} has {len(values)} non-null values")
        if len(values) < window_size + 1:
            logger.warning(f"Insufficient data for {param}: {len(values)} rows, need at least {window_size + 1}")
            return None, None, None, None, None
        scaled = scaler.fit_transform(values)
        X, y = [], []
        for i in range(len(scaled) - window_size):
            X.append(scaled[i:i + window_size])
            y.append(scaled[i + window_size])
        X, y = np.array(X), np.array(y)
        logger.info(f"Created {len(X)} samples for {param}")
        if len(X) < 2:
            logger.warning(f"Not enough data points for {param} after windowing: {len(X)} samples")
            return None, None, None, None, None
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        logger.info(f"Train split: {len(X_train)} samples, Validation split: {len(X_val)} samples")
        return X_train, y_train, X_val, y_val, scaler
    except Exception as e:
        logger.error(f"Error preparing univariate data for {param}: {str(e)}")
        return None, None, None, None, None

# Prepare multivariate data with train-validation split
def prepare_multivariate_data(data, params, window_size=7, val_split=0.2):
    logger.info(f"Preparing multivariate data for {len(params)} parameters")
    try:
        scaler = MinMaxScaler()
        values = data[params].dropna().values
        logger.info(f"Multivariate data has {len(values)} complete rows")
        if len(values) < window_size + 1:
            logger.warning(f"Insufficient data for multivariate training: {len(values)} rows, need at least {window_size + 1}")
            return None, None, None, None, None
        scaled = scaler.fit_transform(values)
        X, y = [], []
        for i in range(len(scaled) - window_size):
            X.append(scaled[i:i + window_size])
            y.append(scaled[i + window_size])
        X, y = np.array(X), np.array(y)
        logger.info(f"Created {len(X)} multivariate samples")
        if len(X) < 2:
            logger.warning(f"Not enough multivariate data points after windowing: {len(X)} samples")
            return None, None, None, None, None
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        logger.info(f"Multivariate train split: {len(X_train)} samples, Validation split: {len(X_val)} samples")
        return X_train, y_train, X_val, y_val, scaler
    except Exception as e:
        logger.error(f"Error preparing multivariate data: {str(e)}")
        return None, None, None, None, None

# Build CNN model
def build_cnn(input_shape):
    logger.info(f"Building CNN model with input shape {input_shape}")
    try:
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1 if input_shape[-1] == 1 else input_shape[-1])
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        logger.info("CNN model built successfully")
        return model
    except Exception as e:
        logger.error(f"Error building CNN model: {str(e)}")
        return None

# Build LSTM model
def build_lstm(input_shape):
    logger.info(f"Building LSTM model with input shape {input_shape}")
    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1 if input_shape[-1] == 1 else input_shape[-1])
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        logger.info("LSTM model built successfully")
        return model
    except Exception as e:
        logger.error(f"Error building LSTM model: {str(e)}")
        return None

# Build Hybrid CNN-LSTM model
def build_hybrid(input_shape):
    logger.info(f"Building Hybrid CNN-LSTM model with input shape {input_shape}")
    try:
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(1 if input_shape[-1] == 1 else input_shape[-1])
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        logger.info("Hybrid model built successfully")
        return model
    except Exception as e:
        logger.error(f"Error building Hybrid model: {str(e)}")
        return None

# Save JSON results with error handling
def save_training_results(results, file_path):
    logger.info(f"Attempting to save training results to {file_path}")
    try:
        with open(file_path, 'w') as f:
            json.dump(results, f)
        logger.info(f"Successfully saved {file_path}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {str(e)}")

# Train and save models
def train_models():
    logger.info("Starting model training")
    df = load_data()
    if df is None or df.empty:
        logger.error("No data loaded. Training aborted.")
        return

    # Define parameters
    try:
        params = sorted([col for col in df.select_dtypes(include=np.number).columns
                        if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition', 'Wind Direction']
                        and df[col].notna().any()])
        logger.info(f"Found {len(params)} numeric parameters: {params}")
    except Exception as e:
        logger.error(f"Error identifying parameters: {str(e)}")
        return

    try:
        os.makedirs('models', exist_ok=True)
        os.makedirs('training_results', exist_ok=True)
        logger.info("Created models and training_results directories")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        return

    # Train univariate models
    for param in params:
        logger.info(f"Training models for {param}")
        try:
            X_train, y_train, X_val, y_val, scaler = prepare_univariate_data(df, param)
            if X_train is None:
                logger.warning(f"Skipping {param} due to insufficient data")
                continue

            # CNN
            logger.info(f"Training CNN for {param}")
            cnn_model = build_cnn((X_train.shape[1], 1))
            if cnn_model is None:
                logger.error(f"Failed to build CNN for {param}")
                continue
            cnn_history = LossHistory()
            cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                          callbacks=[cnn_history], verbose=0)
            cnn_model.save(f"models/cnn_{param}.keras")
            logger.info(f"Saved cnn_{param}.keras")
            y_pred_cnn = cnn_model.predict(X_val, verbose=0)
            y_pred_cnn = scaler.inverse_transform(y_pred_cnn).flatten()
            y_val_cnn = scaler.inverse_transform(y_val).flatten()
            cnn_results = {
                'epochs': 50,
                'loss': cnn_history.losses,
                'val_loss': cnn_history.val_losses,
                'actual': y_val_cnn.tolist(),
                'predicted': y_pred_cnn.tolist()
            }
            save_training_results(cnn_results, f"training_results/cnn_{param}_training_results.json")

            # LSTM
            logger.info(f"Training LSTM for {param}")
            lstm_model = build_lstm((X_train.shape[1], 1))
            if lstm_model is None:
                logger.error(f"Failed to build LSTM for {param}")
                continue
            lstm_history = LossHistory()
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                           callbacks=[lstm_history], verbose=0)
            lstm_model.save(f"models/lstm_{param}.keras")
            logger.info(f"Saved lstm_{param}.keras")
            y_pred_lstm = lstm_model.predict(X_val, verbose=0)
            y_pred_lstm = scaler.inverse_transform(y_pred_lstm).flatten()
            y_val_lstm = scaler.inverse_transform(y_val).flatten()
            lstm_results = {
                'epochs': 50,
                'loss': lstm_history.losses,
                'val_loss': lstm_history.val_losses,
                'actual': y_val_lstm.tolist(),
                'predicted': y_pred_lstm.tolist()
            }
            save_training_results(lstm_results, f"training_results/lstm_{param}_training_results.json")

            # Hybrid
            logger.info(f"Training Hybrid for {param}")
            hybrid_model = build_hybrid((X_train.shape[1], 1))
            if hybrid_model is None:
                logger.error(f"Failed to build Hybrid for {param}")
                continue
            hybrid_history = LossHistory()
            hybrid_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                             callbacks=[hybrid_history], verbose=0)
            hybrid_model.save(f"models/hybrid_{param}.keras")
            logger.info(f"Saved hybrid_{param}.keras")
            y_pred_hybrid = hybrid_model.predict(X_val, verbose=0)
            y_pred_hybrid = scaler.inverse_transform(y_pred_hybrid).flatten()
            y_val_hybrid = scaler.inverse_transform(y_val).flatten()
            hybrid_results = {
                'epochs': 50,
                'loss': hybrid_history.losses,
                'val_loss': hybrid_history.val_losses,
                'actual': y_val_hybrid.tolist(),
                'predicted': y_pred_hybrid.tolist()
            }
            save_training_results(hybrid_results, f"training_results/hybrid_{param}_training_results.json")
        except Exception as e:
            logger.error(f"Error training models for {param}: {str(e)}")
            continue

    # Train multivariate models
    logger.info("Training multivariate models")
    try:
        X_train, y_train, X_val, y_val, scaler = prepare_multivariate_data(df, params)
        if X_train is None:
            logger.warning("Skipping multivariate training due to insufficient data")
            return

        # CNN
        logger.info("Training multivariate CNN")
        cnn_model = build_cnn((X_train.shape[1], X_train.shape[2]))
        if cnn_model is None:
            logger.error("Failed to build multivariate CNN")
            return
        cnn_history = LossHistory()
        cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                      callbacks=[cnn_history], verbose=0)
        cnn_model.save("models/cnn_multivariate.keras")
        logger.info("Saved cnn_multivariate.keras")
        y_pred_cnn = cnn_model.predict(X_val, verbose=0)
        y_pred_cnn = scaler.inverse_transform(y_pred_cnn)
        y_val_cnn = scaler.inverse_transform(y_val)
        cnn_results = {
            'epochs': 50,
            'loss': cnn_history.losses,
            'val_loss': cnn_history.val_losses,
            'actual': {param: y_val_cnn[:, i].tolist() for i, param in enumerate(params)},
            'predicted': {param: y_pred_cnn[:, i].tolist() for i, param in enumerate(params)}
        }
        save_training_results(cnn_results, "training_results/cnn_multivariate_training_results.json")

        # LSTM
        logger.info("Training multivariate LSTM")
        lstm_model = build_lstm((X_train.shape[1], X_train.shape[2]))
        if lstm_model is None:
            logger.error("Failed to build multivariate LSTM")
            return
        lstm_history = LossHistory()
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                       callbacks=[lstm_history], verbose=0)
        lstm_model.save("models/lstm_multivariate.keras")
        logger.info("Saved lstm_multivariate.keras")
        y_pred_lstm = lstm_model.predict(X_val, verbose=0)
        y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
        y_val_lstm = scaler.inverse_transform(y_val)
        lstm_results = {
            'epochs': 50,
            'loss': lstm_history.losses,
            'val_loss': lstm_history.val_losses,
            'actual': {param: y_val_lstm[:, i].tolist() for i, param in enumerate(params)},
            'predicted': {param: y_pred_lstm[:, i].tolist() for i, param in enumerate(params)}
        }
        save_training_results(lstm_results, "training_results/lstm_multivariate_training_results.json")

        # Hybrid
        logger.info("Training multivariate Hybrid")
        hybrid_model = build_hybrid((X_train.shape[1], X_train.shape[2]))
        if hybrid_model is None:
            logger.error("Failed to build multivariate Hybrid")
            return
        hybrid_history = LossHistory()
        hybrid_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),
                         callbacks=[hybrid_history], verbose=0)
        hybrid_model.save("models/hybrid_multivariate.keras")
        logger.info("Saved hybrid_multivariate.keras")
        y_pred_hybrid = hybrid_model.predict(X_val, verbose=0)
        y_pred_hybrid = scaler.inverse_transform(y_pred_hybrid)
        y_val_hybrid = scaler.inverse_transform(y_val)
        hybrid_results = {
            'epochs': 50,
            'loss': hybrid_history.losses,
            'val_loss': hybrid_history.val_losses,
            'actual': {param: y_val_hybrid[:, i].tolist() for i, param in enumerate(params)},
            'predicted': {param: y_pred_hybrid[:, i].tolist() for i, param in enumerate(params)}
        }
        save_training_results(hybrid_results, "training_results/hybrid_multivariate_training_results.json")
    except Exception as e:
        logger.error(f"Error training multivariate models: {str(e)}")

if __name__ == "__main__":
    logger.info("Script execution started")
    train_models()
    logger.info("Script execution completed")