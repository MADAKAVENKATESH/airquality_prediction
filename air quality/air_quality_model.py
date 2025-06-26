import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os

class AirQualityPredictor:
    def __init__(self, sequence_length=24, features=None):
        self.sequence_length = sequence_length
        self.features = features
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = None
        
    def _build_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.features)),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)  # Predicts AQI
        ])
        
        # Use a lower learning rate for better convergence
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        return model
    
    def prepare_data(self, data, target_column='AQI', fit_scalers=True):
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Extract target column
        if target_column not in data_copy.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        y_data = data_copy[target_column].values.reshape(-1, 1)
        X_data = data_copy.drop(columns=[target_column])
        
        # Store feature columns if not already set
        if self.feature_columns is None:
            self.feature_columns = X_data.columns.tolist()
        
        # If features not set, determine from data (excluding target)
        if self.features is None:
            self.features = X_data.shape[1]
            self.model = self._build_model()
        
        # Scale features and target
        if fit_scalers:
            scaled_X = self.feature_scaler.fit_transform(X_data)
            scaled_y = self.target_scaler.fit_transform(y_data)
        else:
            scaled_X = self.feature_scaler.transform(X_data)
            scaled_y = self.target_scaler.transform(y_data) if y_data.size > 0 else None
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_X) - self.sequence_length):
            X.append(scaled_X[i:(i + self.sequence_length)])
            if scaled_y is not None:
                y.append(scaled_y[i + self.sequence_length][0])
                
        X = np.array(X)
        y = np.array(y) if y else None
            
        return X, y
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        # Add learning rate reducer and model checkpoint
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Add model checkpoint to save the best model
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model with callbacks
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Load the best model
        if os.path.exists('models/best_model.h5'):
            self.model = tf.keras.models.load_model('models/best_model.h5')
            
        return history
    
    def predict(self, X):
        if self.feature_scaler is None or self.target_scaler is None:
            raise ValueError("Scaler not initialized. Please train the model first.")
            
        # If input is a pandas DataFrame, make a copy to avoid modifying the original
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            # Ensure we only keep the features used during training
            if self.feature_columns is not None:
                missing_cols = set(self.feature_columns) - set(X.columns)
                if missing_cols:
                    raise ValueError(f"Missing feature columns: {missing_cols}")
                X = X[self.feature_columns]
        
        # Convert to numpy array if it's not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # If we have a single 2D sample, reshape to 3D (batch_size, sequence_length, features)
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
        
        # Scale the input features
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        # Ensure we have the right number of features
        if X_flat.shape[1] != self.features:
            raise ValueError(f"Expected {self.features} features, got {X_flat.shape[1]}")
            
        # Scale features using the same scaler as training
        X_scaled = self.feature_scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Make prediction
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Inverse transform the prediction to original scale
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        # Ensure predictions are within valid AQI range (0-500)
        y_pred = np.clip(y_pred, 0, 500)
        
        return y_pred.reshape(-1)
    
    def save_model(self, model_path='models/air_quality_model.h5'):
        os.makedirs('models', exist_ok=True)
        if self.model is not None:
            self.model.save(model_path)
        # Save scalers and metadata
        scalers = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'features': self.features,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }
        joblib.dump(scalers, 'models/scalers.pkl')
    
    def load_model(self, model_path='models/air_quality_model.h5'):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        
        # Load scalers and metadata
        scaler_path = 'models/scalers.pkl'
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.feature_scaler = scalers['feature_scaler']
            self.target_scaler = scalers['target_scaler']
            self.features = scalers['features']
            self.feature_columns = scalers['feature_columns']
            self.sequence_length = scalers['sequence_length']
            
            # Rebuild model with correct input shape if not loaded from file
            if self.model is None:
                self.model = self._build_model()

def get_preventive_measures(aqi_level):
    """
    Returns preventive measures based on AQI level
    """
    if aqi_level <= 50:
        return "Air quality is good. No significant health risks. Enjoy outdoor activities."
    elif aqi_level <= 100:
        return "Moderate air quality. Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi_level <= 150:
        return "Unhealthy for sensitive groups. People with heart or lung disease, older adults, and children should reduce prolonged outdoor exertion."
    elif aqi_level <= 200:
        return "Unhealthy. Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion."
    elif aqi_level <= 300:
        return "Very Unhealthy. Health alert: everyone may experience more serious health effects. Avoid outdoor activities."
    else:
        return "Hazardous. Health warning of emergency conditions. Everyone should avoid all physical activity outdoors."
