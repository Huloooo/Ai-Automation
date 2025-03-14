import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TrafficPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data, sequence_length=24):
        """
        Prepare data for LSTM model
        data: DataFrame with columns ['timestamp', 'traffic_density', 'weather', 'events']
        """
        # Scale the features
        scaled_data = self.scaler.fit_transform(data[['traffic_density', 'weather', 'events']])
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 0])  # Predict traffic_density
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history
    
    def predict(self, X):
        predictions = self.model.predict(X)
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.hstack([predictions, np.zeros((len(predictions), 2))])
        )[:, 0]
        return predictions
    
    def plot_predictions(self, actual, predicted, title='Traffic Prediction Results'):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Traffic Density')
        plt.legend()
        plt.show()

def generate_sample_data(n_samples=1000):
    """Generate sample data for demonstration"""
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'traffic_density': np.random.normal(0.5, 0.2, n_samples),
        'weather': np.random.randint(0, 4, n_samples),  # 0: sunny, 1: cloudy, 2: rainy, 3: stormy
        'events': np.random.randint(0, 2, n_samples)    # 0: no event, 1: event
    })
    return data

def main():
    # Generate sample data
    data = generate_sample_data()
    
    # Initialize predictor
    predictor = TrafficPredictor()
    
    # Prepare data
    X, y = predictor.prepare_data(data)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    history = predictor.train(X_train, y_train, epochs=50)
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Plot results
    predictor.plot_predictions(y_test, predictions)

if __name__ == '__main__':
    main() 