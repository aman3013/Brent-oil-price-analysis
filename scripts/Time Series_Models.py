import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

def analyze_brent_oil_data(file_path, forecast_steps=30, arima_order=(1, 1, 1), visualize=True):
    # Load the datasets
    brent_oil_data = pd.read_csv(file_path)
    brent_oil_data['Date'] = pd.to_datetime(brent_oil_data['Date'])

    # Drop duplicates and sort by date
    brent_oil_data = brent_oil_data.drop_duplicates().sort_values(by='Date').reset_index(drop=True)

    # Check for missing values
    missing_values = brent_oil_data.isnull().sum()
    print("Missing Values:\n", missing_values)

    # Descriptive statistics
    print("Descriptive Statistics for Brent Oil Prices:")
    print(brent_oil_data['Price'].describe())

    # Moving Average
    brent_oil_data['Price_MA_30'] = brent_oil_data['Price'].rolling(window=30).mean()
    brent_oil_data['Price_MA_365'] = brent_oil_data['Price'].rolling(window=365).mean()

    # Visualization
    if visualize:
        plt.figure(figsize=(14, 7))
        plt.plot(brent_oil_data['Date'], brent_oil_data['Price'], label='Original Price', color='blue', alpha=0.5)
        plt.plot(brent_oil_data['Date'], brent_oil_data['Price_MA_30'], label='30-Day Moving Average', color='red')
        plt.plot(brent_oil_data['Date'], brent_oil_data['Price_MA_365'], label='365-Day Moving Average', color='green')
        plt.title('Brent Oil Prices with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (USD per barrel)')
        plt.legend()
        plt.show()

    # Price Volatility
    brent_oil_data['Daily_Return'] = brent_oil_data['Price'].pct_change()
    volatility = brent_oil_data['Daily_Return'].std()
    print(f"\nPrice Volatility (Daily Return Std. Dev.): {volatility:.4f}")

    # Detect large price movements
    significant_changes = brent_oil_data[brent_oil_data['Daily_Return'].abs() > 0.05]
    print("\nSignificant Price Movements (>5% change in a day):")
    print(significant_changes[['Date', 'Price', 'Daily_Return']])

    # ADF test for stationarity
    def test_stationarity(timeseries):
        result = adfuller(timeseries)
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        if result[1] <= 0.05:
            print('The time series is stationary.')
        else:
            print('The time series is not stationary.')

    test_stationarity(brent_oil_data['Price'])
    brent_oil_data['Price_diff'] = brent_oil_data['Price'].diff().dropna()
    test_stationarity(brent_oil_data['Price_diff'].dropna())

    # Fit ARIMA model
    model = sm.tsa.ARIMA(brent_oil_data['Price'], order=arima_order)
    model_fit = model.fit()

    # Make predictions
    forecast = model_fit.forecast(steps=forecast_steps)

    # Visualization of ARIMA results
    if visualize:
        plt.figure(figsize=(14, 7))
        plt.plot(brent_oil_data['Price'], label='Historical Price', color='blue')
        plt.plot(pd.date_range(start=brent_oil_data['Date'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label='Forecasted Price', color='orange')
        plt.title('Brent Oil Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    # Fit GARCH model
    garch_model = arch_model(brent_oil_data['Price'].dropna(), vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit()
    print(garch_fit.summary())

    # Plot GARCH volatility
    if visualize:
        plt.figure(figsize=(14, 7))
        plt.plot(garch_fit.conditional_volatility, label='GARCH Volatility', color='orange')
        plt.title('GARCH Model Volatility of Brent Oil Prices')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.show()

    # Fit Markov-Switching model
    ms_model = MarkovRegression(brent_oil_data['Price'], k_regimes=2, trend='c', switching_variance=True)
    ms_fit = ms_model.fit()
    print(ms_fit.summary())

    # Plot regime-switching probabilities
    if visualize:
        plt.figure(figsize=(14, 7))
        plt.plot(ms_fit.smoothed_marginal_probabilities[0], label='Regime 1 Probability', color='blue')
        plt.plot(ms_fit.smoothed_marginal_probabilities[1], label='Regime 2 Probability', color='orange')
        plt.title('Regime-Switching Probabilities for Brent Oil Prices')
        plt.xlabel('Date')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()

    # Prepare data for LSTM
    data = brent_oil_data['Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, time_step=30)

    # Convert to PyTorch tensors
    X = torch.FloatTensor(X).unsqueeze(-1)
    y = torch.FloatTensor(y)

    # Create PyTorch Dataset and DataLoader
    class OilPriceDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = OilPriceDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    # Initialize model, loss function, and optimizer
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Make predictions
    model.eval()
    with torch.no_grad():
        predicted = model(X).numpy()

    # Inverse transform predictions
    predicted_price = scaler.inverse_transform(predicted)

    # Plot results
    if visualize:
        plt.figure(figsize=(14, 7))
        plt.plot(brent_oil_data.index[-len(predicted_price):], predicted_price, label='Predicted Price', color='orange')
        plt.plot(brent_oil_data['Price'][-len(predicted_price):], label='Actual Price', color='blue')
        plt.title('LSTM Prediction of Brent Oil Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    # Prepare data for Prophet
    brent_oil_data_prophet = brent_oil_data.rename(columns={'Date': 'ds', 'Price': 'y'})
    prophet_model = Prophet()
    prophet_model.fit(brent_oil_data_prophet)

    # Future dataframe for predictions
    future = prophet_model.make_future_dataframe(periods=forecast_steps)
    forecast = prophet_model.predict(future)

    # Plot Prophet forecast
    if visualize:
        prophet_model.plot(forecast)
        plt.title('Brent Oil Price Forecast using Prophet')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    # Return results
    return {
        'arima_forecast': forecast,
        'garch_fit': garch_fit,
        'markov_fit': ms_fit,
        'predicted_price': predicted_price
    }

# Example usage
if __name__ == "__main__":
    file_path = 'path/to/brent_oil_data.csv'
    results = analyze_brent_oil_data(file_path)
