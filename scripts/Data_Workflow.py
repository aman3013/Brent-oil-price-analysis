import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model
import ruptures as rpt
import math
import warnings

warnings.filterwarnings('ignore')

# Data loading and preprocessing
def load_data(file_path):
    brent_oil_data = pd.read_csv(file_path)
    brent_oil_data.columns = brent_oil_data.columns.str.strip()
    brent_oil_data['Date'] = pd.to_datetime(brent_oil_data['Date'], errors='coerce')
    brent_oil_data.dropna(subset=['Date'], inplace=True)
    brent_oil_data.set_index('Date', inplace=True)
    return brent_oil_data

# Outlier detection
def detect_outliers(data):
    Q1 = data['Price'].quantile(0.25)
    Q3 = data['Price'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data['Price'] < (Q1 - 1.5 * IQR)) | (data['Price'] > (Q3 + 1.5 * IQR)))
    return outliers.sum()

# Price plotting
def plot_prices(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Price'], label='Brent Oil Price', color='blue')
    plt.title("Brent Oil Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.show()

# Seasonal decomposition
def seasonal_decomposition(data):
    decomposition = seasonal_decompose(data['Price'], model='multiplicative', period=365)
    return decomposition

def plot_decomposition(decomposition):
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(decomposition.observed, label='Original', color='blue')
    plt.legend(loc='best')
    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend', color='orange')
    plt.legend(loc='best')
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonality', color='green')
    plt.legend(loc='best')
    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residuals', color='red')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Change point detection
def detect_change_points(data):
    price_series = data['Price'].values
    model = rpt.Pelt(model="rbf").fit(price_series)
    penalty_value = 10  # Adjust as needed
    change_points = model.predict(pen=penalty_value)
    return change_points

def plot_change_points(data, change_points):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Price'], label="Brent Oil Price", color="skyblue")
    for cp in change_points[:-1]:  # Exclude the last index
        plt.axvline(data.index[cp], color="red", linestyle="--")
    plt.title("Change Point Detection in Brent Oil Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend(["Price", "Change Point"])
    plt.show()

# Stationarity test
def test_stationarity(timeseries):
    timeseries = timeseries[np.isfinite(timeseries)]
    timeseries = timeseries.dropna()
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# ARIMA model
def arima_model(train_data, test_data):
    model = ARIMA(train_data, order=(1, 1, 1))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    mse = mean_squared_error(test_data, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(test_data, predictions)
    r2 = r2_score(test_data, predictions)
    print(f"ARIMA RMSE: {rmse}, MAE: {mae}, R-squared: {r2}")
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data, label='Actual')
    plt.plot(test_data.index, predictions, label='Predicted')
    plt.title('ARIMA Model: Actual vs Predicted Brent Oil Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# GARCH model
def garch_model(train_data, test_data):
    garch_model = arch_model(train_data, vol='GARCH', p=1, q=1)
    garch_results = garch_model.fit()
    garch_forecast = garch_results.forecast(horizon=len(test_data))
    garch_predictions = garch_forecast.mean.iloc[-1]
    garch_mse = mean_squared_error(test_data, garch_predictions)
    garch_rmse = math.sqrt(garch_mse)
    garch_mae = mean_absolute_error(test_data, garch_predictions)
    garch_r2 = r2_score(test_data, garch_predictions)
    print(f"GARCH RMSE: {garch_rmse}, MAE: {garch_mae}, R-squared: {garch_r2}")
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data, label='Actual')
    plt.plot(test_data.index, garch_predictions, label='GARCH Predicted')
    plt.title('GARCH Model: Actual vs Predicted Brent Oil Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

# Yearly average price visualization
def plot_yearly_avg(data):
    yearly_avg = data.resample('Y').mean()
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_avg.index, yearly_avg['Price'])
    plt.title('Yearly Average Brent Oil Prices')
    plt.xlabel('Year')
    plt.ylabel('Average Price (USD)')
    plt.show()

# Main function
def analyze_brent_oil_prices(file_path):
    brent_oil_data = load_data(file_path)
    num_outliers = detect_outliers(brent_oil_data)
    print(f"Number of outliers detected: {num_outliers}")
    plot_prices(brent_oil_data)
    decomposition = seasonal_decomposition(brent_oil_data)
    plot_decomposition(decomposition)
    change_points = detect_change_points(brent_oil_data)
    plot_change_points(brent_oil_data, change_points)
    brent_oil_data['Price_diff'] = brent_oil_data['Price'].diff().dropna()
    test_stationarity(brent_oil_data['Price_diff'])
    train_data = brent_oil_data['Price'][:int(0.8*len(brent_oil_data))]
    test_data = brent_oil_data['Price'][int(0.8*len(brent_oil_data)):]
    arima_model(train_data, test_data)
    garch_model(train_data, test_data)
    plot_yearly_avg(brent_oil_data)

# Example usage
if __name__ == "__main__":
    file_path = os.path.join('data', 'brent_oil_prices.csv')  # Update with actual file path
    analyze_brent_oil_prices(file_path)
