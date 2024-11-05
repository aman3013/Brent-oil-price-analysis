from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the dataset
try:
    data = pd.read_csv('../data/Copy of BrentOilPrices.csv')
    logging.info("Data loaded successfully.")
except FileNotFoundError:
    logging.error("The specified CSV file was not found.")
    data = pd.DataFrame()  # Fallback to an empty DataFrame

# Convert Date to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date'])

# Prepare the data for modeling
def prepare_data():
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    feature_columns = ['Year', 'Month', 'Day']
    X = data[feature_columns]
    y = data['Price']

    return X, y

# Load and train model once
X, y = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/model-outputs', methods=['GET'])
def get_model_outputs():
    # Predict the prices using the trained model
    y_pred = model.predict(X_test)
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    return jsonify(predictions.to_dict(orient='records'))

@app.route('/api/performance-metrics', methods=['GET'])
def get_performance_metrics():
    # Calculate the MSE and R-squared
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return jsonify({'mse': mse, 'r2': r2})

if __name__ == '__main__':
    logging.info("Brent Oil Prices Data Shape: %s", data.shape)
    logging.info("Data Head:\n%s", data.head())
    app.run(debug=True)
