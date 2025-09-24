# forecasting.py

import yfinance as yf
import pandas as pd
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# 1. Fetch Stock Data
# ---------------------------
def get_stock_data(ticker, start="2020-01-01", end="2025-09-01"):
    """
    Download stock data using yfinance
    """
    data = yf.download(ticker, start=start, end=end)
    return data

# ---------------------------
# 2. Prophet Forecast
# ---------------------------

def prophet_forecast(data, periods=30):
    """
    Forecast next 'periods' days using Prophet
    """
    # Make a copy
    df = data.copy()

    # Reset index to get datetime column
    df = df.reset_index()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # Detect datetime column
    datetime_col = None
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            datetime_col = col
            break
    if datetime_col is None:
        # fallback: create date range
        df['ds'] = pd.date_range(start='2020-01-01', periods=len(df))
    else:
        df['ds'] = pd.to_datetime(df[datetime_col], errors='coerce')

    # Detect 'Close' column
    close_col = None
    for col in df.columns:
        if 'Close' in col:
            close_col = col
            break
    if close_col is None:
        raise KeyError("No 'Close' column found in dataframe")

    # Create 'y' column
    df['y'] = pd.to_numeric(df[close_col], errors='coerce')

    # Drop rows with missing values
    df = df[['ds', 'y']].dropna()

    # Debug print
    print("Columns for Prophet:", df.columns)
    print(df.head())

    # Train Prophet
    model = Prophet()
    model.fit(df)

    # Forecast future
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# ---------------------------
# 3. LSTM Forecast
# ---------------------------
def lstm_forecast(data, time_steps=60, epochs=5):
    """
    Predict next day stock price using LSTM
    """
    df = data[['Close']].values
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(time_steps, len(df_scaled)):
        X.append(df_scaled[i-time_steps:i, 0])
        y.append(df_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    # Predict next day
    last_input = df_scaled[-time_steps:]
    last_input = np.reshape(last_input, (1, time_steps, 1))
    prediction = model.predict(last_input)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]

# ---------------------------
# 4. Main Testing (Optional)
# ---------------------------
if __name__ == "__main__":
    ticker = "TSLA"
    data = get_stock_data(ticker)

    print("Running Prophet Forecast...")
    forecast = prophet_forecast(data, periods=30)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    print("\nRunning LSTM Forecast...")
    lstm_pred = lstm_forecast(data, time_steps=60, epochs=5)
    print(f"Next Day Predicted Price (LSTM): ${lstm_pred:.2f}")
