# forecasting.py
import pandas as pd
import yfinance as yf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np


# ----------------------------
# 1. Fetch Stock Data
# ----------------------------
def get_stock_data(ticker, period="5y"):
    """
    Download stock data and return standardized DataFrame
    with columns: Date, Close
    """
    df = yf.download(ticker, period=period)

    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    # Ensure Close column exists
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    df = df[["Date", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = df["Close"].astype(float)

    return df


# ----------------------------
# 2. Prophet Forecast
# ----------------------------
def prophet_forecast(df, periods=30):
    """
    Forecast stock prices using Prophet.
    """
    df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet = df_prophet.dropna(subset=["ds", "y"])

    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast, df_prophet
# ----------------------------
# 3. LSTM Forecast
# ----------------------------
def lstm_forecast(df, look_back=60):
    """
    Forecast next-day stock price using LSTM.
    Returns a single predicted value.
    """
    close_data = df["Close"].values.reshape(-1, 1)

    # scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    # create training sequences
    X_train, y_train = [], []
    for i in range(look_back, len(scaled_data)):
        X_train.append(scaled_data[i - look_back:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # define LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # train
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # predict next day
    last_sequence = scaled_data[-look_back:]
    x_input = np.reshape(last_sequence, (1, look_back, 1))
    pred_scaled = model.predict(x_input, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)

    return float(pred[0, 0])
