import streamlit as st
import plotly.graph_objects as go
from forecasting import get_stock_data, prophet_forecast, lstm_forecast
from sentiment import analyze_sentiment
from recommender import recommendation

st.set_page_config(page_title="AI Stock Market Insights", layout="wide")

st.title("📈 AI-Powered Stock Market Insights")
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL, GOOG):", "TSLA")

if st.button("Analyze"):
    # Get stock data
    data = get_stock_data(ticker)
    st.subheader(f"{ticker} Closing Prices")
    st.line_chart(data["Close"])

    # Prophet forecast
    forecast = prophet_forecast(data, periods=30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Prediction"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Actual"))
    st.subheader("Prophet Forecast")
    st.plotly_chart(fig)

    # LSTM forecast
    lstm_pred = lstm_forecast(data)
    st.subheader("LSTM Next Day Prediction")
    st.metric("Predicted Price", f"${lstm_pred:.2f}")

    # Sentiment Analysis
    st.subheader("Market Sentiment")
    sample_texts = [
        "Tesla stock is expected to rise due to strong demand.",
        "Investors worry about Tesla’s production delays.",
        "Analysts say Tesla is performing stable this quarter."
    ]
    sentiment = analyze_sentiment(sample_texts)
    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{sentiment['positive']}%")
    col2.metric("Negative", f"{sentiment['negative']}%")
    col3.metric("Neutral", f"{sentiment['neutral']}%")

    # Final Recommendation
    last_close = float(data["Close"].iloc[-1])
    pred_price = float(lstm_pred)
    price_trend = 1 if pred_price > last_close else -1

    decision = recommendation(price_trend, sentiment)
    st.subheader("📌 Recommendation")
    st.success(decision)
