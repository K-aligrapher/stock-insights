# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from forecasting import get_stock_data, prophet_forecast, lstm_forecast
from sentiment import analyze_sentiment, get_financial_news

st.set_page_config(page_title="AI Stock Market Insights", layout="wide")
st.title("📈 AI-Powered Stock Market Insights")

# ----------------------------
# Input
# ----------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL, GOOG):", "TSLA")

if st.button("Analyze"):

    # ----------------------------
    # 1. Fetch Stock Data
    # ----------------------------
    data = get_stock_data(ticker)
    st.subheader(f"{ticker} Closing Prices")
    st.line_chart(data.set_index("Date")["Close"])

    # ----------------------------
    # 2. Prophet Forecast
    # ----------------------------
    if data.empty:
      st.error("Error: No data was found for this ticker symbol. Please check the symbol and try again.")
      st.stop() # Stops the app execution gracefully
    else:
      forecast, df_prophet = prophet_forecast(data, periods=30)

    # Plot actual + forecast
    fig = go.Figure()

    # Actual (red)
    fig.add_trace(go.Scatter(
    x=df_prophet["ds"],
    y=df_prophet["y"],
    mode="lines",
    name="Actual",
    line=dict(color="red")
    ))


    # Forecast (green)
    fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="Forecast",
    line=dict(color="green")
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(0,255,0,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Confidence Interval"
    ))

    fig.update_layout(
        title="Prophet Forecast vs Actual",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.subheader("Prophet Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # Calculate Prophet trend
    prophet_trend = 1 if forecast["yhat"].iloc[-1] > forecast["yhat"].iloc[-2] else -1

    # ----------------------------
    # 3. LSTM Forecast
    # ----------------------------
    lstm_pred = lstm_forecast(data)
    st.subheader("LSTM Next Day Prediction")
    st.metric("Predicted Price", f"${lstm_pred:.2f}")

    last_close = float(data["Close"].iloc[-1])
    price_trend = 1 if lstm_pred > last_close else -1

    # ----------------------------
    # 4. Sentiment Analysis (FinBERT)
    # ----------------------------
    st.subheader("Market Sentiment")

    # Fetch real news headlines for the given ticker
    news_headlines = get_financial_news(ticker)

    # If news headlines were successfully fetched, analyze their sentiment
    if news_headlines:
        sentiment = analyze_sentiment(news_headlines)
    else:
        # Handle the case where no news could be found
        st.write("Could not fetch recent news for this ticker.")
        sentiment = {"positive": 0, "negative": 0, "neutral": 0}

    col1, col2, col3 = st.columns(3)
    col1.metric("Positive", f"{sentiment['positive']}%")
    col2.metric("Negative", f"{sentiment['negative']}%")
    col3.metric("Neutral", f"{sentiment['neutral']}%")

    # ----------------------------
    # 5. Final Recommendation
    # ----------------------------
    recommendation_score = price_trend + prophet_trend
    if sentiment["positive"] > sentiment["negative"]:
        recommendation_score += 1
    elif sentiment["negative"] > sentiment["positive"]:
        recommendation_score -= 1

    if recommendation_score > 0:
        decision = "BUY"
    elif recommendation_score < 0:
        decision = "SELL"
    else:
        decision = "HOLD"

    st.subheader("📌 Recommendation")
    if decision == "BUY":
        st.success(f"{decision} ✅")
    elif decision == "SELL":
        st.error(f"{decision} ❌")
    else:
        st.warning(f"{decision} ⚠️")

    # ----------------------------
    # 6. Debug Info
    # ----------------------------
    #st.write("Data head:", data.head())
    #st.write("Forecast head:", forecast.head())
    
