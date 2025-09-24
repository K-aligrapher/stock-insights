import os
import streamlit as st
from transformers import pipeline
import logging

# Removed the NewsAPI-related imports

# --- Configuration ---
# You can leave your API key here, but the code will use the mocked data
NEWSAPI_KEY = " YUMJ60VLYDEK3GM0"

# Load FinBERT model for financial sentiment analysis
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

sentiment_model = load_sentiment_model()

# --- Mocked News Data ---
def get_financial_news(ticker: str, num_articles: int = 10):
    """
    Returns a mocked list of news headlines for development purposes.
    """
    # Replace the API call with a simple, hardcoded list of headlines.
    # This allows the rest of your app to function.
    st.info("Using mocked news data since the API key is not working.")
    
    # You can add more headlines and sentiments to test different scenarios
    return [
        f"Analysts say {ticker} stock has a strong buy rating today.",
        f"Concerns about {ticker} supply chain issues are rising.",
        f"The market sentiment for {ticker} is mixed.",
        f"{ticker} announces record-breaking quarterly earnings.",
        f"A new lawsuit is filed against {ticker} over recent product recall.",
        f"No significant news on {ticker} has been reported recently."
    ]

# --- Sentiment Analysis Logic ---
def analyze_sentiment(texts: list):
    """
    Analyzes the sentiment of a list of texts...
    (The rest of this function remains the same as your original code)
    """
    if not texts:
        return {"positive": 0, "negative": 0, "neutral": 0}

    try:
        results = sentiment_model(texts)
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return {"positive": 0, "negative": 0, "neutral": 0}

    pos = sum(1 for r in results if r['label'].lower() == 'positive')
    neg = sum(1 for r in results if r['label'].lower() == 'negative')
    neu = sum(1 for r in results if r['label'].lower() == 'neutral')

    total = len(results)
    if total == 0:
        return {"positive": 0, "negative": 0, "neutral": 0}
        
    return {
        "positive": round((pos / total) * 100, 2),
        "negative": round((neg / total) * 100, 2),
        "neutral": round((neu / total) * 100, 2)
    }

# This section is for testing the functions directly
if __name__ == "__main__":
    test_ticker = "TSLA"
    news = get_financial_news(test_ticker)
    if news:
        print(f"Using mocked articles for {test_ticker}:")
        for i, headline in enumerate(news):
            print(f"{i+1}. {headline}")
        
        sentiment_scores = analyze_sentiment(news)
        print("\nSentiment Analysis Results:")
        for sentiment, score in sentiment_scores.items():
            print(f"- {sentiment.capitalize()}: {score}%")
    else:
        print(f"No news articles found for {test_ticker}.")