import os
import streamlit as st
from newsapi import NewsApiClient
from transformers import pipeline
import logging

# Set up logging to catch any errors
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
# Store your API key in an environment variable for security
# It is better to use `os.environ.get()` in a real app
# But for a simple script, you can define it here.
NEWSAPI_KEY = "e2dacd0e1b9c48c8a3a92119f81e1cca"

# Load FinBERT model for financial sentiment analysis
@st.cache_resource
def load_sentiment_model():
    """Load the FinBERT model and cache it to avoid re-downloading."""
    return pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

sentiment_model = load_sentiment_model()

# --- News API Integration ---
@st.cache_data(show_spinner="Fetching recent financial news...")
def get_financial_news(ticker: str, num_articles: int = 10):
    """
    Fetches the latest financial news headlines for a given stock ticker
    using the NewsAPI.
    """
    if not NEWSAPI_KEY or NEWSAPI_KEY == "e2dacd0e1b9c48c8a3a92119f81e1cca":
        logging.warning("NewsAPI key is not set. Cannot fetch real news.")
        return []

    try:
        newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        
        # Use the 'everything' endpoint to search for the ticker
        # Sorting by 'relevancy' ensures the most relevant articles are returned first.
        articles = newsapi.get_everything(
            q=f"stock {ticker}", # Query for the stock ticker
            language="en",
            sort_by="relevancy",
            page_size=num_articles
        )

        headlines = [article.get('title') for article in articles.get('articles', []) if article.get('title')]
        logging.info(f"Successfully fetched {len(headlines)} headlines for {ticker}.")
        return headlines

    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return []


# --- Sentiment Analysis Logic ---
def analyze_sentiment(texts: list):
    """
    Analyzes the sentiment of a list of texts and returns the percentage
    of positive, negative, and neutral sentiments.
    """
    if not texts:
        return {"positive": 0, "negative": 0, "neutral": 0}

    # Perform sentiment analysis in batches for efficiency
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
        print(f"Found {len(news)} articles for {test_ticker}:")
        for i, headline in enumerate(news):
            print(f"{i+1}. {headline}")
        
        sentiment_scores = analyze_sentiment(news)
        print("\nSentiment Analysis Results:")
        for sentiment, score in sentiment_scores.items():
            print(f"- {sentiment.capitalize()}: {score}%")
    else:
        print(f"No news articles found for {test_ticker}.")