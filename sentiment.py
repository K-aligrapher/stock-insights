# sentiment.py
from transformers import pipeline

# Load FinBERT model for financial sentiment
sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def analyze_sentiment(texts):
    """
    Analyze sentiment of a single text or list of texts.
    Returns percentage of positive, negative, neutral sentiments.
    """
    if isinstance(texts, str):
        texts = [texts]  # Convert single string to list

    if not texts:
        return {"positive": 0, "negative": 0, "neutral": 0}

    results = sentiment_model(texts)
    
    pos = sum(1 for r in results if r['label'].lower() == 'positive')
    neg = sum(1 for r in results if r['label'].lower() == 'negative')
    neu = sum(1 for r in results if r['label'].lower() == 'neutral')

    total = len(results)
    return {
        "positive": round((pos / total) * 100, 2),
        "negative": round((neg / total) * 100, 2),
        "neutral": round((neu / total) * 100, 2)
    }
