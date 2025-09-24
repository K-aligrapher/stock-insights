from transformers import pipeline

# Load FinBERT model
sentiment_model = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def analyze_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    results = sentiment_model(texts)
    pos, neg, neu = 0, 0, 0
    # ...existing code...
    for r in results:
         label = r['label'].lower()
         if label == 'positive':
           pos += 1
         elif label == 'negative':
           neg += 1
         elif label == 'neutral':
           neu += 1
         else:
           print(f"Unknown label: {r['label']}")
# ...existing code...
    total = len(results)
    return {
        "positive": round((pos/total)*100, 2),
        "negative": round((neg/total)*100, 2),
        "neutral": round((neu/total)*100, 2)
    }
