# 📈 AI-Powered Stock Market Insights
This is a Streamlit-based web application that provides a comprehensive analysis of a given stock ticker by combining historical data forecasting and real-time news sentiment analysis. The app uses a multi-layered approach to deliver a clear "BUY," "SELL," or "HOLD" recommendation.

## ✨ Features
- Historical Data Visualization: Displays the historical closing prices of any stock using interactive line charts.
- Prophet Forecasting: Utilizes Facebook's Prophet model to generate a long-term forecast and predict future price trends.
- LSTM Prediction: Employs a trained LSTM (Long Short-Term Memory) neural network to provide a next-day price prediction, offering a short-term outlook.
- Sentiment Analysis: Analyzes the sentiment of recent financial news headlines for the given stock ticker using the FinBERT model from the Hugging Face transformers library.
- Unified Recommendation: Synthesizes outputs from both forecasting models and the sentiment analysis to produce a final, actionable recommendation (BUY, SELL, or HOLD).

## 🚀 Getting Started
Prerequisites
To run this application locally, you need Python installed. You also need to install the required libraries.

## Installation
Clone this repository:
``` Bash
git clone [https://github.com/K-aligrapher/stock-insights]
cd stock-insights
```
Install the required Python packages. It is highly recommended to use a virtual environment.
``` Bash
pip install -r requirements.txt
```
## Get a News API Key:
The application requires an API key to fetch real-time news headlines for sentiment analysis.

Sign up for a free developer account at NewsAPI.org.

Replace the placeholder "YOUR_NEWSAPI_KEY" with your actual API key in the sentiment.py file.

Alternatively, you can choose to use a different news API with a free tier, such as Alpha Vantage or Finnhub.

## How to Run the App
After installing all the dependencies, run the application from your terminal:
``` Bash
streamlit run app.py
```
This command will start a local Streamlit server and open the application in your default web browser.

## 🔗 Deployment
You can deploy this application using a service like Streamlit Community Cloud.

View the live app here: [https://ai-stock-market-insights.streamlit.app/]

## 🛠 Tech Stack

- Frontend: [Streamlit](https://streamlit.io/)  
- Data Manipulation: [Pandas](https://pandas.pydata.org/)  
- Forecasting: [Prophet](https://facebook.github.io/prophet/)  
- Long-term Forecasting: [Prophet](https://facebook.github.io/prophet/)  
- Short-term Forecasting: [Keras](https://keras.io/) / [TensorFlow](https://www.tensorflow.org/) (Custom LSTM)  
- Sentiment Analysis: [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone) via [Hugging Face Transformers](https://huggingface.co/transformers/)  
- Visualization: [Plotly](https://plotly.com/python/)  
- News Integration: [NewsAPI.org](https://newsapi.org/) with [newsapi-python](https://github.com/mattlisiv/newsapi-python)  


# 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

