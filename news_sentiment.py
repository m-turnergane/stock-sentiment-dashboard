import os
import streamlit as st
from dotenv import load_dotenv
import requests
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, BartForConditionalGeneration, BartTokenizer
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Load environment variables
load_dotenv()

@st.cache_data
def get_ticker(input_text):
    # Common company names to ticker mapping
    name_to_ticker = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "facebook": "META",
        "meta": "META",
        "netflix": "NFLX",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        # Add more common stocks as needed
    }
    
    # Check if input is in our common name list
    cleaned_input = input_text.lower().strip()
    if cleaned_input in name_to_ticker:
        return name_to_ticker[cleaned_input]
    
    # If not found in the common list, return the input as is
    return input_text.upper()

@st.cache_data
def fetch_news(query, days=14):  # Increased to 7 days to get more articles
    api_key = os.getenv('NEWS_API_KEY')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://newsapi.org/v2/everything?q={query}&from={start_date.date()}&to={end_date.date()}&sortBy=relevancy&pageSize=10&apiKey={api_key}"
    
    response = requests.get(url)
    news_data = response.json()
    
    if news_data['status'] != 'ok':
        raise Exception(f"Error fetching news: {news_data['message']}")
    
    articles = news_data['articles']
    headlines = [{'title': article['title'], 'datetime': article['publishedAt'], 'source': article['source']['name'], 'url': article['url']} for article in articles]
    
    df = pd.DataFrame(headlines)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

@st.cache_resource
def load_sentiment_analyzer():
    model_name = "ProsusAI/finbert"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(headlines, sentiment_analyzer):
    results = sentiment_analyzer(headlines['title'].tolist())
    
    sentiment_scores = []
    sentiment_labels = []
    for result in results:
        if result['label'] == 'positive':
            sentiment_scores.append(result['score'])
            sentiment_labels.append('POSITIVE')
        elif result['label'] == 'negative':
            sentiment_scores.append(-result['score'])
            sentiment_labels.append('NEGATIVE')
        else:
            sentiment_scores.append(0)
            sentiment_labels.append('NEUTRAL')
    
    headlines['sentiment_score'] = sentiment_scores
    headlines['sentiment_label'] = sentiment_labels
    
    return headlines

def categorize_sentiment(score):
    if score <= -0.5:
        return "Overwhelmingly Negative"
    elif score <= -0.1:
        return "Somewhat Negative"
    elif score <= 0.1:
        return "Neutral"
    elif score <= 0.5:
        return "Somewhat Positive"
    else:
        return "Overwhelmingly Positive"

def summarize_sentiment(analyzed_headlines):
    avg_sentiment = analyzed_headlines['sentiment_score'].mean()
    summary = categorize_sentiment(avg_sentiment)
    positive_ratio = (analyzed_headlines['sentiment_label'] == 'POSITIVE').mean()
    negative_ratio = (analyzed_headlines['sentiment_label'] == 'NEGATIVE').mean()
    neutral_ratio = (analyzed_headlines['sentiment_label'] == 'NEUTRAL').mean()
    return summary, avg_sentiment, positive_ratio, negative_ratio, neutral_ratio

@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

def generate_summary(analyzed_headlines, ticker, sentiment_summary, avg_sentiment):
    summarizer = load_summarization_model()
    
    # Prepare the input text for summarization
    input_text = f"Recent news about {ticker}:\n\n"
    for _, row in analyzed_headlines.head(10).iterrows():
        input_text += f"- {row['title']} (Sentiment: {row['sentiment_label']})\n"
    
    input_text += f"\nOverall sentiment: {sentiment_summary.lower()} with an average score of {avg_sentiment:.2f}."
    
    # Generate summary
    summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    return summary

@st.cache_data
def get_historical_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist, stock.info

def plot_line_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price'
    ))
    
    fig.update_layout(
        title='Historical Price Chart (1 Year)',
        yaxis_title='Price',
        xaxis_title='Date',
        height=400
    )
    
    return fig

def display_fundamental_data(info):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Open:", f"${info.get('open', 'N/A'):.2f}")
        st.write("High:", f"${info.get('dayHigh', 'N/A'):.2f}")
        st.write("Low:", f"${info.get('dayLow', 'N/A'):.2f}")
    
    with col2:
        st.write("Mkt cap:", f"${info.get('marketCap', 'N/A'):,.0f}")
        st.write("P/E ratio:", f"{info.get('trailingPE', 'N/A'):.2f}")
        st.write("Div yield:", f"{info.get('dividendYield', 'N/A')*100:.2f}%" if info.get('dividendYield') else "N/A")
    
    with col3:
        st.write("52-wk high:", f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
        st.write("52-wk low:", f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}")
        st.write("Volume:", f"{info.get('volume', 'N/A'):,}")

@st.cache_data
def get_sector(ticker):
    stock = yf.Ticker(ticker)
    return stock.info.get('sector', 'Unknown')

@st.cache_data
def get_sector_tickers(sector):
    all_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sector_tickers = all_tickers[all_tickers['GICS Sector'] == sector]['Symbol'].tolist()
    return sector_tickers[:5]  # Limit to 5 tickers for performance

def analyze_sector_sentiment(sector):
    sector_tickers = get_sector_tickers(sector)
    sector_sentiment = []
    
    for ticker in sector_tickers:
        headlines = fetch_news(ticker, days=7)
        analyzed_headlines = analyze_sentiment(headlines, load_sentiment_analyzer())
        avg_sentiment = analyzed_headlines['sentiment_score'].mean()
        sector_sentiment.append(avg_sentiment)
    
    return np.mean(sector_sentiment)

def main():
    st.title("Stock Sentiment Analysis Dashboard")

    input_text = st.text_input("Enter a stock ticker or company name:", "AAPL")
    
    if st.button("Analyze"):
        try:
            ticker = get_ticker(input_text)
            st.write(f"Analyzing news and price data for: {ticker}")
            
            with st.spinner("Fetching historical data..."):
                hist_data, info = get_historical_data(ticker)
                
                if not hist_data.empty:
                    st.subheader("Price Chart and Fundamental Data")
                    fig = plot_line_chart(hist_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    current_price = hist_data['Close'].iloc[-1]
                    st.write(f"Current Closing Price: ${current_price:.2f}")
                    
                    display_fundamental_data(info)
                else:
                    st.warning(f"No historical data available for {ticker}")
            
            with st.spinner("Fetching news and analyzing sentiment..."):
                sentiment_analyzer = load_sentiment_analyzer()

                headlines = fetch_news(ticker)
                analyzed_headlines = analyze_sentiment(headlines, sentiment_analyzer)
                sentiment_summary, avg_sentiment, positive_ratio, negative_ratio, neutral_ratio = summarize_sentiment(analyzed_headlines)
                
                st.subheader(f"Sentiment Analysis for {ticker}")
                st.write(f"Overall Sentiment: {sentiment_summary}")
                st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
                st.write(f"Positive Headlines: {positive_ratio:.2%}")
                st.write(f"Negative Headlines: {negative_ratio:.2%}")
                st.write(f"Neutral Headlines: {neutral_ratio:.2%}")
                
                st.subheader("Top 5 Most Relevant Headlines and Their Sentiment")
                for _, row in analyzed_headlines.head(5).iterrows():
                    st.write(f"[{row['datetime']}] {row['title']}")
                    st.write(f"Source: {row['source']}")
                    st.write(f"Sentiment: {row['sentiment_label']} (Score: {row['sentiment_score']:.2f})")
                    st.write(f"[Read more]({row['url']})")
                    st.write("---")
                
                st.subheader("AI-Generated Summary")
                summary = generate_summary(analyzed_headlines, ticker, sentiment_summary, avg_sentiment)
                st.write(summary)
                
                # Sector Analysis
                sector = get_sector(ticker)
                st.subheader(f"Sector Analysis: {sector}")
                sector_sentiment = analyze_sector_sentiment(sector)
                st.write(f"Average sector sentiment: {sector_sentiment:.2f}")
                st.write(f"Stock sentiment compared to sector: {'Above' if avg_sentiment > sector_sentiment else 'Below'} average")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again or contact support if the problem persists.")

if __name__ == "__main__":
    main()