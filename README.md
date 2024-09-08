# Stock News Sentiment Analysis Dashboard

## Overview

This project is a web-based dashboard that provides comprehensive stock analysis by combining financial data, news sentiment, and sector comparisons. It uses natural language processing and machine learning techniques to analyze recent news articles and generate insights about a given stock.

## Example of how the end product will end up looking upon completion.

https://github.com/user-attachments/assets/00b2cf74-215c-4abf-a384-7fd86c54e1da

## Features

- Stock Information Retrieval: Accepts both stock tickers and common company names as input
- Price Visualization: Displays a 1-year historical price chart and current closing price
- Fundamental Data Display: Shows key financial metrics (e.g., market cap, P/E ratio, dividend yield)
- News Sentiment Analysis: Analyzes sentiment of recent headlines using a fine-tuned FinBERT model
- AI-Generated Summary: Provides a concise summary of recent news and overall sentiment using BART
- Sector Analysis: Compares the stock's sentiment to the sector average
- User-Friendly Interface: Built with Streamlit for an interactive web application

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/m-turnergane/stock-sentiment-dashboard.git
   cd stock-sentiment-dashboard
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add your NewsAPI key:
   ```
   NEWS_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run news_sentiment.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501)

3. Enter a stock ticker or company name and click "Analyze" to view the dashboard

## Project Structure

stock-sentiment-dashboard/
│
├── venv/
├── .env
├── .gitignore
├── news_sentiment.py
├── requirements.txt
└── README.md

## Dependencies

- streamlit
- yfinance
- plotly
- pandas
- numpy
- requests
- transformers
- python-dotenv

## Contributing

Contributions to improve the dashboard are welcome. Please feel free to submit a Pull Request.

## Acknowledgments

- NewsAPI for providing access to news articles
- yfinance for stock market data
- Hugging Face for access to open-source pre-trained NLP models

## Future Improvements

- Implement user authentication for personalized experiences
- Add historical sentiment trends over time
- Integrate more data sources for a more comprehensive analysis
- Optimize performance for faster analysis of multiple stocks
- Add export functionality for reports

## Troubleshooting

If you encounter any issues, please check the following:
1. Ensure all dependencies are correctly installed
2. Verify that your NewsAPI key is valid and correctly set in the .env file
3. Check your internet connection, as the app requires access to external APIs

For any persistent problems, please open an issue on the GitHub repository.
