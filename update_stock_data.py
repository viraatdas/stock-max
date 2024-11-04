import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional
import logging
from env import NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import re
import pandas as pd
import yfinance as yf
import logging
from typing import List, Set
import praw  # Python Reddit API Wrapper
import requests

class TickerFetcher:
    """Handles fetching and managing lists of tickers"""
    
    @staticmethod
    def fetch_market_tickers() -> List[str]:
        """
        Fetch tickers that are currently being mentioned on Reddit's WallStreetBets
        and in recent news articles, indicating potential volatility.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting market ticker fetch from Reddit and News")

        # tickers_from_reddit = TickerFetcher.fetch_tickers_from_reddit()
        tickers_from_news = TickerFetcher.fetch_tickers_from_news()

        combined_tickers = list(set(tickers_from_news + tickers_from_news))
        logger.info(f"Fetched {len(combined_tickers)} unique tickers from Reddit and News")
        return combined_tickers

    @staticmethod
    def fetch_tickers_from_reddit() -> List[str]:
        """Fetch tickers mentioned in recent posts on r/WallStreetBets"""
        logger = logging.getLogger(__name__)
        logger.info("Fetching tickers from Reddit's r/WallStreetBets")

        # Initialize Reddit API client
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        subreddit = reddit.subreddit('wallstreetbets')
        hot_posts = subreddit.hot(limit=100)

        potential_tickers = set()
        ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')  # Pattern for tickers (1-5 uppercase letters)

        for post in hot_posts:
            tickers_in_title = ticker_pattern.findall(post.title)
            tickers_in_selftext = ticker_pattern.findall(post.selftext)
            all_tickers = tickers_in_title + tickers_in_selftext

            for ticker in all_tickers:
                if TickerFetcher.is_valid_ticker(ticker):
                    potential_tickers.add(ticker)

        logger.info(f"Found {len(potential_tickers)} tickers on Reddit")
        return list(potential_tickers)

    @staticmethod
    def fetch_tickers_from_news() -> List[str]:
        """Fetch tickers mentioned in recent financial news articles"""
        logger = logging.getLogger(__name__)
        logger.info("Fetching tickers from recent news articles")

        url = ('https://newsapi.org/v2/everything?'
               'q=stocks OR shares OR market&'
               'language=en&'
               'sortBy=publishedAt&'
               f'apiKey={NEWS_API_KEY}')

        response = requests.get(url)
        articles = response.json().get('articles', [])
        print(articles)

        potential_tickers = set()
        ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')  # Pattern for tickers (1-5 uppercase letters)

        for article in articles:
            content = article.get('title', '') + ' ' + article.get('description', '')
            tickers_in_content = ticker_pattern.findall(content)

            for ticker in tickers_in_content:
                if TickerFetcher.is_valid_ticker(ticker):
                    potential_tickers.add(ticker)

        logger.info(f"Found {len(potential_tickers)} tickers in news articles")
        return list(potential_tickers)

    @staticmethod
    def is_valid_ticker(ticker: str) -> bool:
        """Check if the ticker is valid using yfinance"""
        try:
            ticker_info = yf.Ticker(ticker).info
            return 'regularMarketPrice' in ticker_info
        except Exception:
            return False

class StockDataCollector:
    """Handles collecting stock data and sentiment analysis"""
    
    @staticmethod
    def get_stock_price(ticker: str) -> Optional[float]:
        """Fetch current stock price"""
        print(f"\nFetching price for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period='1d')['Close'].iloc[-1]
            return price
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            return None
    
    @staticmethod
    def calculate_sentiment(ticker: str) -> Optional[float]:
        """Calculate sentiment score"""
        try:
            sentiment = np.random.normal(0.5, 0.2)
            return sentiment
        except Exception as e:
            logger.error(f"Error calculating sentiment for {ticker}: {e}")
            return None

class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def update_ticker_data(self, ticker_data: List[Dict]) -> bool:
        """Update database with new ticker data"""
        try:
            conn = sqlite3.connect(self.db_path)
            new_df = pd.DataFrame(ticker_data)
            
            if len(new_df) > 0:
                today = datetime.now().date()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM stock_sentiment WHERE date = ?", 
                             (today.strftime('%Y-%m-%d'),))
                new_df.to_sql('stock_sentiment', conn, if_exists='append', index=False)
            
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database update error: {e}")
            return False

def update_stock_data(db_path: str) -> None:
    """Main function to update stock data"""
    logger.info("Starting Stock Data Update")
    # Initialize components
    ticker_fetcher = TickerFetcher()
    stock_collector = StockDataCollector()
    db_manager = DatabaseManager(db_path)
    
    # Get tickers
    logger.info("Fetching market tickers")
    market_tickers = ticker_fetcher.fetch_market_tickers()
    db_tickers = ticker_fetcher.get_db_tickers(db_path)
    all_tickers = ticker_fetcher.combine_tickers(market_tickers, db_tickers)
    
    # Collect data for each ticker
    logger.info("Collecting data for tickers")
    today = datetime.now().date()
    new_data = []
    
    for ticker in all_tickers:
        logger.info(f"Processing {ticker}")
        price = stock_collector.get_stock_price(ticker)
        sentiment = stock_collector.calculate_sentiment(ticker)
        
        if price is not None and sentiment is not None:
            new_data.append({
                'ticker': ticker,
                'date': today,
                'sentiment_score': sentiment,
                'closing_price': price
            })
            print(f"Added data for {ticker}")
    
    # Update database
    logger.info("Updating database")
    success = db_manager.update_ticker_data(new_data)
    logger.info(f"Updated {len(new_data)} tickers. Success: {success}")

if __name__ == "__main__":
    logger.info("Starting script")
    STOCK_DATA_DB = 'stock_data_mock.db'
    update_stock_data(STOCK_DATA_DB)
    logger.info("Script complete")