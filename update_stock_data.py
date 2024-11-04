import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional
import logging

# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TickerFetcher:
    """Handles fetching and managing lists of tickers"""
    
    @staticmethod
    def fetch_market_tickers() -> List[str]:
        """Fetch tickers from market/internet source"""
        logger.info("Starting market ticker fetch")
        
        # Placeholder tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC']
        
        logger.info(f"Fetched {len(tickers)} market tickers")
        return tickers
    
    @staticmethod
    def get_db_tickers(db_path: str) -> List[str]:
        """Get existing tickers from database"""
        print(f"Fetching existing tickers from {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            existing_df = pd.read_sql("SELECT DISTINCT ticker FROM stock_sentiment", conn)
            conn.close()
            
            tickers = existing_df['ticker'].tolist()
            logger.info(f"Found {len(tickers)} existing tickers in DB: {tickers}")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching DB tickers: {e}")
            return []
    
    @staticmethod
    def combine_tickers(market_tickers: List[str], db_tickers: List[str]) -> List[str]:
        """Combine and deduplicate tickers"""
        combined = list(set(market_tickers + db_tickers))
        print(f"Combined {len(combined)} unique tickers: {combined}")
        return combined

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

def update_stock_data(db_path: str = 'stock_data_mock.db') -> None:
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
    update_stock_data()
    logger.info("Script complete")