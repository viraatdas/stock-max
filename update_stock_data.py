import json
import sqlite3
from groq import Groq
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Optional
import logging
from env import FMP_API_KEY, FMP_BASE_URL, GROQ_API_KEY, NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_PASSWORD, REDDIT_USER_AGENT, REDDIT_USERNAME

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
import praw
import requests

class TickerFetcher:
    """Handles fetching and managing lists of tickers"""

    # Common words and invalid patterns to filter out
    COMMON_WORDS = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'THE', 'AND', 'FOR', 'NEW', 'CEO', 'CFO', 'IPO', 'USA', 'GDP',
        'AI', 'API', 'SEC', 'FDA', 'NYSE', 'ETF', 'ESG', 'CEO', 'CTO',
        'RSS', 'HTTP', 'HTML', 'XML', 'JSON', 'USD', 'EUR', 'GBP',
        'Q1', 'Q2', 'Q3', 'Q4', 'YTD', 'TTM', 'EPS', 'PE', 'ROI',
        'EBIT', 'GAAP', 'IMO', 'TBH', 'FOMO', 'YOLO', 'HODL',
        'DD', 'PR', 'IR', 'CEO', 'COO', 'CTO', 'CFO', 'CMO',
        'BOND', 'CASH', 'LOAN', 'DEBT', 'GAIN', 'LOSS'
    }

    @staticmethod
    def clean_ticker(ticker: str) -> str:
        """
        Clean and validate a ticker symbol
        Returns empty string if invalid
        """
        try:
            # Remove common prefixes/suffixes and whitespace
            cleaned = ticker.strip('$£€¥').strip()
            cleaned = cleaned.split('.')[0]  # Remove exchange suffixes like .L, .DE
            cleaned = cleaned.split(':')[0]  # Remove exchange prefixes like NYSE:
            cleaned = cleaned.split('-')[0]  # Remove share class suffixes
            
            cleaned = cleaned.strip('$').strip()
            
            # Basic validation
            if not cleaned:
                return ''
            if cleaned in TickerFetcher.COMMON_WORDS:
                return ''
            if len(cleaned) > 5:  # Most tickers are 1-5 characters
                return ''
            if not cleaned.isalpha():  # Should only contain letters
                return ''
            if not cleaned.isupper():  # Should be all uppercase
                return ''
                
            return cleaned
            
        except Exception as e:
            logger.debug(f"Error cleaning ticker {ticker}: {e}")
            return ''

    @staticmethod
    def get_db_tickers(db_path: str) -> List[str]:
        """Get existing tickers from database"""
        try:
            conn = sqlite3.connect(db_path)
            query = "SELECT DISTINCT ticker FROM stock_sentiment"
            existing_df = pd.read_sql(query, conn)
            conn.close()
            
            db_tickers = existing_df['ticker'].tolist()
            logger.info(f"Found {len(db_tickers)} existing tickers in database")
            return db_tickers
            
        except Exception as e:
            logger.error(f"Error fetching DB tickers: {e}")
            return []
            
    # Also add this method since it's called but missing
    @staticmethod
    def combine_tickers(market_tickers: List[str], db_tickers: List[str]) -> List[str]:
        """Combine and deduplicate tickers from different sources"""
        combined = list(set(market_tickers + db_tickers))
        logger.info(f"Combined {len(combined)} unique tickers from all sources")
        return combined
    
    @staticmethod
    def fetch_market_tickers() -> List[str]:
        """
        Fetch tickers that are currently being mentioned on Reddit's WallStreetBets
        and in recent news articles, indicating potential volatility.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting market ticker fetch from Reddit and News")

        tickers_from_reddit = TickerFetcher.fetch_tickers_from_reddit()
        # tickers_from_news = TickerFetcher.fetch_tickers_from_news()

        combined_tickers = list(set(tickers_from_reddit))
        logger.info(f"Combined tickers: {combined_tickers}")
        logger.info(f"Fetched {len(combined_tickers)} unique tickers from Reddit and News")
        return combined_tickers

    @staticmethod
    def fetch_tickers_from_reddit() -> List[str]:
        """Fetch tickers mentioned in recent posts from multiple investing subreddits"""
        logger = logging.getLogger(__name__)
        logger.info("Fetching tickers from Reddit investing subreddits")

        # Initialize Reddit API client
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        subreddits = ['wallstreetbets', 'smallstreetbets', 
                    'pennystocks', 'spacs', 
                    'dividends', 'biotechstocks',
                    'investing', 'stocks']

        potential_tickers = set()
        ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')  # Pattern for tickers (1-5 uppercase letters)

        all_found_tickers = set()
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                hot_posts = subreddit.hot(limit=50)  # Reduced limit per subreddit

                for post in hot_posts:
                    # Clean tickers as we find them
                    tickers_in_title = [
                        t for t in ticker_pattern.findall(post.title)
                        if TickerFetcher.clean_ticker(t)
                    ]
                    tickers_in_selftext = [
                        t for t in ticker_pattern.findall(post.selftext)
                        if TickerFetcher.clean_ticker(t)
                    ]
                    all_tickers = tickers_in_title + tickers_in_selftext
                    all_found_tickers.update(all_tickers)

            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {str(e)}")
                continue

        if all_found_tickers:  # Only make API call if any tickers were found
            valid_tickers = TickerFetcher.are_valid_tickers(list(all_found_tickers))
            potential_tickers.update(valid_tickers.keys())

        logger.info(f"Found {len(potential_tickers)} unique tickers across {len(subreddits)} subreddits")
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
            valid_tickers = TickerFetcher.are_valid_tickers(tickers_in_content)
            for ticker in valid_tickers:
                potential_tickers.add(ticker)

        logger.info(f"Found {len(potential_tickers)} tickers in news articles")
        return list(potential_tickers)

    @staticmethod
    def are_valid_tickers(tickers: List[str]) -> Dict[str, dict]:
        """Check if multiple stock tickers are valid."""
        # Clean and filter tickers
        cleaned_tickers = []
        for t in tickers:
            clean_t = TickerFetcher.clean_ticker(t)
            if clean_t:  # Only include non-empty cleaned tickers
                cleaned_tickers.append(clean_t)
        
        if not cleaned_tickers:
            return {}
            
        # Remove duplicates while preserving order
        cleaned_tickers = list(dict.fromkeys(cleaned_tickers))
        
        # Log what we're about to validate
        logger.info(f"Validating tickers: {cleaned_tickers}")
        
        # Make API call
        tickers_str = ','.join(cleaned_tickers)
        url = f"{FMP_BASE_URL}/quote/{tickers_str}?apikey={FMP_API_KEY}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            
            # Additional validation of response data
            valid_tickers = {}
            for stock in data:
                symbol = stock.get('symbol', '')
                if symbol and isinstance(stock.get('price'), (int, float)):
                    valid_tickers[symbol] = stock
            
            logger.info(f"Found {len(valid_tickers)} valid tickers out of {len(cleaned_tickers)} tested")
            return valid_tickers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error validating tickers: {e}")
            return {}

class StockDataCollector:
    """Handles collecting stock data and sentiment analysis"""


    @staticmethod
    def get_stock_news(ticker: str) -> str:
        """Fetch recent news about the stock from Reddit"""
        try:
            # Initialize Reddit API client
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )

            # Subreddits relevant to stock discussion
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'pennystocks', 'smallstreetbets']
            news_text = ""
            
            for subreddit_name in subreddits:
                subreddit = reddit.subreddit(subreddit_name)
                hot_posts = subreddit.search(f"{ticker}", limit=5)  # Search posts with ticker mention
                
                for post in hot_posts:
                    news_text += f"Title: {post.title}\n"
                    news_text += f"Description: {post.selftext[:200]}...\n\n"  # Limit description for readability
                
                if news_text:
                    return news_text  # Return results if found in any subreddit

            return f"No recent news found for {ticker} on Reddit."

        except Exception as e:
            logger.error(f"Error fetching Reddit news for {ticker}: {e}")
            return f"Error fetching news for {ticker}"
    
    @staticmethod
    def get_stock_price(ticker: str) -> Optional[float]:
        """Fetch current stock price.
        Returns the most recent closing price, which is typically from the previous trading day
        since markets close at 4pm ET."""
        print(f"\nFetching price for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period='1d')['Close'].iloc[-1]
            print(f"Price: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            return None
    
    @staticmethod
    def calculate_sentiment(ticker: str) -> Optional[float]:
        """Calculate sentiment score using Groq LLM"""
        try:
            # Get news about the stock
            news = StockDataCollector.get_stock_news(ticker)
            
            # Create Groq client
            client = Groq(
                api_key=GROQ_API_KEY
            )
            
            # Construct the prompt
            prompt = f"""
            You are an expert stock market analyst who is tasked with analyzing the sentiment of a stock based on recent news.

            You are smart and understand how different news can have different impacts on a stock's sentiment. Think deeply about the news and how different trigger factors can impact sentiment.

            Based on the following news about {ticker}, analyze the overall sentiment and provide a single number between -1.0 (extremely negative) and +1.0 (extremely positive).
            
            News:
            {news}
            
            Rules:
            - Return ONLY a number between -1.0 and +1.0
            - -1.0 means extremely negative sentiment
            - 0.0 means neutral sentiment
            - +1.0 means extremely positive sentiment
            - Consider factors like financial performance, market reception, and future outlook focusing on the short term.
            
            Output your sentiment score as a json. This is the format of the json:
            {{
                "sentimentScore": <sentimentScore>,
                "reasoning": <explanation of how you came up with the sentiment score>
            }}
            """
            
            # Get completion from Groq
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="mixtral-8x7b-32768",  # Using Mixtral for better analysis
                temperature=0.1  # Low temperature for more consistent outputs
            )
            
            # Extract the sentiment score
            response = chat_completion.choices[0].message.content.strip()
            
            # Clean the response to get only the JSON part
            try:
                # Find the JSON object boundaries
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx == -1 or end_idx == 0:
                    logger.error(f"No JSON found in response: {response}")
                    return 0.0, "Error: Invalid response format"
                
                # Extract just the JSON part
                json_str = response[start_idx:end_idx]
                
                # Replace problematic characters
                json_str = ''.join(char for char in json_str if ord(char) >= 32)
                
                # Parse the JSON
                parsed_json = json.loads(json_str)
                sentiment = float(parsed_json['sentimentScore'])
                reasoning = parsed_json['reasoning']

                # Ensure sentiment is within bounds
                sentiment = max(-1.0, min(1.0, sentiment))
                
                logger.info(f"Parsed sentiment: {sentiment}")
                logger.info(f"Parsed reasoning: {reasoning}")
                
                return sentiment, reasoning
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error parsing response: {response}")
                logger.error(f"Error details: {str(e)}")
                return 0.0, f"Error parsing sentiment: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error calculating sentiment for {ticker}: {e}")
            return None

class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.create_tables()  
    
    def create_tables(self):
        """Create necessary database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create stock_sentiment table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_sentiment (
                    ticker TEXT,
                    date DATE,
                    sentimentScore REAL,
                    reasoning TEXT,
                    closingPrice REAL
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
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
    
    for ticker in all_tickers[:5]:
        logger.info(f"Processing {ticker}")
        price = stock_collector.get_stock_price(ticker)
        if price is None:
            logger.error(f"No price found for {ticker}")
            continue
        sentiment, reasoning = stock_collector.calculate_sentiment(ticker)
        
        if price is not None and sentiment is not None:
            new_data.append({
                'ticker': ticker,
                'date': today,
                'sentimentScore': sentiment,  
                'reasoning': reasoning,
                'closingPrice': price
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