import praw
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import logging
from groq import Groq
import json
from env import (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, 
                REDDIT_USER_AGENT, GROQ_API_KEY)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalDataCreator:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
    def get_stock_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical stock data and ensure we have valid trading days"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.error(f"No data found for {ticker}")
                return pd.DataFrame()
                
            # Keep only trading days with valid prices
            df = df[df['Close'].notna()]
            logger.info(f"Found {len(df)} trading days for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def get_reddit_sentiment(self, ticker: str, date: datetime) -> tuple:
        """Get Reddit posts and calculate sentiment for a specific date"""
        try:
            def to_timestamp(dt):
                """Convert datetime to Unix timestamp"""
                return int(datetime.combine(dt.date(), datetime.min.time()).timestamp())

            # Get start and end timestamps for the specific day
            start_ts = to_timestamp(date)
            end_ts = to_timestamp(date + timedelta(days=1)) - 1
            
            posts_text = []
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            
            logger.info(f"Searching Reddit for {ticker} between {datetime.fromtimestamp(start_ts)} and {datetime.fromtimestamp(end_ts)}")
            
            for sub in subreddits:
                try:
                    subreddit = self.reddit.subreddit(sub)
                    
                    # Try different search variations
                    search_variations = [
                        f'timestamp:{start_ts}..{end_ts} {ticker}',
                    ]
                    
                    for query in search_variations:
                        logger.info(f"Searching r/{sub} with query: {query}")
                        posts = list(subreddit.search(query, sort='new', limit=5))
                        print(posts)
                        
                        for post in posts:
                            post_time = datetime.fromtimestamp(post.created_utc)
                            # Verify post is within our time window
                            if start_ts <= post.created_utc <= end_ts:
                                posts_text.append(
                                    f"[{post_time.strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"Subreddit: r/{sub}\n"
                                    f"Title: {post.title}\n"
                                    f"Content: {post.selftext[:300]}..."
                                )
                                logger.info(f"Found post in r/{sub}: {post.title}")
                    
                except Exception as e:
                    logger.warning(f"Error searching {sub}: {e}")
                    continue
            
            if not posts_text:
                logger.warning(f"No Reddit posts found for {ticker} on {date.date()}")
                return 0.0, "No Reddit posts found for sentiment analysis"
            
            # Log found posts
            logger.info(f"Found {len(posts_text)} posts for {ticker} on {date.date()}")
            
            # Calculate sentiment
            news_text = "\n\n".join(posts_text)
            prompt = f"""
            You are an expert stock market analyst. Analyze the sentiment for {ticker} based on these Reddit posts from {date.date()}:

            {news_text}

            Return ONLY a JSON with:
            - sentimentScore: number between -1.0 (very negative) and 1.0 (very positive)
            - reasoning: brief explanation of the score

            Example:
            {{"sentimentScore": 0.5, "reasoning": "Positive outlook due to strong earnings"}}
            """
            
            completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.1
            )
            
            response = completion.choices[0].message.content.strip()
            result = json.loads(response)
            
            sentiment = max(-1.0, min(1.0, float(result['sentimentScore'])))
            reasoning = result['reasoning']
            
            logger.info(f"Calculated sentiment for {ticker}: {sentiment}")
            return sentiment, reasoning
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0, f"Error calculating sentiment: {str(e)}"
def create_historical_data():
    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    db_path = 'stock_data_historical.db'
    
    logger.info(f"Creating historical data from {start_date.date()} to {end_date.date()}")
    
    # Initialize
    creator = HistoricalDataCreator()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Setup database
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_sentiment (
            ticker TEXT,
            date DATE,
            sentimentScore REAL,
            reasoning TEXT,
            closingPrice REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    cursor.execute("DELETE FROM stock_sentiment")  # Clear existing data
    
    # Process each ticker
    for ticker in tickers:
        logger.info(f"\nProcessing {ticker}")
        
        # Get stock data
        stock_data = creator.get_stock_data(ticker, start_date, end_date)
        if stock_data.empty:
            continue
            
        # Process each trading day
        for date, row in stock_data.iterrows():
            trading_date = date.date()
            closing_price = row['Close']
            
            logger.info(f"\nAnalyzing {ticker} for {trading_date}")
            logger.info(f"Closing price: ${closing_price:.2f}")
            
            # Get sentiment
            sentiment, reasoning = creator.get_reddit_sentiment(ticker, date)
            
            # Save to database
            cursor.execute("""
                INSERT INTO stock_sentiment 
                (ticker, date, sentimentScore, reasoning, closingPrice)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ticker,
                trading_date,
                sentiment,
                reasoning,
                closing_price
            ))
            
            logger.info(f"Saved: Sentiment={sentiment:.2f}, Price=${closing_price:.2f}")
            conn.commit()  # Commit after each record
    
    conn.close()
    logger.info("\nHistorical data creation complete")

if __name__ == "__main__":
    create_historical_data()