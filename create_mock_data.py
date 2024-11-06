import sqlite3
import pandas as pd
import numpy as np
import os

# Delete existing database if it exists
STOCK_DATA_DB = 'stock_data_mock.db'
if os.path.exists(STOCK_DATA_DB):
    os.remove(STOCK_DATA_DB)

# Create connection to SQLite database
conn = sqlite3.connect(STOCK_DATA_DB)

# Generate mock stock sentiment data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC']
dates = pd.date_range(start='2024-01-01', end='2024-03-15', freq='D')

data = []
for ticker in tickers:
    for date in dates:
        sentiment_score = np.random.normal(0.5, 0.2)  # Random sentiment between -1 and 1
        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp between -1 and 1
        price = np.random.uniform(100, 500)
        
        data.append({
            'ticker': ticker,
            'date': date,
            'sentimentScore': sentiment_score,
            'reasoning': f'Mock analysis for {ticker}: The stock shows {"positive" if sentiment_score > 0 else "negative"} sentiment based on market trends and technical analysis.',
            'closingPrice': price
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to SQLite with correct table name
df.to_sql('stock_sentiment', conn, if_exists='replace', index=False)

conn.close()

print("Mock data has been created successfully!")