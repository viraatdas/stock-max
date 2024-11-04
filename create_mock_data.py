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
        volume = np.random.randint(1000000, 10000000)
        price = np.random.uniform(100, 500)
        catalyst_type = np.random.choice(['Earnings', 'FDA Approval', 'Product Launch', 'Analyst Update', 'None'])
        
        data.append({
            'ticker': ticker,
            'date': date,
            'sentiment_score': sentiment_score,
            'closing_price': price,
            'catalyst_type': catalyst_type
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to SQLite
df.to_sql('stock_sentiment', conn, if_exists='replace', index=False)

# Create mock upcoming catalysts
upcoming_catalysts = []
future_dates = pd.date_range(start='2024-03-16', end='2024-04-15', freq='D')

for ticker in tickers:
    num_events = np.random.randint(1, 4)
    event_dates = np.random.choice(future_dates, num_events, replace=False)
    
    for date in event_dates:
        catalyst_type = np.random.choice(['Earnings', 'FDA Approval', 'Product Launch', 'Analyst Update'])
        impact_score = np.random.uniform(0.1, 0.9)
        
        upcoming_catalysts.append({
            'ticker': ticker,
            'date': date,
            'catalyst_type': catalyst_type,
            'impact_score': impact_score,
            'description': f'Mock {catalyst_type} event for {ticker}'
        })

# Create DataFrame for upcoming catalysts
upcoming_df = pd.DataFrame(upcoming_catalysts)
upcoming_df.to_sql('upcoming_catalysts', conn, if_exists='replace', index=False)

conn.close()

print("Mock data has been created successfully!") 