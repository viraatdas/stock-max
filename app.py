import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlite3

from update_stock_data import update_stock_data

STOCK_DATA_DB = 'stock_data.db'
update_stock_data(STOCK_DATA_DB)

def get_stock_data():
    conn = sqlite3.connect(STOCK_DATA_DB)
    print(f"Connected to {STOCK_DATA_DB}")
    query = "SELECT ticker, date, sentiment_score, closing_price FROM stock_sentiment"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_latest_sentiments():
    df = get_stock_data()
    # Get the latest entry for each ticker
    latest = df.sort_values('date').groupby('ticker').last()
    # Sort by sentiment score in descending order
    return latest.sort_values('sentiment_score', ascending=False)

app = dash.Dash(__name__)

# Get latest sentiment scores for dropdown and sort by sentiment
latest_data = get_latest_sentiments()
def format_sentiment(ticker, score):
    """Format sentiment with color and +/- sign"""
    sign = '+' if score > 0 else ''
    return f"{ticker:4} │ {sign} {score:.3f}"

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True

# Get latest sentiment scores for dropdown
latest_data = get_latest_sentiments()
dropdown_options = [
    {'label': format_sentiment(ticker, latest_data.loc[ticker, 'sentiment_score']), 
     'value': ticker} 
    for ticker in latest_data.index
]

app.layout = html.Div([
    html.H1("Stock Catalyst Tracker"),
    
    html.Div([
        html.Label("Select Stock (sorted by sentiment score):", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='stock-selector',
            options=dropdown_options,
            value=dropdown_options[0]['value'],
            style={'width': '400px'}
        ),
    ], style={'margin': '20px 0'}),
    
    dcc.Graph(id='sentiment-price-chart'),
])

@app.callback(
    Output('sentiment-price-chart', 'figure'),
    [Input('stock-selector', 'value')]
)
def update_chart(selected_ticker):
    # Get stock data
    df = get_stock_data()
    df_filtered = df[df['ticker'] == selected_ticker]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['closing_price'],
            name="Price ($)",
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add sentiment score line on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['sentiment_score'],
            name="Sentiment Score",
            line=dict(color='red')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f'{selected_ticker} Price vs Sentiment',
        xaxis_title="Date",
        hovermode='x unified',
        showlegend=True
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score (-1 to 1)", secondary_y=True)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)