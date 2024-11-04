import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlite3

def get_stock_data():
    conn = sqlite3.connect('stock_data.db')
    query = "SELECT ticker, date, sentiment_score, closing_price FROM stock_sentiment"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Catalyst Tracker"),
    
    dcc.Dropdown(
        id='stock-selector',
        options=[{'label': ticker, 'value': ticker} for ticker in get_stock_data()['ticker'].unique()],
        value='AAPL'
    ),
    
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