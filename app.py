import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlite3
from threading import Thread
import time

from update_stock_data import update_stock_data

USE_MOCK_DATA = False
UPDATE_STOCK_DATA = True

if USE_MOCK_DATA:
    STOCK_DATA_DB = 'stock_data_mock.db'
else:
    STOCK_DATA_DB = 'stock_data.db'

app = dash.Dash(__name__)
app.scripts.config.serve_locally = True

def get_stock_data():
    conn = sqlite3.connect(STOCK_DATA_DB)
    print(f"Connected to {STOCK_DATA_DB}")
    query = "SELECT ticker, date, sentimentScore, reasoning, closingPrice FROM stock_sentiment"
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date']).dt.date
    conn.close()
    return df

def get_latest_sentiments():
    df = get_stock_data()
    latest = df.sort_values('date').groupby('ticker').last()
    return latest.sort_values('sentimentScore', ascending=False)

def format_sentiment(ticker, score):
    """Format sentiment with +/- sign"""
    sign = '+' if score > 0 else ''
    return f"{ticker:4} │ {sign} {score:.3f}"

app.layout = html.Div([
    html.H1("Stock Sentiment Tracker"),

    html.Div([
        html.Label("Select Stock (sorted by sentiment score):", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='stock-selector',
            options=[],  # Start empty, will be populated by callback
            value=None,  # No initial selection
            style={'width': '400px'}
        ),
    ], style={'margin': '20px 0'}),
    
    dcc.Graph(id='sentiment-price-chart'),
    
    # Add a div for displaying the reasoning
    html.Div([
        html.H3("Latest Analysis"),
        html.Div(id='reasoning-display', style={
            'padding': '15px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'marginTop': '10px'
        })
    ])
])

@app.callback(
    [Output('sentiment-price-chart', 'figure'),
     Output('reasoning-display', 'children'),
     Output('stock-selector', 'options')],
    [Input('stock-selector', 'value')]
)
def update_chart(selected_ticker):
    try:
        # Get stock data
        df = get_stock_data()
        latest_data = get_latest_sentiments()

        # Update dropdown options
        dropdown_options = [
            {'label': format_sentiment(ticker, latest_data.loc[ticker, 'sentimentScore']), 
             'value': ticker} 
            for ticker in latest_data.index
        ]

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig, "No data available", dropdown_options  
            
            
        if not selected_ticker and dropdown_options:
            selected_ticker = dropdown_options[0]['value']
        elif not selected_ticker:
            fig = go.Figure()
            fig.add_annotation(text="Please select a ticker", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig, "Please select a ticker", dropdown_options

        df_filtered = df[df['ticker'] == selected_ticker]
        if df_filtered.empty:
            # Return empty figure if no data for selected ticker
            fig = go.Figure()
            fig.add_annotation(text=f"No data available for {selected_ticker}", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig, f"No data available for {selected_ticker}", dropdown_options

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['closingPrice'],
                name="Price ($)",
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        # Add sentiment score line
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['sentimentScore'],
                name="Sentiment Score",
                line=dict(color='red'),
                hovertemplate=(
                    "<b>Date:</b> %{x|%Y-%m-%d}<br>" +
                    "<b>Sentiment:</b> %{y:.3f}<br>" +
                    "<extra></extra>"
                ),
                text=df_filtered['reasoning']
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f'{selected_ticker} Price vs Sentiment',
            xaxis_title="Date",
            xaxis=dict(
                tickformat="%Y-%m-%d",
                type="date"
            ),
            hovermode='x unified',
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score (-1 to 1)", secondary_y=True)
        
        try:
            # Get latest reasoning for display
            latest_data = df_filtered.sort_values('date').iloc[-1]
            latest_reasoning = html.Div([
                html.P([
                    html.Strong("Date: "), 
                    html.Span(latest_data['date'].strftime('%Y-%m-%d'))
                ]),
                html.P([
                    html.Strong("Sentiment Score: "), 
                    html.Span(f"{latest_data['sentimentScore']:.3f}")
                ]),
                html.P([
                    html.Strong("Analysis: "), 
                    html.Span(latest_data['reasoning'])
                ])
            ])
        except IndexError:
            latest_reasoning = "No recent data available"
        
        return fig, latest_reasoning, dropdown_options
        
    except Exception as e:
        print(f"Error in callback: {str(e)}")
        # Return empty figure and error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig, f"Error: {str(e)}", []

if __name__ == '__main__':
    if USE_MOCK_DATA == False and UPDATE_STOCK_DATA:
        update_stock_data(STOCK_DATA_DB)
    
    app.run_server(debug=True)
