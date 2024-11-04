import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import sqlite3

# Initialize Dash app
app = dash.Dash(__name__)

def get_stock_data():
    conn = sqlite3.connect('stock_data.db')
    query = "SELECT * FROM stock_sentiment"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_upcoming_catalysts():
    conn = sqlite3.connect('stock_data.db')
    query = "SELECT * FROM upcoming_catalysts"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Layout
app.layout = html.Div([
    html.H1("Stock Catalyst Tracker"),
    
    dcc.Dropdown(
        id='stock-selector',
        options=[{'label': ticker, 'value': ticker} for ticker in get_stock_data()['ticker'].unique()],
        value='AAPL'
    ),
    
    dcc.Graph(id='sentiment-price-chart'),
    
    html.H2("Upcoming Catalysts"),
    html.Div(id='catalyst-table')
])

@app.callback(
    [Output('sentiment-price-chart', 'figure'),
     Output('catalyst-table', 'children')],
    [Input('stock-selector', 'value')]
)
def update_charts(selected_ticker):
    # Get stock data
    df = get_stock_data()
    df_filtered = df[df['ticker'] == selected_ticker]
    
    # Create price and sentiment chart
    fig = px.line(df_filtered, x='date', y=['price', 'sentiment_score'], 
                  title=f'{selected_ticker} Price and Sentiment')
    
    # Get upcoming catalysts
    catalysts_df = get_upcoming_catalysts()
    catalysts_filtered = catalysts_df[catalysts_df['ticker'] == selected_ticker]
    
    # Create catalyst table
    catalyst_table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in ['Date', 'Type', 'Impact Score', 'Description']])),
        html.Tbody([
            html.Tr([
                html.Td(row['date']),
                html.Td(row['catalyst_type']),
                html.Td(f"{row['impact_score']:.2f}"),
                html.Td(row['description'])
            ]) for _, row in catalysts_filtered.iterrows()
        ])
    ])
    
    return fig, catalyst_table

if __name__ == '__main__':
    app.run_server(debug=True) 