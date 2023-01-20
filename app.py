import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import datetime as dt
import pandas_datareader as web






data = yf.download(tickers='TSLA', period='3d', interval='1m')
print(data)
fig = go.Figure()


#Candlestick
fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name = 'market data'))

#Add titles
fig.update_layout(
    title='Uber live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

#X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()


data = yf.download(tickers='TSLA', period='1d', interval='1m')
fig = go.Figure()

