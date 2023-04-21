import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import datetime as datetime

def correlation(start_date_str,end_date_str,symbols_list):
    # Prompt user for number of stocks to analyze and their symbols
  
    symbols_list = []
    temp = ["AAPL","AMZN","MSFT","GOOGL","meta","JPM","JNJ","V","NVDA","NFLX","DIS","BA","BAC","WMT","PG","XOM","TSLA","CSCO","KO","PFE","IBM","GE","CRM","AMD","GS","VZ","UNH","CVX","WFC","ORCL"]
    for a in temp:
        symbols_list.append(a) 

    # Retrieve stock data from Yahoo Finance using yfinance
    symbols = []
    for ticker in symbols_list:
        try:
            r = yf.download(ticker, start=start_date)
            # Add ticker symbol as a column
            r['Symbol'] = ticker 
            symbols.append(r)
        except Exception as e:
           print(f"Error retrieving data for symbol {ticker}: {e}")

    # Combine stock data into a single DataFrame
    df = pd.concat(symbols)
    df = df.reset_index()
    df = df[['Date', 'Close', 'Symbol']]

    # Pivot table to get data in the required format for correlation matrix
    df_pivot = df.pivot('Date', 'Symbol', 'Close').reset_index()

    # Calculate Spearman correlation matrix
    corr_df = df_pivot.corr(method='spearman')

    # Display correlation matrix as heatmap using Seaborn
    plt.figure(figsize=(20, 15))  # Specify the size of the figure
    heatmap = sns.heatmap(corr_df, cmap='coolwarm',linecolor="white", annot=True, fmt=".2f" ,annot_kws={"color": "white"}, cbar_kws={'orientation': 'vertical'})  # Customize the heatmap
    cbar = heatmap.collections[0].colorbar  # Get the colorbar
    cbar.ax.tick_params(labelcolor='white')  # Set the tick labels color to white
    cbar.set_label('Correlation', color='white')  # Set the color of the label to white
    plt.title('Spearman Correlation Matrix')
    plt.savefig('static/img/correlation.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)  # Save the figure as a PNG file with transparent background
    print("Correlation matrix saved as correlation.png")