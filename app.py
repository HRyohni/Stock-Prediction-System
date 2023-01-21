import yfinance as yf
import pandas as pd
import numpy as np
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
msft = yf.Ticker("TSLA") # for getting aditional data and stuff



data = get_data("TSLA", start_date = None, end_date = None, index_as_date = True, interval = "1d")
print(data["open"])





# Graph
dataframe = pd.read_excel("output.xlsx")
x = 100
y = 50
plt.scatter(x, y)
plt.show()  # or plt.savefig("name.png")
    #exports data
#df1 = pd.DataFrame(data)
#df1.to_excel("output.xlsx") 

#for x in msft.info.keys():
#    print(str(x)+":  "+ str(msft.info.get(x)))

