
import numpy as np
import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data # adding stock 
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from datetime import datetime


def findData(tiker="TSLA",startDate='01/01/2018',Interval="1d"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = None, interval = Interval)
    except:
        
        return "Error"
    return data


data = findData()

datumi = []
for x in data["open"].index.values:
    datumi.append(str(x)[:-19])




## test 2

int_list = data["open"]

# Date List
date_list = datumi
date_list = [datetime.strptime(date, '%Y-%m-%d') for date in date_list]

# Create a DataFrame with the int and date lists
df = pd.DataFrame({'int': int_list, 'date': date_list})
df['date'] = (df['date'] - df['date'].min())  / np.timedelta64(1,'D')
print(df['date'][0])
print( model.predict(df[['date']])[0])
# Create a Linear Regression Model
model = LinearRegression()
model.fit(df[['date']], df['int'])
#print(df["date"])
print(model.predict(df[['date']]))

# Plot the linear regression graph
plt.scatter(df['date'], df['int'], color='red')
plt.plot(df['date'], model.predict(df[['date']]), color='blue')
plt.xlabel('Date')
plt.ylabel('Int')
plt.show()