
import numpy as np
import matplotlib.pyplot as plt
from yahoo_fin.stock_info import get_data # adding stock 
import pandas as pd
import statsmodels.formula.api as smf
import datetime as dt

from sklearn.linear_model import LinearRegression


def findData(tiker="TSLA",startDate='01/01/2018',Interval="1wk"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = None, interval = Interval)
    except:
        
        return "Error"
    return data
  
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)
  
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
  
    # predicted response vector
    y_pred = b[0] + b[1]*x
  
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
  
    # putting labels
    plt.xlabel('value') #x
    plt.ylabel('datum') #y
  
    # function to show plot
    plt.show()
  
#def main():
    # observations / data


   # x = np.array([])
   # y = np.array([])

   # # estimating coefficients
   # b = estimate_coef(x, y)
   # print("Estimated coefficients:\nb_0 = {}  \
   #         \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    #plot_regression_line(x, y, b)
  


# Load data into a pandas dataframe

# Generate example data
dates = [dt.datetime.strptime("2022-06-01", "%Y-%m-%d"), 
         dt.datetime.strptime("2022-06-02", "%Y-%m-%d"),
         dt.datetime.strptime("2022-06-03", "%Y-%m-%d"),
         dt.datetime.strptime("2022-06-04", "%Y-%m-%d"),
         dt.datetime.strptime("2022-06-05", "%Y-%m-%d")]
prices = [100, 120, 130, 140, 150]

# Convert dates to numerical values
x = [date.toordinal() for date in dates]
x = np.array(x).reshape(-1, 1)
print(x)
y = np.array(prices).reshape(-1, 1)

# Fit linear regression model
reg = np.polyfit(x.flatten(), y.flatten(), 1)

print("Slope (a): ", reg[0])
print("Intercept (b): ", reg[1])
# Predict y values
y_pred = reg[0] * x + reg[1]

# Plot the data and the regression line
plt.scatter(dates, prices, color='blue')
plt.plot(dates, y_pred, color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Linear Regression of Prices vs. Date')
plt.show()

print(findData())