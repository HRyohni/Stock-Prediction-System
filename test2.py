import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import statsmodels.api as sm
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
from flask import Flask, render_template, request,redirect
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler


def findData(tiker="TSLA",startDate='01/01/18',Interval= "1wk",end_date="03/03/23"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = end_date, interval = Interval)
  
    except:
        
        return "Error"
    return data

def Simulation(UserStartDate,UserEndDate,priceAmount,tikerName):
    data = findData(tiker=tikerName,startDate=UserStartDate, end_date=UserEndDate)

    startPrice = data.loc[UserStartDate][1] # get stock by the date
    endPrice = data.loc[UserEndDate][1]

    print(str((startPrice / endPrice) * 100)+"%")
    return str((startPrice / endPrice) * 100)+"%"

Simulation(UserStartDate= "2018-01-01",UserEndDate= "2019-04-22",priceAmount= 23, tikerName="AAPL")
