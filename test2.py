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

def Simulation(UserStartDate,UserEndDate,amount,tikerName):
    data = findData(tiker=tikerName,startDate=UserStartDate, end_date="2023-01-01")
  
    print(data)
    startPrice = data.loc[UserStartDate][1] # get stock by the date
    endPrice = data.loc[UserEndDate][1]
    


    profit = endPrice - startPrice 
    
    
    return str(round(profit,2)*amount)+"$ bought with:"+ str(round(startPrice,2)*amount)

print("---->",Simulation(UserStartDate= "2013-01-01",UserEndDate= "2022-12-27",amount= 80, tikerName="meta"))
