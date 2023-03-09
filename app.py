import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
from flask import Flask, render_template, request,redirect
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


def findData(tiker="TSLA",startDate='01/01/20',Interval= "1wk"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = "03/03/23", interval = Interval)
  
    except:
        
        return "Error"
    return data






def roundNummber(data):
        # round nummbers for better view
    data["open"] = list(np.around(np.array(data["open"]),2))
    data["low"] = list(np.around(np.array(data["low"]),2))
    data["high"] = list(np.around(np.array(data["high"]),2))
    data["adjclose"] = list(np.around(np.array(data["adjclose"]),2))
    return data

def LinearnaAgregacija(data):

    datumi = []
    for x in data["open"].index.values:
        datumi.append(str(x)[:-19])
    
    int_list = data["open"]

    # Date List
    date_list = datumi
    date_list = [datetime.strptime(date, '%Y-%m-%d') for date in date_list]
   
    # Create a DataFrame with the int and date lists
    df = pd.DataFrame({'int': int_list, 'date': date_list})
    df['date'] = (df['date'] - df['date'].min())  / np.timedelta64(1,'D')

    # Create a Linear Regression Model
    model = LinearRegression()
    model.fit(df[['date']], df['int'])
    return model.predict(df[['date']]) , df['int']


def logistic_regression(x, y):          # test
    # Convert the date values to numeric values
    y = pd.to_datetime(y)
    y = pd.to_numeric(y)

    # Create a logistic regression model
    model = LogisticRegression()

    # Fit the model to the data
    model.fit(np.array(x).reshape(-1, 1), y)

    # Get the coefficients of the model
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]


    # plot

    return coef, intercept

def logistic_reg ():
    x = np.arange(10).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression(solver='liblinear', random_state=0)

    model.fit(x, y)


    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

    model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
    model.predict_proba(x)

    model.predict(x)
    model.score(x, y)
    print(classification_report(y, model.predict(x)))

@app.route('/',methods = ['GET','POST'])
def main ():
    logistic_reg ()

        #default variables
    datumi =[]
    dionice = []

    # comppersion arrays
    dionice1 = []
    dionice2 = []
    tiker = "BRK-B"
    data = findData(tiker)

    # compersion tikers
    tiker1 = "TSLA"
    prvaDionica = findData(tiker1)

    tiker2 = "BRK-B"            
    drugaDionica = findData(tiker2)

    errormsg = ""

        # on post
    if request.method == "POST":
            # if post is search
        if 'search' in request.form:

                # change data with user input
            search = request.form['search']
            tiker= str(search.upper())

                # get get data based on user input
            data= findData(str(search))

                # if search cant be found
            try:
                data= findData(str(search))
                if data == "Error":
                    tiker ="Cant find that."
                print("Cant find that")
                # setting up tiker to default to prevent any kind of errors
                data = findData()
            except:
                print("correct")

        if 'primary' in request.form and 'secondary' in request.form :
            # getting first tiker from user input
            tiker1 = request.form['primary'].upper()
            prvaDionica = findData(tiker1)
            # getting second tiker from user input
            tiker2 = request.form['secondary'].upper()
            drugaDionica = findData(tiker2)
        

                

        #creating better format with data 
            #importing open tiker -> dionice
    for y in data["open"]:
        try:
            dionice.append(int(y))
        except:
            break
        

            #importing date -> datumi
    for x in data["open"].index.values:
        datumi.append(str(x)[:-19])
        
        # importing first Tiker for comperison
    try:
        for y in prvaDionica["open"]:
            dionice1.append(int(y))

            # importing second Tiker for comperison

        for y in drugaDionica["open"]:
            dionice2.append(int(y))
    except:
        errormsg = "error occured"
        
        # correlation done
    print(np.corrcoef(dionice1,dionice2))
   
    # getting linear agression
    x,y = LinearnaAgregacija(data)
    
    # round nummbers for better view
    roundNummber(data)

    x = [1, 2, 3, 4, 5]
    y = ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05']

    #coef, intercept = logistic_regression(dionice,datumi)

    #print("Coefficients: ", coef)
    #print("Intercept: ", intercept)
        
 
    print(findData())
    print(datumi)
    print(findData()["open"])

    #plot_regression_line(dionice,datumi, b)
    return render_template ('index.html',datumi=datumi, datumiLen = len(datumi),dionice=dionice,dioniceLen=len(dionice),data=data,imedionice = tiker,x=x,y=y,linearLen=len(x),dionice2=dionice2, tiker2= tiker2, tiker1 = tiker1, errormsg = errormsg ,dionice1 = dionice1)






    # basic Flask tamplate for testing
@app.route('/test',methods = ['GET','POST'])
def test ():


 
    #plot_regression_line(dionice,datumi, b)
    return str(45*3)




    # just keep it like this
if __name__ == "__main__":
    app.run(debug = True)
