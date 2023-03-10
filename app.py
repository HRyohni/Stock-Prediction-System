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
app = Flask(__name__)

from sklearn.model_selection import train_test_split
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


def perform_time_series_regression(cost_array, date_array):
    # Create a pandas dataframe with the cost and date arrays
    df = pd.DataFrame({'cost': cost_array, 'date': pd.to_datetime(date_array)})
    
    # Set the date as the index of the dataframe
    df = df.set_index('date')
    
    # Perform the time series regression using the statsmodels library
    model = sm.tsa.ARIMA(df['cost'], order=(1, 1, 1))
    results = model.fit()
    
    # Print the summary of the regression results
    print(results.summary())
    
    # Return the predicted values of the cost stock based on the regression model
    return results.predict(start=df.index[0], end=df.index[-1], dynamic=True)

def Polynomial_Regression(y):
  x=[]
  for i in range(len(y)):
      x.append(i+1)
  mymodel = np.poly1d(np.polyfit(x, y, 3))

  myline = np.linspace(1, len(y), len(y))
  print(">>>>>",myline)
  
  plt.scatter(x, y)
  plt.plot(myline, mymodel(myline))
  #plt.show()
  return mymodel(myline)

def logistic_regression(x, y):          # test
    X = []
    y = []
    # Loop through each date string and extract the corresponding data
    for date_string in y:
        # Convert date string to datetime object
        date = datetime.strptime(date_string, '%Y-%m-%d')
        # Extract data for the given date
        data = [d for d in x if d['date'] == date]
        if len(data) > 0:
            # Append data to X and y arrays
            X.append(data[0]['predictor'])
            y.append(data[0]['outcome'])

    # Convert X and y arrays to numpy arrays
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on test data and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy



@app.route('/',methods = ['GET','POST'])
def main ():
   

        #default variables
    datumi =[]
    dionice = []

    # comppersion arrays
    dionice1 = []
    dionice2 = []
    tiker = "BRK-B"
    data = findData(tiker)
    poly1= []
    poly2= []

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
    # gets polynomial regression
        # poly from first stock
    poly1 = Polynomial_Regression(dionice1)
    print("----------->",len(poly1),len(datumi))
        # poly from second stock
    poly2 = Polynomial_Regression(dionice2)


    cost_array = [28, 31, 35, 38, 38, 54, 51, 61, 52, 50, 42, 25, 36, 33, 36, 49, 46, 52, 51, 54, 54, 54, 59, 66, 65, 66, 72, 93, 102, 106, 100, 99, 98, 124, 137, 159, 118, 146, 135, 140, 139, 149, 140, 138, 143, 138, 149, 183, 185, 217, 209, 210, 224, 252, 284, 286, 290, 292, 281, 259, 237, 229, 233, 218, 222, 215, 229, 256, 234, 232, 227, 200, 184, 202, 206, 200, 199, 210, 226, 221, 223, 219, 215, 237, 237, 223, 235, 244, 253, 248, 247, 259, 258, 270, 288, 346, 392, 336, 354, 360, 386, 350, 317, 321, 366, 382, 359, 347, 317, 309, 311, 304, 276, 290, 279, 269, 326, 363, 357, 327, 343, 299, 301, 265, 248, 207, 251, 240, 220, 234, 230, 230, 225, 246, 263, 305, 297, 303, 297, 280, 273, 292, 308, 283, 245, 215, 219, 219, 226, 190, 191, 173, 182, 175, 159, 139, 110, 109, 122, 136, 141, 173, 196, 211, 197, 206]
    date_array = ['2020-01-01', '2020-01-08', '2020-01-15', '2020-01-22', '2020-01-29', '2020-02-05', '2020-02-12', '2020-02-19', '2020-02-26', '2020-03-04', '2020-03-11', '2020-03-18', '2020-03-25', '2020-04-01', '2020-04-08', '2020-04-15', '2020-04-22', '2020-04-29', '2020-05-06', '2020-05-13', '2020-05-20', '2020-05-27', '2020-06-03', '2020-06-10', '2020-06-17', '2020-06-24', '2020-07-01', '2020-07-08', '2020-07-15', '2020-07-22', '2020-07-29', '2020-08-05', '2020-08-12', '2020-08-19', '2020-08-26', '2020-09-02', '2020-09-09', '2020-09-16', '2020-09-23', '2020-09-30', '2020-10-07', '2020-10-14', '2020-10-21', '2020-10-28', '2020-11-04', '2020-11-11', '2020-11-18', '2020-11-25', '2020-12-02', '2020-12-09', '2020-12-16', '2020-12-23', '2020-12-30', '2021-01-06', '2021-01-13', '2021-01-20', '2021-01-27', '2021-02-03', '2021-02-10', '2021-02-17', '2021-02-24', '2021-03-03', '2021-03-10', '2021-03-17', '2021-03-24', '2021-03-31', '2021-04-07', '2021-04-14', '2021-04-21', '2021-04-28', '2021-05-05', '2021-05-12', '2021-05-19', '2021-05-26', '2021-06-02', '2021-06-09', '2021-06-16', '2021-06-23', '2021-06-30', '2021-07-07', '2021-07-14', '2021-07-21', '2021-07-28', '2021-08-04', '2021-08-11', '2021-08-18', '2021-08-25', '2021-09-01', '2021-09-08', '2021-09-15', '2021-09-22', '2021-09-29', '2021-10-06', '2021-10-13', '2021-10-20', '2021-10-27', '2021-11-03', '2021-11-10', '2021-11-17', '2021-11-24', '2021-12-01', '2021-12-08', '2021-12-15', '2021-12-22', '2021-12-29', '2022-01-05', '2022-01-12', '2022-01-19', '2022-01-26', '2022-02-02', '2022-02-09', '2022-02-16', '2022-02-23', '2022-03-02', '2022-03-09', '2022-03-16', '2022-03-23', '2022-03-30', '2022-04-06', '2022-04-13', '2022-04-20', '2022-04-27', '2022-05-04', '2022-05-11', '2022-05-18', '2022-05-25', '2022-06-01', '2022-06-08', '2022-06-15', '2022-06-22', '2022-06-29', '2022-07-06', '2022-07-13', '2022-07-20', '2022-07-27', '2022-08-03', '2022-08-10', '2022-08-17', '2022-08-24', '2022-08-31', '2022-09-07', '2022-09-14', '2022-09-21', '2022-09-28', '2022-10-05', '2022-10-12', '2022-10-19', '2022-10-26', '2022-11-02', '2022-11-09', '2022-11-16', '2022-11-23', '2022-11-30', '2022-12-07', '2022-12-14', '2022-12-21', '2022-12-28', '2023-01-04', '2023-01-11', '2023-01-18', '2023-01-25', '2023-02-01', '2023-02-08', '2023-02-15', '2023-02-22', '2023-03-01']
            # works dont delete time thingi
    #predicted_values = perform_time_series_regression(dionice1, datumi)
    #print(type(predicted_values))


    return render_template ('index.html',datumi=datumi, datumiLen = len(datumi),dionice=dionice,dioniceLen=len(dionice),data=data,imedionice = tiker,x=x,y=y,linearLen=len(x),dionice2=dionice2, tiker2= tiker2, tiker1 = tiker1, errormsg = errormsg ,dionice1 = dionice1, poly1 = poly1,poly2 = poly2)






    # basic Flask tamplate for testing
@app.route('/test',methods = ['GET','POST'])
def test ():


 
    #plot_regression_line(dionice,datumi, b)
    return str(45*3)




    # just keep it like this
if __name__ == "__main__":
    app.run(debug = True)
