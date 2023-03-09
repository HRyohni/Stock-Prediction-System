# poziv pandas biblioteke || alat za analizu i manipulaciju podataka || pip install pandas || pip install pandas-datareader
import pandas as pd
import pandas_datareader as pdr

# poziv numpy biblioteke || implementira razne matematičke funkcije || pip install numpy
import numpy as np

# poziv datetime biblioteke || pruža abstraktne podatke za vrijeme i datume || dolazi sa pythonom kad ga instaliraš
from datetime import dt

# poziv za matplotlib || koristi se za interaktivne chartove || pip install matplotlib
import matplotlib.pyplot as plt

# poziv flaska || web framework za aplikaciju || pip install flask
from flask import Flask, render_template, request,redirect

# pozivi iz scikit learn biblioteke || unsupervised machine learning || pip install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

# pozivi yahoo finance biblioteke || koriste isti API ali su dva različita autora || pip install yahoo-fin || pip install yfinance
from yahoo_fin import stock_info as si
from yahoo_fin.stock_info import get_data
import yfinance as yf



app = Flask(__name__)


def findData(tiker="TSLA",startDate='01/01/22',Interval="1d"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = "03/03/23", interval = "1wk")
        print(data)
            
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
    date_list = [dt.strptime(date, '%Y-%m-%d') for date in date_list]

    # Create a DataFrame with the int and date lists
    df = pd.DataFrame({'int': int_list, 'date': date_list})
    df['date'] = (df['date'] - df['date'].min())  / np.timedelta64(1,'D')

    # Create a Linear Regression Model
    model = LinearRegression()
    model.fit(df[['date']], df['int'])
    return model.predict(df[['date']]) , df['int']



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
        

   
    # getting linear agression
    x,y = LinearnaAgregacija(data)
    
    # round nummbers for better view
    roundNummber(data)
        
 
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
