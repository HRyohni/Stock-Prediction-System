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
import math
import seaborn as sns
from sklearn.cluster import KMeans
app = Flask(__name__)

from sklearn.model_selection import train_test_split
def findData(tiker="TSLA",startDate='01/01/15',Interval= "1wk",end_date="01/04/24"):
    try:
        
       
        data = get_data(tiker, start_date = startDate, end_date = end_date, interval = Interval)
  
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



def correlation(start_date,end_date_str,symbols_list):

    # Retrieve stock data from Yahoo Finance using yfinance
    symbols = []
    for ticker in symbols_list:
        try:
            r = yf.download(ticker, start=start_date, end=end_date_str)
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
    heatmap = sns.heatmap(corr_df, cmap='coolwarm',linecolor="black", annot=True, fmt=".2f" ,annot_kws={"color": "black"}, cbar_kws={'orientation': 'vertical'})  # Customize the heatmap
    sns.color_palette("rocket", as_cmap=True)
    cbar = heatmap.collections[0].colorbar  # Get the colorbar
    cbar.set_label('Correlation', color='white')  # Set the color of the label to white
    plt.title('Spearman Correlation Matrix')
    plt.savefig('static/img/correlation.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)  # Save the figure as a PNG file with transparent background
    print("Correlation matrix saved as correlation.png")


    corrMatrica = corr_df.to_numpy()
    tikerParovi=[]
    # Get indices of the upper triangle (excluding diagonal) and convert them to ticker pairs
    rowIndeks, kolIndeks = np.triu_indices(corrMatrica.shape[0], k=1)
    tikerParovi = [(symbols_list[row], symbols_list[col]) for row, col in zip(rowIndeks, kolIndeks)]

    # Get the upper triangle values and reshape them into a 2D array with a single column
    KMPodaci = corrMatrica[rowIndeks, kolIndeks].reshape(-1, 1)

    print(KMPodaci)

    kMeans = KMeans(n_clusters=5, init='k-means++', n_init=10)
    kMeans.fit(KMPodaci)

    oznake = kMeans.labels_  # klaster oznake za svaki podatak

    fig, ax = plt.subplots(figsize=(8, 6))  # stvaranje grafa

    scatterGraf = ax.scatter(KMPodaci, oznake, c=oznake, cmap='viridis')

    ax.set_xlabel('Vrijednosti korelacije gornjeg sektora')
    ax.set_ylabel('Klaster')
    ax.set_title('Rezultati klasteriranja')
    plt.savefig('static/img/kmeans.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    

    # Creating a dataframe to store the ticker pairs, correlation values, and assigned clusters
    klasteriraniPodaci = pd.DataFrame(tikerParovi, columns=['Prvi Tiker', 'Drugi Tiker'])
    klasteriraniPodaci['Korelacija'] = KMPodaci.flatten()
    klasteriraniPodaci['Klasa'] = oznake

    # Displaying the resulting dataframe
    return klasteriraniPodaci

    

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

  
  plt.scatter(x, y)
  plt.plot(myline, mymodel(myline))
  #plt.show()
  return mymodel(myline)


def Simulation(UserStartDate, UserEndDate, amount, tikerName):
    data = findData(tiker=tikerName, startDate=UserStartDate, end_date="2023-04-01", Interval="1d")
    PoVD_AS = []
    GS = 0
    
    start_date = datetime.strptime(UserStartDate, '%Y-%m-%d')
    end_date = datetime.strptime(UserEndDate, '%Y-%m-%d')
    time_difference = (end_date - start_date).days
    
    startPrice = data.loc[UserStartDate][1] # get stock by the date
    endPrice = data.loc[UserEndDate][1]
    #end_date = date_1 + datetime.timedelta(days=10)
    
        # racunanje Artimeticke sredine
    godine = []
    for x in range(int(UserStartDate.split('-')[0]),int(UserEndDate.split('-')[0])):
        godine.append(x)
        
    
    for x in range(len(godine)):
        
        try: # za svaku godinu NISTA NE RADI
            
            temp = findData(tiker=tikerName, startDate= str (str(godine[x]) + "-" + str(UserStartDate)[5:]), end_date= pd.to_datetime(str (str(godine[x]) + "-" + str(UserStartDate)[5:])) + pd.DateOffset(days=1), Interval="1d")
            temp2 = findData(tiker=tikerName, startDate= str (str(godine[x]+1) + "-" + str(UserStartDate)[5:]), end_date= pd.to_datetime(str (str(godine[x]+1) + "-" + str(UserStartDate)[5:])) + pd.DateOffset(days=1), Interval="1d")
            GS *= temp2 ["open"][0] / temp["open"][0]
            
            
            
            
        except: # za vikende
            print("Error_Godine")
    for x in PoVD_AS:
        GS += GS ** 1/ len(godine)-1
   


    profit = int(amount) * endPrice  # profit
    PoVD = endPrice / startPrice # Povrat za vrijeme drˇzanja
    
    AP = PoVD ** (1 / 3)
   
    
    
    # return str(round(profit,2)*amount)+"$ bought with:"+ str(round(startPrice,2)*amount)
    return AP, round(PoVD,2),profit

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


def days_between_dates(date1, date2):
    date_format = "%Y-%m-%d"
    date1_obj = datetime.strptime(date1, date_format)
    date2_obj = datetime.strptime(date2, date_format)
    diff_days = abs((date2_obj - date1_obj).days)
    return diff_days


@app.route('/',methods = ['GET','POST'])
def main ():
    # simulacija
    AP = 0  
    profit = 0
    days =""
    PoVD = ""
    Sdatumi =[]
    SDionice=[]

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

    #correlation
    tikerOd = ""
    kmeans = []
    

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

        if 'STikername' in request.form or 'SAmount' in request.form:
            
            # importing input from web
            Stikername = request.form['STikername']   
            SAmount = request.form['SAmount'] 
            SDateFrom = request.form['SDateFrom'] 
            SDateTo = request.form['SDateTo'] 
            days = days_between_dates(SDateFrom,SDateTo)
            
            AP ,PoVD, profit = Simulation(UserStartDate= SDateFrom,UserEndDate= SDateTo,amount= SAmount, tikerName=Stikername)
            SDionice = findData(tiker=Stikername, startDate=SDateFrom, end_date=SDateTo, Interval="1d")
            for x in SDionice["open"].index.values:
                Sdatumi.append(str(x)[:-19])
        #  ? tikeri korelacija

        if 'tikeri' in request.form:
            tikeri = request.form['tikeri'] 
            tikerDo = request.form['tikerDo'] 
            tikerOd = request.form['tikerOd'] 
            tikeri = tikeri.split(" ")
            kmeans = correlation(tikerOd,tikerDo,tikeri)
            print(kmeans)
        


                

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
    poly2 = Polynomial_Regression(dionice2)

    
    return render_template ('index.html',datumi=datumi, datumiLen = len(datumi),dionice=dionice,dioniceLen=len(dionice),data=data,imedionice = tiker,x=x,y=y,linearLen=len(x),dionice2=dionice2, tiker2= tiker2, tiker1 = tiker1, errormsg = errormsg ,dionice1 = dionice1, poly1 = poly1,poly2 = poly2, AP = round(AP*100,2),PoVD=PoVD,days=days,Sdatumi = Sdatumi,SDionice = SDionice,SdatumiLen = len(Sdatumi),SDioniceLen = len(SDionice),profit = profit,tikerOd = tikerOd,kmeans = kmeans, kmeansLen = len(kmeans))





@app.route('/simulation',methods = ['GET','POST'])
def Simulation2 ():
   
 return render_template ('index.html')

    # basic Flask tamplate for testing
@app.route('/test',methods = ['GET','POST'])
def test ():

    Simulation("2018-01-01","2019-04-22",23,"AAPL")
 
    #plot_regression_line(dionice,datumi, b)
    return str(45*3)




    # just keep it like this
if __name__ == "__main__":
    app.run(debug = True)
    #app.run(host='0.0.0.0', port=5000)
