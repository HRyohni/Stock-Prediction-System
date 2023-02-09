import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
from flask import Flask, render_template, request,redirect


app = Flask(__name__)


        # collerraction
#x = [11, 2, 7, 4, 15, 6, 10, 8, 9, 1, 11, 5, 13, 6, 15]
#y = [2, 5, 17, 6, 10, 8, 13, 4, 6, 9, 11, 2, 5, 4, 7]

# to return the upper three quartiles
#pearsons_coefficient = np.corrcoef(x, y)
#print("The pearson's coeffient of the x and y inputs are: \n" ,pearsons_coefficient)


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

def findData(tiker="TSLA",startDate='01/01/2023',Interval="1d"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = None, interval = Interval)
    except:
        
        return "Error"
    return data








@app.route('/',methods = ['GET','POST'])
def main ():

        #default set to TSLA
    datumi =[]
    dionice = []
    data = findData()
    tiker="TSLA"
    


        # on post
    if request.method == "POST":
            # if post is search
        if 'search' in request.form:


                # change data with user input
            search = request.form['search']
            tiker= str(search)

            data= findData(str(search))
            
            try:
                data= findData(str(search))
                if data == "Error":
                    tiker ="Cant find that."
                print("Cant find that")
                data = findData()
            except:
                
                print("correct")

                

        #for editing data 
    for y in data["open"]:
        dionice.append(int(y))


    for x in data["open"].index.values:
        datumi.append(str(x)[:-19])
        
    test = findData()    
    data["open"] = list(np.around(np.array(data["open"]),2))
    data["low"] = list(np.around(np.array(data["open"]),2))
    data["high"] = list(np.around(np.array(data["open"]),2))
    data["adjclose"] = list(np.around(np.array(data["open"]),2))
    data["volume"] = list(np.around(np.array(data["open"]),2))
    print("datum---------------------------->"+ str(len(datumi)))
    print("value---------------------------->"+ str(data["open"]))
    print("value---------------------------->"+ str(datumi))
    #plot_regression_line(dionice,datumi, b)
    return render_template ('index.html',datumi=datumi, datumiLen=len(datumi),dionice=dionice,dioniceLen=len(dionice),data=data,tiker = tiker)




if __name__ == "__main__":
    app.run(debug = True)

##print(type(data))
#dionica = []
#datum = []

#print (str(data.index.values[0])[: -19]) # getting good date format
#for x in range(len(data)):
#    datum.append ("Datum: "+str(data.index.values[x])[: -19])
#    dionica.append("Dionica: "+ str(data["open"][x]))

#plt.plot(dionica,datum)

#datetime = datetime.strptime(str(data.index.values[0]), '%Y-%m-%d')
#print(datetime)


#print(data)


#plt.show()


# Graph
#dataframe = pd.read_excel("output.xlsx")
 # or plt.savefig("name.png")
    #exports data
#df1 = pd.DataFrame(data)
#df1.to_excel("output.xlsx") 

#for x in msft.info.keys():
#    print(str(x)+":  "+ str(msft.info.get(x)))

