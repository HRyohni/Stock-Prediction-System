import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
from flask import Flask, render_template, request,redirect



def findData(tiker="TSLA",startDate='01/01/2021',Interval="1wk"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = None, interval = Interval)
    except:
        
        return "Error"
    return data

datumi =[]
dionice = []

app = Flask(__name__)


@app.route('/',methods = ['GET','POST'])
def main ():

        #default set to TSLA
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

