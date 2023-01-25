import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
from flask import Flask, render_template, request,redirect



msft = yf.Ticker("TSLA") # for getting aditional data and stuff
data = get_data("TSLA", start_date = '01/01/2023', end_date = None, interval = "1d")

print(data)

datumi =[]
dionice = []

for y in data["open"]:
    dionice.append(int(y))


for x in data["open"].index.values:
    datumi.append(str(x)[:-19])



app = Flask(__name__)


@app.route('/',methods = ['GET','POST'])
def main ():




    return render_template ('index.html',datumi=datumi, datumiLen=len(datumi),dionice=dionice,dioniceLen=len(dionice),data=data)




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

