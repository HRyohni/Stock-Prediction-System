Sveučilište Jurja Dobrile u Puli

Fakultet informatike u Puli

[![img](https://camo.githubusercontent.com/5430dc4a26787150f1809dfb3fda00e3bea22726561972627b3b050ed1274037/68747470733a2f2f63646e2e646973636f72646170702e636f6d2f6174746163686d656e74732f3933313331303331383638373233363132362f313036343330363530313937373634353037362f556e6970752d6c6f676f2d6c61742e706e67)](https://camo.githubusercontent.com/5430dc4a26787150f1809dfb3fda00e3bea22726561972627b3b050ed1274037/68747470733a2f2f63646e2e646973636f72646170702e636f6d2f6174746163686d656e74732f3933313331303331383638373233363132362f313036343330363530313937373634353037362f556e6970752d6c6f676f2d6c61742e706e67)

Dokumentacija uz projektni zadatak

Projekt dionice

Izradili: Leo Matošević, Mateo Kocev, Stevan Čorak

Studijski smjer: Informatika

Kolegij: Statistika

Mentor: doc. Darko Brborović

## Sadržaj

[TOC]

## Uvod 

U sljedećoj dokumentaciji bit će predstavljen rad i karakteristike projektnog zadatka dionica. Korištenjem Flask web okvira i Python programskog jezika stvoren je ovaj projekt. Glavni cilj ovog projekta je olakšati korisnicima dobivanje informacija vezanih za dionice. Korisnici mogu brzo pristupiti informacijama poput cijena dionica, povijesnih podataka i drugih važnih pokazatelja uz pomoć projekta dionice. Korisnici projekta imat će pristup jednostavnom i korisnički prijateljskom sučelju koje će im olakšati istraživanje i dobivanje željenih informacija. Projekt također uključuje širok spektar naprednih značajki, uključujući ažuriranja dionica u stvarnom vremenu, vizualizaciju regresija po pojedinim dionicama, odnose između dviju dionica(korelacija) i klasteriranje(grupiranje) dionica. Zahvaljujući ovim značajkama, korisnici mogu ostati ažurirani s trendovima na tržištu i donositi informirane odluke o svojim investicijama. Sveukupno, projekt dionice je izvrsni alat za sve zainteresirane za ulaganje na burzi. 

## Linearna regresija

U statistici, odnos između ovisne varijable (također poznate kao varijabla odziva) i jedne ili više nezavisnih varijabli (također poznate kao prediktorske varijable ili kovarijate) modelira se procesom nazvanim linearna regresija. Cilj linearnog modela je pronaći najbolji linearni odnos između varijabli koji se koristi za predviđanje vrijednosti ovisne varijable iz vrijednosti nezavisnih varijabli.

Prikazuje se sljedećom formulom: 
$$
\begin{equation}
y_i = \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \epsilon_i = x'_i\beta + \epsilon_i, \quad i=1,\ldots,n
\end{equation}
$$
Ova formula predstavlja model linearne regresije za predviđanje vrijednosti ovisne varijable y na temelju skupa nezavisnih varijabli x1, x2, ..., xp. Model pretpostavlja da je odnos između y i varijabli x linearan, te se može izraziti kao linearna kombinacija varijabli x s nekom greškom ε.

Formula se može interpretirati na sljedeći način:

- yi je vrijednost ovisne varijable za i-tu opažanje u skupu podataka.
- β1, β2, ..., βp su koeficijenti regresijskog modela, koji predstavljaju utjecaj svake nezavisne varijable na ovisnu varijablu. Ovi koeficijenti se procjenjuju iz podataka.
- xi1, xi2, ..., xip su vrijednosti nezavisnih varijabli za i-tu opažanje.
- εi je greška za i-to opažanje, koja predstavlja razliku između predviđene vrijednosti y i stvarne vrijednosti y za to opažanje. Pretpostavlja se da je ova greška normalno distribuirana s nultom srednjom vrijednošću i konstantnom varijancom.

Formula pokazuje da se predviđena vrijednost yi može izraziti kao zbroj umnožaka svake nezavisne varijable xi s njenim odgovarajućim koeficijentom βi, uz dodatak greške εi. Ovo se također može zapisati u vektorskom obliku kao y = Xβ + ε, gdje je y vektor duljine n (broj opažanja), X matrica dimenzija n x p nezavisnih varijabli, β je vektor dimenzija p x 1 koeficijenata, a ε je vektor dimenzija n x 1 grešaka. 

Na temelju vrijednosti nezavisnih varijabli, mogu se napraviti predviđanja o vrijednostima ovisne varijable pomoću linearnog modela regresije. Model se također može koristiti za određivanje smjera i jačine odnosa između varijabli. Različite mjere, poput koeficijenta determinacije (R-kvadrat), koji mjeri postotak varijance u ovisnoj varijabli koji je objašnjen nezavisnim varijablama, mogu se koristiti za procjenu kvalitete modela.

## Polinomna regresija


Veza između nezavisne varijable x i zavisne varijable y u polinomnoj regresiji, vrsti linearne regresije, opisuje se kao polinom n-tog stupnja. To podrazumijeva identifikaciju krivulje koja najbolje odgovara skupu podataka.

Veza između x i y modelira se kao pravac u linearnoj regresiji. Međutim, pravac neće uvijek moći prikazati temeljni obrazac podataka kada je veza između varijabli nelinearna. Polinomna regresija se može koristiti za modeliranje veze u takvim slučajevima.

Algoritam polinomne regresije izračunava vrijednosti koeficijenata koji minimiziraju sumu kvadrata pogrešaka između predviđenih vrijednosti i stvarnih vrijednosti zavisne varijable kako bi se identificirala krivulja koja najbolje odgovara podacima. Najčešće se za to koristi metoda najmanjih kvadrata.

## Korelacije 

Statistička veza ili odnos između dvije varijable naziva se korelacija. Ona posebno mjeri koliko su blisko povezane varijacije u jednoj varijabli sa varijacijama u drugoj. Koeficijent korelacije, koji ima vrijednost između -1 i +1, može se koristiti za kvantificiranje korelacije.

Korelacijski koeficijent se računa pomoću sljedeće formule:
$$
\boldsymbol{r} = \frac{\displaystyle\sum_{i=1}^n \left(\begin{matrix}x_i - \bar{x}\end{matrix}\right)\left(\begin{matrix}y_i - \bar{y}\end{matrix}\right)}{\sqrt{\displaystyle\sum_{i=1}^n \left(\begin{matrix}x_i - \bar{x}\end{matrix}\right)^2}\sqrt{\displaystyle\sum_{i=1}^n \left(\begin{matrix}y_i - \bar{y}\end{matrix}\right)^2}}
$$
Dvije varijable koje se uspoređuju u ovoj formuli su x i y, a srednje vrijednosti x i y su njihove prosječne vrijednosti. U formulu se dodaje razlika između svake x vrijednosti i srednje vrijednosti x, a ekvivalentna razlika između svake y vrijednosti i srednje vrijednosti y se dijeli sa sumom tih razlika. Kvocijent dobiven iz ovog dijeljenja se koristi za dijeljenje umnoška kvadratnog korijena sume kvadrata razlika između svake vrijednosti x i srednje vrijednosti x, pomnoženog s kvadratnim korijenom sume kvadrata razlika između svake vrijednosti y i srednje vrijednosti y. Koeficijent korelacije, označen dobivenom vrijednosti r, nalazi se između -1 i +1.

Važno je zapamtiti da veza ne uvijek ukazuje na uzrok. Dvije varijable ne uzrokuju jedna drugu samo zato što su povezane. Veza između dvije varijable može biti utjecana drugim faktorima. Stoga je pri tumačenju koeficijenata korelacije važno pažljivo razmotriti kontekst i temeljne mehanizme.

#### Upotreba Spearmanovog koeficijenta 

Statistička mjera nazvana korelacija može se koristiti za određivanje jačine i smjera odnosa između dviju varijabli. Korelacija je pojam koji se koristi za opisivanje koliko se blisko cijene dvije ili više dionica kreću u odnosu jedna na drugu u kontekstu tržišta dionica. Cijene se obično kreću u istom smjeru kada postoji pozitivna korelacija, dok se cijene kreću u suprotnim smjerovima kada postoji negativna korelacija. Razumijevanje korelacija na tržištu dionica može biti korisno za upravljanje rizicima, metode trgovanja i diversifikaciju portfelja.

Jačina i smjer monotonskih korelacija između dvije varijable mjeri se pomoću Spearmanovog koeficijenta rang-korelacije, neparametrijske statističke tehnike. Budući da se temelji na rangovima podataka umjesto na stvarnim vrijednostima, ne pretpostavlja se bilo koja posebna distribucija podataka, što ga čini robustnim na outliers. Raspon Spearmanovog koeficijenta rang-korelacije je od -1 do 1, gdje 1 označava potpuno pozitivnu monotonsku povezanost, a 0 označava da nema monotonske povezanosti.

Funkcija korelacije na temelju Spearmanovog koeficijenta u Pandas-u: Spearmanov koeficijent rang-korelacije može se odrediti pomoću ugrađene funkcije u popularnom alatu za obradu podataka u Pythonu, Pandas-u. Modul scipy.stats sadrži metodu s imenom spearmanr ().

```python
import pandas as pd
from scipy.stats import spearmanr

# Upisivanje podataka u Pandas DataFrame
stock1 = [10, 20, 30, 40, 50]
stock2 = [5, 15, 25, 35, 45]
df = pd.DataFrame({'Stock1': stock1, 'Stock2': stock2})

# Izračun Spearmanovog korelacijskog koeficijenta
correlation, _ = spearmanr(df['Stock1'], df['Stock2'])

# Ispis korelacijskog koeficijenta 
print("Spearman's Rank Correlation Coefficient: ", correlation)

```

U prethodnom primjeru, u DataFrame df (Stock1 i Stock2) stvorene su dvije kolone koje predstavljaju cijene dviju dionica. Zatim, koristeći funkciju spearmanr() izračunavamo Spearmanov koeficijent korelacije između ove dvije kolone te rezultat spremamo u varijablu korelacija. Crta (_ ) označava opcionalni izlaz koji u ovom slučaju nije potreban. Na kraju koristimo naredbu print() za ispis dobivenog koeficijenta korelacije.

Zaključak: Točne investicijske odluke zahtijevaju temeljito razumijevanje korelacija na tržištu dionica. Pandas nudi praktičan pristup za izračunavanje Spearmanovog koeficijenta korelacije rangova koristeći funkciju spearmanr() iz modula scipy.stats. Ovaj statistički alat koristan je za određivanje jačine i smjera monotone korelacije između cijena dionica. Investitori mogu diverzificirati svoja portfelja, bolje upravljati rizikom i stvarati trgovačke strategije na temelju korelacija između cijena dionica.

## K-Means

K-Means algoritam se smatra među popularnijim metodama ne nadziranog strojnog učenja te se koristi za grupiranje podataka u predefiniranu količinu klasa tj. skupina prema sličnosti unesenih podataka. 

Proces od kojeg se sastoji algoritam se dijeli na 4 jasno definirana koraka:

- Pripremanje podataka i algoritma: važno je pažljivo odabrati unesene podatke te odabrati broj klasa (klastera) koji očekujemo da će algoritam formirati. Python nam znatno olakšava proces preko već spremnih i usavršenih alata poput NumPy za sve matematičke potrebe, Pandas za pripremu podataka koju ćemo koristiti u algoritmu i SKLearn što je gotova Python biblioteka koja već nudi razne spremne algoritme za umjetnu inteligenciju te testiranje integriteta samog modela. Prvim pokretanjem programa također se nasumično dodjeljuju centri klastera (količina klastera/klasa je određena unaprijed).

  

- Dodjeljivanje podataka klasterima: svaka točka se dodjeljuje najbližem klasteru tj. njegovom centru. Daljina se računa korištenjem formule Euklidske udaljenosti:

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

  Formula naglašava da udaljenost dvije točke je jednaka korijenu zbroja kvadrata razlika njihovih koordinata u svakoj dimenziji.

  U našoj primjeni, za svaku točku računamo udaljenost od svih centra te je dodjeljujemo na centar od kojeg ima najmanju udaljenost.

  

- Pomicanje centra: Za svaki formirani klaster se računa prosječna vrijednost svih točaka u klasteru te prosječna vrijednost postaje novi centar. Za računanje prosječne vrijednosti koristimo sljedeći sustav:


  - Pretpostavimo da imamo tri točke u dvodimenzionalnom prostoru:
    **Točka 1 (4, 6)**, **Točka 2 (6, 7)**, **Točka 3 (8, 2)**

  - Kako bi izračunali novi centar, zbrojimo sve vrijednosti u svakoj dimenziji i podjelimo s brojem točaka u klasteru:

  - **x = (4 + 6 + 8) / 3**

    **y = (6 + 7 + 2) / 3**

  - novi centar će u ovom slučaju biti **(6, 5)**.

- Iteracija: Dodjeljivanje točaka i novih centra se ponavlja sve dok se centri klastera ne stabiliziraju i ostaju nepromijenjeni u sljedećoj iteraciji.

Važno je napomenuti da će se rezultati mijenjati sa različitom količinom klastera te ovisno o količini podataka i predznanju možemo preciznije odlučiti količinu klastera koju želimo formirati putem algoritma što nam određuje da je više testiranja potrebno da bi dobili zadovoljavajuće rezultate.

## Generalno objasnjenje koda

```python
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
from sklearn.model_selection import train_test_split
app = Flask(__name__)

def findData(tiker="TSLA",startDate='01/01/15',Interval= "1wk",end_date="01/04/24"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = end_date, interval = Interval)
  
    except:
        return "Error"
    return data


def roundNummber(data):
    
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

    
    date_list = datumi
    date_list = [datetime.strptime(date, '%Y-%m-%d') for date in date_list]
   
    df = pd.DataFrame({'int': int_list, 'date': date_list})
    df['date'] = (df['date'] - df['date'].min())  / np.timedelta64(1,'D')
    
    model = LinearRegression()
    model.fit(df[['date']], df['int'])
    return model.predict(df[['date']]) , df['int']



def correlation(start_date,end_date_str,symbols_list):

    symbols = []
    for ticker in symbols_list:
        try:
            r = yf.download(ticker, start=start_date, end=end_date_str)
            r['Symbol'] = ticker 
            symbols.append(r)
        except Exception as e:
           print(f"Error retrieving data for symbol {ticker}: {e}")

    df = pd.concat(symbols)
    df = df.reset_index()
    df = df[['Date', 'Close', 'Symbol']]

    df_pivot = df.pivot('Date', 'Symbol', 'Close').reset_index()

    corr_df = df_pivot.corr(method='spearman')

    plt.figure(figsize=(20, 15))
    heatmap = sns.heatmap(corr_df, cmap='coolwarm',linecolor="black", annot=True, fmt=".2f" ,annot_kws={"color": "black"}, cbar_kws={'orientation': 'vertical'})
    sns.color_palette("rocket", as_cmap=True)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Correlation', color='white')
    plt.title('Spearman Correlation Matrix')
    plt.savefig('static/img/correlation.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    print("Correlation matrix saved as correlation.png")


    corrMatrica = corr_df.to_numpy()
    tikerParovi=[]
    rowIndeks, kolIndeks = np.triu_indices(corrMatrica.shape[0], k=1)
    tikerParovi = [(symbols_list[row], symbols_list[col]) for row, col in zip(rowIndeks, kolIndeks)]

    KMPodaci = corrMatrica[rowIndeks, kolIndeks].reshape(-1, 1)

    print(KMPodaci)

    kMeans = KMeans(n_clusters=5, init='k-means++', n_init=10)
    kMeans.fit(KMPodaci)

    oznake = kMeans.labels_ 

    fig, ax = plt.subplots(figsize=(8, 6))

    scatterGraf = ax.scatter(KMPodaci, oznake, c=oznake, cmap='viridis')

    ax.set_xlabel('Vrijednosti korelacije gornjeg sektora')
    ax.set_ylabel('Klaster')
    ax.set_title('Rezultati klasteriranja')
    plt.savefig('static/img/kmeans.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    
    klasteriraniPodaci = pd.DataFrame(tikerParovi, columns=['Prvi Tiker', 'Drugi Tiker'])
    klasteriraniPodaci['Korelacija'] = KMPodaci.flatten()
    klasteriraniPodaci['Klasa'] = oznake

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
    
    startPrice = data.loc[UserStartDate][1]
    endPrice = data.loc[UserEndDate][1]
    #end_date = date_1 + datetime.timedelta(days=10)
    
        # racunanje Artimeticke sredine
    godine = []
    for x in range(int(UserStartDate.split('-')[0]),int(UserEndDate.split('-')[0])):
        godine.append(x)
        
    
    for x in range(len(godine)):
        
        try:
            
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
```

## Korišteni alati

Program je napravljen u programskom jeziku Python te sa raznim importiranim alatima (bibliotekama) smo olakšali posao i razumijevanje samog koda.

**Korištene biblioteke:**

1. `yfinance` je biblioteka koja omogućuje lagani dohvat financijskih podataka o burzi. Pruža pristup povijesnim tržišnim podatcima, financijskim izvještajima i ostalim financijskim informacijama. Podatci se hvataju direktno od Yahoo Finance usluge preko njihovog javno dostupnog API.
2. `pandas` je biblioteka za analizu podataka koja pruža jednostavne strukture podataka i alate za analizu. Omogućuje manipulaciju i analizu podataka na različite načine, kao što su čišćenje, filtriranje, grupiranje i sažimanje podataka.
3. `datetime` je modul koji pruža klase za rad s datumima i vremenom. Omogućuje manipulaciju i formatiranje datuma i vremena.
4. `numpy`  pruža alate za numeričku obradu podataka u Pythonu. Pruža brze i precizne numeričke operacije na nizovima (vektorima) i matricama, te uključuje funkcije za matematičke operacije i linearnu algebru.
5. `statsmodels` pruža niz statističkih alata za analizu podataka. Uključuje funkcije za analizu regresije, vremenskih serija i testiranja hipoteza.
6. `yahoo_fin` je slična biblioteka poput `yfinance` koju smo koristili za testiranje točnosti dobivenih podataka te kako bi smo odlučili koja je pouzdaniji izvor.
7. `matplotlib` pruža alate za stvaranje vizualizacija poput grafikona, dijagrama i grafova. Ovaj alat smo koristili za prikaz i generiranje tablica (npr. korelacija i kmeans).
8. `Flask` je jednostavni web framework koji smo koristili za izgradnju web aplikacije (naša metoda prezentiranja projekta).
9. `sklearn` ili scikit-learn pruža niz alata za strojno učenje i rudarenje podataka. Pruža funkcionalnosti za klasifikaciju, regresiju i  klasteriranje, te nam je bio jedan od korisnijih alata za implementaciju kmeans-a.
10. `math` je osnovni Python alat koji pruža opširnu kolekciju matematičkih operacija ugrađene direktno u Python.
11. `seaborn` pruža dodatne alate za vizualizaciju podataka, uključujući naprednije grafikone i dijagrame te smo ga koristili uz `matplotlib` za olakšavanje rada na projektu.

## Pripremljene funkcije

#### Funkcija za dohvaćanje podataka

```python
def findData(tiker="TSLA",startDate='01/01/15',Interval= "1wk",end_date="01/04/24"):
    try:
        data = get_data(tiker, start_date = startDate, end_date = end_date, interval = Interval)
  
    except:
        return "Error"
    return data
```

Ova funkcija se koristi za spremanje tjednih podataka o odabranom tikeru u vremenskom intervalu između početnog datuma `startDate` i krajnjeg datuma `end_date` te u slučaju greške vrati grešku. Koristimo `yfinance` biblioteku za `get_data` metodu kako bi pristupili podacima burze.



####  Funkcija za zaokruživanje podataka

```python
def roundNummber(data):
        # round nummbers for better view
    data["open"] = list(np.around(np.array(data["open"]),2))
    data["low"] = list(np.around(np.array(data["low"]),2))
    data["high"] = list(np.around(np.array(data["high"]),2))
    data["adjclose"] = list(np.around(np.array(data["adjclose"]),2))
    return data
```

Iz našeg NumPy polja  uzivamo podatke u raznim kolonama te ih zaokružimo na dvije decimale sa `np.around` metodom.

#### Linearna regresija

```python
def LinearnaAgregacija(data):

    datumi = []
    for x in data["open"].index.values:
        datumi.append(str(x)[:-19])
    
    int_list = data["open"]

    # Lista Datuma
    date_list = datumi
    date_list = [datetime.strptime(date, '%Y-%m-%d') for date in date_list]
   
    # deklaracija datafremea sa datumima i podacima
    df = pd.DataFrame({'int': int_list, 'date': date_list})
    df['date'] = (df['date'] - df['date'].min())  / np.timedelta64(1,'D')

    # deklaracija modela za linearnu regresiju
    model = LinearRegression()
    model.fit(df[['date']], df['int'])
    return model.predict(df[['date']]) , df['int']
```

Ovu funkciju koristimo za definiranje trenda koristeći linearnu regresiju na podacima o cijenama dionica.

Kao argument funkciji dodjeljujemo ulazne podatke o dionicama te izvlačimo datume preko iteracije kroz polje. U odvojenu listu spremamo cijenu dionice na taj datum te ih zajedno sa datumima spajamo u jedan dataframe. Na kraju umetnemo podatke u model linearne regresije te vračamo rezultate na frontend sa `return` funkcijom.



#### Korelacija i K-Means

```python
def correlation(start_date,end_date_str,symbols_list):

    # uzimamo podatke o tikerima spremljenima u symbols_list varijabli
    symbols = []
    for ticker in symbols_list:
        try:
            r = yf.download(ticker, start=start_date, end=end_date_str)
            # dodajemo tikere kao kolonu
            r['Symbol'] = ticker 
            symbols.append(r)
        except Exception as e:
           print(f"Error retrieving data for symbol {ticker}: {e}")

    # spajamo sve potrebne podatke u jedan DataFrame
    df = pd.concat(symbols)
    df = df.reset_index()
    df = df[['Date', 'Close', 'Symbol']]

    # stvaranje "pivot" tablice za točan format podataka za računanje korelacije
    df_pivot = df.pivot('Date', 'Symbol', 'Close').reset_index()

    # računanje korelacije sa Spearman metodom
    corr_df = df_pivot.corr(method='spearman')

    # prikazivanje rezultate korelacije pomoću seaborn biblioteke u obliku heatmape
    plt.figure(figsize=(20, 15))  # veličina tablice (fizički prikaz na ekranu, ručno podesena za dobar prikaz tablice)
    heatmap = sns.heatmap(corr_df, cmap='coolwarm',linecolor="black", annot=True, fmt=".2f" ,annot_kws={"color": "black"}, cbar_kws={'orientation': 'vertical'})  # uređivanje tablice (boja, itd.)
    sns.color_palette("rocket", as_cmap=True)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Correlation', color='white')
    plt.title('Spearman Correlation Matrix')
    plt.savefig('static/img/correlation.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)  # spremanje tablice u PNG datoteku za frontend
    print("Correlation matrix saved as correlation.png")


    corrMatrica = corr_df.to_numpy()
    tikerParovi=[]
    #ovim dijelom uzimamo gornju polovicu podataka (bez dijagonale) te ih pretvaramo u parove tikera
    rowIndeks, kolIndeks = np.triu_indices(corrMatrica.shape[0], k=1)
    tikerParovi = [(symbols_list[row], symbols_list[col]) for row, col in zip(rowIndeks, kolIndeks)]

    # mijenjamo format polja podataka
    KMPodaci = corrMatrica[rowIndeks, kolIndeks].reshape(-1, 1)

    print(KMPodaci)
	
    #inicijaliziramo model
    kMeans = KMeans(n_clusters=5, init='k-means++', n_init=10)
    kMeans.fit(KMPodaci)

    oznake = kMeans.labels_  # spremamo klaster oznake za svaki podatak

    fig, ax = plt.subplots(figsize=(8, 6))  # stvaranje i spremanje grafa u PNG sliku

    scatterGraf = ax.scatter(KMPodaci, oznake, c=oznake, cmap='viridis')

    ax.set_xlabel('Vrijednosti korelacije gornjeg sektora')
    ax.set_ylabel('Klaster')
    ax.set_title('Rezultati klasteriranja')
    plt.savefig('static/img/kmeans.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    

    # Stvaramo novi DataFrame za povezivanje svih potrebnih podataka za ispisivanje detaljnih rezutata
    klasteriraniPodaci = pd.DataFrame(tikerParovi, columns=['Prvi Tiker', 'Drugi Tiker'])
    klasteriraniPodaci['Korelacija'] = KMPodaci.flatten()
    klasteriraniPodaci['Klasa'] = oznake

    return klasteriraniPodaci
```

U sljedećoj funkciji po preporuci profesora smo napravili alat koji će računati korelaciju naših podataka te će ih prikazati u generiranoj tablici te će rezultate korelacije svrstati u kmeans algoritam za klasteriranje i također prikazati rezultate u tablici pomoću `matplotlib` biblioteke i također će ispisati detalje rezultate klasteriranja u dodatnu tablicu.

#### Vremenska regresija

```python
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
```

Funkcija perform_time_series_regression  izvodi regresiju vremenskih nizova na nizu troškova (cost_array) i nizu datuma (date_array). Metoda stvara okvir podataka iz ulaznih nizova koristeći biblioteke pandas i statsmodels, a primjenjuje ARIMA (Autoregressive Integrated Moving Average) model s redom (1,1,1) kako bi izvršila regresiju vremenskih nizova. Metoda summary() iz statsmodels biblioteke koristi se za izvješćivanje o rezultatima regresije, dok funkcija predict() iz iste biblioteke vraća predviđene vrijednosti za troškove.

#### Polinomna regresija

```python
def Polynomial_Regression(y):
  x=[]
  for i in range(len(y)):
      x.append(i+1)
  mymodel = np.poly1d(np.polyfit(x, y, 3))


  myline = np.linspace(1, len(y), len(y))

  
  plt.scatter(x, y)
  plt.plot(myline, mymodel(myline))
  return mymodel(myline)
```

Funkcija se koristi za stvaranje koji se sastoji od brojeva od 1 do duljine ulaznog niza y. Ovaj niz se koristi za stvaranje polinomne funkcije. Koristeći `polyfit` metodu, izračunavaju se koeficijente polinoma koji najbolje odgovaraju ulaznom nizu y. Zatim se koriste za stvaranje polinomne funkcije s pomoću `poly1d` metode. Ova funkcija vraća polinomnu funkciju koja se može koristiti za predviđanje vrijednosti na temelju ulaznog niza. Sljedeće stvaramo vrijednosti na x-osi s pomoću `linspace` metode, također stvaramo niz vrijednosti koje su jednako udaljene na x-osi i koje se kreću od 1 do duljine ulaznog niza y. Na kraju crtamo graf te sa `return` šaljemo rezultate u implementaciju na frontend.



#### Simulacija i ponašanje dionica kroz vrijeme

```python
def Simulation(UserStartDate, UserEndDate, amount, tikerName):
    data = findData(tiker=tikerName, startDate=UserStartDate, end_date="2023-04-01", Interval="1d")
    PoVD_AS = []
    GS = 0
    
    start_date = datetime.strptime(UserStartDate, '%Y-%m-%d')
    end_date = datetime.strptime(UserEndDate, '%Y-%m-%d')
    time_difference = (end_date - start_date).days
    
    startPrice = data.loc[UserStartDate][1]
    endPrice = data.loc[UserEndDate][1]
    #end_date = date_1 + datetime.timedelta(days=10)
    
        # racunanje Artimeticke sredine
    godine = []
    for x in range(int(UserStartDate.split('-')[0]),int(UserEndDate.split('-')[0])):
        godine.append(x)
        
    
    for x in range(len(godine)):
        
        try:
            
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
   
    return AP, round(PoVD,2),profit
```

Funkcija poziva funkciju `findData` kako bi pristupili podacima o traženom tikeru u određenom vremenskom rasponu, računamo prosječnu stopu rasta (GS) tijekom vremenskog razdoblja za koje su dostupni podaci, koristeći aritmetičku sredinu godišnjih stopa rasta (PoVD_AS). Sljedeće izračunavamo profit koji bi se mogao ostvariti ako bi se uložio traženi iznos, pri čemu je cijena dionice na kraju razdoblja množena s uloženim iznosom.

Funkcijom također računamo povrat investicije (PoVD) i godišnju stopu rasta (AP) na temelju cijene dionica na početku i kraju perioda ulaganja.



#### Razlika vremena u danima

```python
def days_between_dates(date1, date2):
    date_format = "%Y-%m-%d"
    date1_obj = datetime.strptime(date1, date_format)
    date2_obj = datetime.strptime(date2, date_format)
    diff_days = abs((date2_obj - date1_obj).days)
    return diff_days
```

Jednostavna pomoćna funkcija koja računa razliku vremena u danima te vrača rezultat.

#### Main / Frontend funkcija

```python
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
```

U našoj glavnoj funkciji koja se koristi za gradnju web aplikacije imamo deklarirane potrebne varijable te njihove default vrijednosti.

Aplikacija prikazuje osnovne podatke o vrijednosti i kretanju dionica te omogućava pretraživanje istih. Aplikacija također pruža mogućnost uspoređivanja dionica te predstavlja ih jedan pored drugog u istom grafu za lakšu usporedbu. Preko `if` selekcije provjeravamo je li podatak pronađen sa `findData()` funkcijom te ispisuje rezultate.

`main` također poziva i priprema ulazne podatke za simulaciju kretanja dionice gledajući povjesne podatke, prezentaciju podataka za korelaciju i prezentaciju podataka i tablice K-Means klasteriranja.

Vraćamo obrađene podatke pomoću `return` funkcije na frontend.



## Zaključak

Zaključno, Flask web okvir i Python programski jezik korišteni su za stvaranje projekta dionica, koji nastoji korisnicima pružiti jednostavan pristup informacijama vezanim za dionice. Projekt pruža intuitivno korisničko sučelje zajedno s nizom sofisticiranih značajki, uključujući ažuriranja dionica u stvarnom vremenu, vizualizaciju regresije, korelaciju analizu i grupiranje dionica. Korisnici mogu ostati u tijeku s trendovima na tržištu i donositi mudre financijske odluke uz pomoć ovih usluga. Sveukupno, projekt dionice je fantastičan alat za sve zainteresirane za ulaganje na burzi.

##  Literatura 

Darko Brborović- Upravljanje financijskom imovinom(knjiga)

https://realpython.com/numpy-scipy-pandas-correlation-python/

https://www.interviewqs.com/blog/py-stock-correlation

https://scikit-learn.org/stable/

https://pandas.pydata.org/

https://numpy.org/doc/stable/user/index.html#user

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

https://seaborn.pydata.org/

https://matplotlib.org/stable/users/index

http://www.mathos.unios.hr/ptfstatistika/Vjezbe/materijali_7.pdf

https://www.analyticsvidhya.com/blog/2021/10/understanding-polynomial-regression-model/
