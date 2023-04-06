import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #high-low (volatility) %
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #daily % change

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] #features/attributes (what may cause the adjusted close price in 10% of the data to change)

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #fill not available with -99999 to replace NaN, treated as outlier

forecast_out = int(math.ceil(0.01*len(df))) #This is the number of days out, predict out 10% of the dataframe. Use regression to forecast out. math.ceil will take anything and get to the ceiling (round everything up to the nearest whole), and makes it an int.

df['label'] = df[forecast_col].shift(-forecast_out) #this is creating the label, and shifting the columns up so each row is the adjusted close 10 days into the future

X = np.array(df.drop(['label'], 1)) #everything but the label column, returns a new dataframe and converts to numpy array
X = preprocessing.scale(X) #scale it so its normalized between all the data points including all your other values, typically this is skipped
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

#X = X[:-forecast_out+1] #because typically we wouldn't have labels there for those X's, but because we use the dropna earlier, they are removed
#df.dropna(inplace=True) don't need to drop this either cause the NaN's were dropped before
df.dropna(inplace=True)
y = np.array(df['label']) #just the label column

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #splits up the dataframe, 20% is used for testing

#clf = LinearRegression(n_jobs=-1) #establishes the model
#clf = svm.SVR(kernel='poly') #stablishes an svm model, change kernels for different values
#clf.fit(X_train, y_train) #trains with the training data

#with open('linearregression.pickle', 'wb') as f:
    #pickle.dump(clf, f)

#pickle basically saves us the time of retraining the model every time, saves the model

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) #gets a percentage accuraccy (tests)

forecast_set = clf.predict(X_lately) #pass in an order or set of values that help go through each forecast

print(forecast_set, accuracy, forecast_out) #shows the predicted values for the next 30 days

df['Forecast'] = np.nan

last_date = df.iloc[-1].name # last date
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set: #iterating through forecast set, taking each foreacast and day, adn setting each of those as values in the dataframe.
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #referencing the index for the dataframe, and referencing the next day (predictions), the date IS the index, if it existed use it if not, create it. I is the list of forecasts, mashes the lists together for each row

print(df.head())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()