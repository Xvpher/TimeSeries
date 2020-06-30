# %%
"""
# Notebook for analyzing my ecg data

"""

# %%
"""
### 1) Import all necessary libraries
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
#import spectrum
import statsmodels.api as sma
from statsmodels.tsa.stattools import pacf, adfuller
from scipy.stats import boxcox
from scipy import signal
import warnings
from datetime import datetime
import pprint

# %%

##############################
# Suppress Warnings Settings and Pretty Print Settings
##############################

# suppress warnings completely
warnings.filterwarnings('ignore')

# suppress warnings after seen once
# warnings.filterwarnings(action='once')

# pretty print settings
pp = pprint.PrettyPrinter(indent=4)

# %%
"""
### 2) Select Data and view them in data frames
"""

# %%
##############################
# Import Dataset
##############################

# import dataset from csv
dataframe = pd.read_csv(r'D:\5th\Project\Time series analysis\dataset\sample1.csv', engine='python')

# select lead to be used
lead = 'II'



# %%
# seperate data for hold out (split) testing
use_dataframe = dataframe.iloc[225:1225][lead]
test_dataframe=dataframe.iloc[1225:2225][lead]
test_dataframe2=dataframe.iloc[1324:2324][lead]

test_dataframe3 = dataframe.iloc[2405:3405][lead]

# values of time series data
y = use_dataframe.values
y_test = test_dataframe.values

# %%
# dataset head
dataframe.head(5)

# %%
# dataset characteristics
dataframe.describe()

# %%
##############################
# Visualize Data from all leads
##############################

dataframe.plot(figsize=(18, 12), subplots=True)
plt.xticks(rotation=90)
plt.show()

# %%
dataframe.plot(figsize=(18, 12))
plt.xticks(rotation=90)
## plt.savefig('all.pdf')
plt.show()

# %%
dataframe.iloc[0:15000][lead].plot(figsize=(18, 12))

plt.show()

# %%
a,b=signal.find_peaks(dataframe.iloc[0:15000][lead],height=400)

# %%
a


# %%
#aa=signal.peak_widths(dataframe.iloc[0:15000][lead],a,rel_height=1.0)

# %%
#start the use data frame with 225 -1225
#                              1225 -2225  therfore 1324 - 2324 
#                              2405-3405
                                                                     

# %%
A=pd.DataFrame(a)

# %%
B=A.shift(1)
B=B.dropna()



# %%
C=A-B

# %%
#C

# %%
596-225


# %%
776-424

# %%
C.mean()

# %%
# Visualize Data from Selected Lead
##############################

use_dataframe.plot(figsize=(18, 7), color='g')
plt.title('Train Data for Lead {}'.format(lead))
plt.xticks(rotation=90)
plt.show()

test_dataframe.plot(figsize=(18, 7), color='darkblue')
plt.title('Test Data for Lead {}'.format(lead))
plt.xticks(rotation=90)
plt.show()

test_dataframe2.plot(figsize=(10, 7), color='black')
plt.title('Test Data for Lead {}'.format(lead))
plt.xticks(rotation=90)
plt.show()





# %%
def stationarity(timeseries):
    wind=365
    rolmean=timeseries.rolling(window=wind).mean()
    rolstd=timeseries.rolling(window=wind).std()
    
    plt.figure(figsize=(20,10))
    actual=plt.plot(timeseries, color='red', label='Actual')
    mean_6=plt.plot(rolmean, color='green', label='Rolling Mean') 
    std_6=plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Dickey-Fuller Test: ')
    dftest=adfuller(timeseries, autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# %%
stationarity(test_dataframe)

# %%
"""

## analysis part


"""

# %%
A=use_dataframe.max(0)
A

# %%
Use_dataframe=use_dataframe.div(1500)

# %%
plt.plot(Use_dataframe)

# %%


# %%
"""
####  
     A) Box Cox Tranformation
    
"""

# %%
##############################
# Visualize Box Cox Transformation
##############################

# make df positive by adding constant
constant = Use_dataframe.min(0)
positive_dataframe = Use_dataframe + np.abs(constant) + 0.1

# box cox transformation
lmbda = 2
Y_boxcox = boxcox(positive_dataframe, lmbda)

# plot box cox transform
x_vals = np.arange(1, 1001)
plt.subplots(figsize=(18, 7))
plt.title('Box Cox Transformation for Lead {}'.format(lead))
plt.plot(x_vals, Y_boxcox, label='Box Cox (Lambda {:0.3f})'.format(lmbda))

# plot original data
plt.plot(x_vals, positive_dataframe.values, label='Original Data')

plt.legend()
plt.show()


# %%


# %%
"""
### log transformation
"""

# %%
##############################
# Visualize Log transformation
##############################

# make df positive by adding constant
constant = Use_dataframe.min(0)
positive_dataframe = Use_dataframe + np.abs(constant) + 0.3

# log transformation
Y_log = positive_dataframe.apply(np.log)

# plot log transform
x_vals = np.arange(1, 1001)
plt.subplots(figsize=(18, 7))
plt.title('Log Transformation for Lead {}'.format(lead))
plt.plot(x_vals, Y_log, label='Log Transform')

# plot original data
plt.plot(x_vals, positive_dataframe.values, label='Original Data')

plt.legend()
plt.show()

# %%
constant

# %%
"""
### square root tranformation
"""

# %%
##############################
# Visualize Square Root Transformation
##############################

# make df positive by adding constant
constant = Use_dataframe.min(0)
positive_dataframe = Use_dataframe + np.abs(constant) + 1.5

# sqrt transformation
Y_sqrt = positive_dataframe.apply(np.sqrt)

# plot sqrt transform
x_vals = np.arange(1, 1001)
plt.subplots(figsize=(18, 7))
plt.title('Square Root Transformation for Lead {}'.format(lead))
plt.plot(x_vals, Y_sqrt, label='Square Root Transform')

# plot original data
plt.plot(x_vals, positive_dataframe.values, label='Original Data')

plt.legend()
plt.show()

# %%
"""
### power transformation 
"""

# %%
##############################
# Visualize Power Transformation
##############################

# power transformation
power = 2
Y_power = Use_dataframe ** power - 1

# plot sqrt transform
x_vals = np.arange(1, 1001)
plt.subplots(figsize=(18, 7))
plt.title('Power Transformation for Lead {}'.format(lead))
plt.plot(x_vals, Y_power, label='Power Transform (P = {})'.format(power))

# plot original data
plt.plot(x_vals, Use_dataframe.values, label='Original Data')

plt.legend()
plt.show()


# %%


# %%


# %%
"""
## Model Identification
"""

# %%
df_boxcox = pd.Series(Y_boxcox)
stationarity(df_boxcox)

# %%
"""

### plot acf and pacf
"""

# %%
##############################
# Plot ACF and PACF
##############################

y = Y_boxcox

fig = plt.figure(figsize=(18,18))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsaplots.plot_acf(y, lags=999, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsaplots.plot_pacf(y, lags=999, ax=ax2, **{'marker': '.'})
plt.show()


# %%


# %%
"""
## Spectogram Plot
"""

# %%
##############################
# Plot Magnitude Spectrum for freq=375
##############################
fs = 500.6
f, t, Sxx = signal.spectrogram(df_boxcox, fs, scaling='density')

fig = plt.figure(figsize=(18,10))
plt.pcolormesh(t, f, Sxx)
plt.show()

# %%
"""
## Spectral Analysis 
"""

# %%
##############################
# Using Periodogram
##############################

# f = 3000

fig = plt.figure(figsize=(18,10))


plt.psd(Use_dataframe,Fs=500.6,marker='.')
plt.xlabel('Frequency (Hz)')
plt.title('Spectral Analysis')
plt.show()

# %%


# %%
"""
## Seasonal differencing
"""

# %%
##############################
# Apply Seasonal Differencing
##############################

df_seasonal_diff = df_boxcox - df_boxcox.shift(365)
df_seasonal_diff.dropna(inplace=True)

stationarity(df_seasonal_diff)

# %%
plt.figure(figsize=(20,10))
plt.plot(df_boxcox,label='boxcox')
plt.plot(df_boxcox.shift(365),label='bxcx.shift(365)')
plt.legend()

# %%
## acf and pcf of seasonal diff

# %%
##############################
# Plot ACF and PACF
##############################

y = df_seasonal_diff.values

fig = plt.figure(figsize=(18,18))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsaplots.plot_acf(y, lags=50, ax=ax1)
plt.xlim([-1, 50])
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsaplots.plot_pacf(y, lags=50, ax=ax2, **{'marker': '.'})
plt.xlim([-1, 50])
plt.show()


# %%
"""
## sesonal decomposition
"""

# %%
##############################
# Decompose the Transformed Data
##############################
decomp = sm.tsa.seasonal.seasonal_decompose(df_boxcox , model='additive', freq=365)

trend=decomp.trend
seasonal=decomp.seasonal
residual=decomp.resid

plt.figure(figsize=(20,10))

plt.subplot(411)
plt.plot(df_boxcox, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')

#plt.tight_layout()


# %%


# %%


# %%
"""
## Parameter Estimation and AIC Score
"""

# %%
%%time
##############################
# Build ARIMA (10,0,0)
##############################

model = sm.tsa.arima_model.ARIMA(df_boxcox, order=(4, 1, 0))
model_deseason_1 = model.fit()
print(model_deseason_1.summary())


# %%
results=model.fit(disp=-1)
plt.figure(figsize=(20,10))
plt.plot(residual)
plt.plot(results.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results.fittedvalues-residual)**2))
print('plotting ARIMA model')

# %%
results.fittedvalues[365:]


# %%
%%time
##############################
# Build SARIMA (1,1,0)x(1,0,0)s
##############################

model = sm.tsa.statespace.sarimax.SARIMAX(df_boxcox, order=(1,1,0), seasonal_order=(1,0,0,365))
model_4 = model.fit()
print(model_4.summary())


# %%
