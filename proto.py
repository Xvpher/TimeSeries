import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
#import spectrum
import statsmodels.api as sma
from statsmodels.tsa.stattools import pacf, adfuller
from scipy.stats import boxcox
from scipy import signal
# import warnings
# from datetime import datetime
# import pprint



def main():
    path = "/home/xvpher/TimeSeries/Data/sample1.csv"
    df = pd.read_csv(path)
    leads = df.columns
    X = df.iloc[0:5000][leads[1]]

    # i=0
    # fig, axs = plt.subplots(df.columns.size)
    # for lead in leads:
    #     X = df.iloc[1:5000][lead]
    #     axs[i].plot(X)
    #     i+=1
    # plt.show()

    peaks = signal.find_peaks(X, height=500)
    # mark = np.reshape(peaks[0], (-1,))
    plt.plot(X, '-bo',mfc='r',mec='g',ms=4, markevery=np.ndarray.tolist(peaks[0]))
    plt.show()



if __name__ == '__main__':
    main()
