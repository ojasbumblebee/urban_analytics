import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import plotly.plotly as py
import plotly.graph_objs as go

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error


os.chdir("/media/ojas/elements/AoT_Chicago.complete.2019-02-26/results")

def load_csv_data(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

dataframe = load_csv_data("combined.csv")
dataframe.drop(['Unnamed: 0'], axis=1, inplace=True)

nodes = dataframe.node_id.unique()
dataframe = dataframe[dataframe["node_id"] == nodes[0]]
print(dataframe.head())

trace0 = go.Scatter(
    x = dataframe['timestamp'],
    y = dataframe['value_hrf_temperature_bmp180'],
    mode = 'lines',
    name = 'temperature bmp180'
)
trace1 = go.Scatter(
    x = dataframe['timestamp'],
    y = dataframe['value_hrf_pressure'],
    mode = 'lines',
    name = 'pressure'
)
trace2 = go.Scatter(
    x = dataframe['timestamp'],
    y = dataframe['value_hrf_humidity'],
    mode = 'lines',
    name = 'humidity'
)
trace3 = go.Scatter(
    x = dataframe['timestamp'],
    y = dataframe['value_hrf_temperature_htu21d'],
    mode = 'lines',
    name='temperature htu21d'
)
data = [trace0, trace1, trace2, trace3]
#py.plot(data, filename='scatter-mode')

start = '2018-09-01 00:00:00'
end = '2019-02-25 12:00:00'
dataframe = dataframe[dataframe['timestamp'].between(start, end)]

orignal_dataframe = dataframe
#dataframe.plot(x='timestamp', y='value_hrf_humidity')
#plt.show()

dataframe['value_hrf_humidity'] = np.log(dataframe['value_hrf_humidity'])
dataframe.dropna(inplace=True)
dataframe['value_hrf_humidity'] = dataframe['value_hrf_humidity'] - dataframe['value_hrf_humidity'].shift(1)
dataframe.dropna(inplace=True)

#dataframe.plot(x='timestamp', y='value_hrf_humidity')
#plt.show()

#dataframe.plot(x='timestamp', y='value_hrf_pressure')
#plt.show()

dataframe['value_hrf_pressure'] = np.log(dataframe['value_hrf_pressure'])
dataframe['value_hrf_pressure'] = dataframe['value_hrf_pressure'] - dataframe['value_hrf_pressure'].shift(1)
dataframe.dropna(inplace=True)

#dataframe.plot(x='timestamp', y='value_hrf_pressure')
#plt.show()

#dataframe.plot(x='timestamp', y='value_hrf_temperature_bmp180')
#plt.show()

dataframe['value_hrf_temperature_bmp180'] = np.log(dataframe['value_hrf_temperature_bmp180'])
dataframe['value_hrf_temperature_bmp180'] = dataframe['value_hrf_temperature_bmp180'] - dataframe['value_hrf_temperature_bmp180'].shift(1)
dataframe.dropna(inplace=True)


#dataframe.plot(x='timestamp', y='value_hrf_temperature_bmp180')
#plt.show()

#Pressure
lag_acf = acf(dataframe['value_hrf_pressure'], nlags=20)
lag_pacf = pacf(dataframe['value_hrf_pressure'], nlags=20, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataframe['value_hrf_pressure'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataframe['value_hrf_pressure'])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataframe['value_hrf_pressure'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataframe['value_hrf_pressure'])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

#Humidity
lag_acf = acf(dataframe['value_hrf_humidity'], nlags=20)
lag_pacf = pacf(dataframe['value_hrf_humidity'], nlags=20, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataframe['value_hrf_humidity'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataframe['value_hrf_humidity'])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataframe['value_hrf_humidity'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataframe['value_hrf_humidity'])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()


#Temoerature
lag_acf = acf(dataframe['value_hrf_temperature_bmp180'], nlags=20)
lag_pacf = pacf(dataframe['value_hrf_temperature_bmp180'], nlags=20, method='ols')
#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataframe['value_hrf_temperature_bmp180'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataframe['value_hrf_temperature_bmp180'])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dataframe['value_hrf_temperature_bmp180'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(dataframe['value_hrf_temperature_bmp180'])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()


dataframe.index = dataframe.timestamp

johan_test_temp = dataframe[["value_hrf_pressure", "value_hrf_humidity", "value_hrf_temperature_bmp180"]]
#johan_test_temp = dataframe[["value_hrf_pressure", "value_hrf_humidity"]]
orignal_dataframe = orignal_dataframe[["value_hrf_pressure", "value_hrf_humidity"]]

jh_results = coint_johansen(johan_test_temp, 0, 1)             # 0 - constant term; 1 - log 1
print(jh_results.lr1)                           # dim = (n,) Trace statistic
print(jh_results.cvt)                           # dim = (n,3) critical value table (90%, 95%, 99%)
print(jh_results.evec)
print(jh_results.eig)

train = johan_test_temp[:int(0.8*(len(johan_test_temp)))]
valid = johan_test_temp[int(0.8*(len(johan_test_temp))):]

train_orignal = orignal_dataframe[:int(0.8*(len(orignal_dataframe)))]
valid_orignal = orignal_dataframe[int(0.8*(len(orignal_dataframe))):]


order = [2,3,4,5,6]
for i in order:

    model = VARMAX(train, order=(i,0), trend='c')
    model_result = model.fit(maxiter= 1000)
    print(model_result.summary())
    model_result.plot_diagnostics(variable=0)
    plt.show()
    model_result.plot_diagnostics(variable=1)
    plt.show()
    model_result.plot_diagnostics(variable=2)
    plt.show()
    """
    VAR_forecast_value_hrf_pressure = np.exp(train["value_hrf_pressure"]) * train_orignal['value_hrf_pressure'][-2:]
    VAR_forecast_value_hrf_humidity = np.exp(train["value_hrf_humidity"]) * train_orignal['value_hrf_humidity'][-2:]
    #VAR_forecast_value_hrf_pressure = np.exp(train["value_hrf_temperature_bmp180"]) * train['value_hrf_temperature_bmp180'][-2:]

    rmse_value_hrf_pressure = math.sqrt(mean_squared_error(train_orignal['value_hrf_pressure'][-2:], VAR_forecast_value_hrf_pressure))
    rmse_value_hrf_humidity = math.sqrt(mean_squared_error(train_orignal['value_hrf_humidity'][-2:], VAR_forecast_value_hrf_humidity))
    print(rmse_value_hrf_pressure)
    print(rmse_value_hrf_humidity)
    """
