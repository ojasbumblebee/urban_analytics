import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
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
    mode = 'markers',
    name = 'temperature bmp180'
)
trace1 = go.Scatter(
    x = dataframe['timestamp'],
    y = dataframe['value_hrf_pressure'],
    mode = 'lines+markers',
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
py.plot(data, filename='scatter-mode')

start = '2018-09-01 00:00:00'
end = '2019-02-25 12:00:00'
dataframe = dataframe[dataframe['timestamp'].between(start, end)]

dataframe.plot(x='timestamp', y='value_hrf_humidity')
plt.show()

dataframe['value_hrf_humidity'] = np.log(dataframe['value_hrf_humidity'])
dataframe.dropna(inplace=True)
dataframe['value_hrf_humidity'] = dataframe['value_hrf_humidity'] - dataframe['value_hrf_humidity'].shift(1)
dataframe.dropna(inplace=True)

dataframe.plot(x='timestamp', y='value_hrf_humidity')
plt.show()


dataframe.plot(x='timestamp', y='value_hrf_pressure')
plt.show()

dataframe['value_hrf_pressure'] = np.log(dataframe['value_hrf_pressure'])
dataframe['value_hrf_pressure'] = dataframe['value_hrf_pressure'] - dataframe['value_hrf_pressure'].shift(1)
dataframe.dropna(inplace=True)

dataframe.plot(x='timestamp', y='value_hrf_pressure')
plt.show()

dataframe.plot(x='timestamp', y='value_hrf_temperature_bmp180')
plt.show()

dataframe['value_hrf_temperature_bmp180'] = np.log(dataframe['value_hrf_temperature_bmp180'])
dataframe['value_hrf_temperature_bmp180'] = dataframe['value_hrf_temperature_bmp180'] - dataframe['value_hrf_temperature_bmp180'].shift(1)
dataframe.dropna(inplace=True)


dataframe.plot(x='timestamp', y='value_hrf_temperature_bmp180')
plt.show()


#dataframe.index = dataframe.timestamp

johan_test_temp = dataframe[["value_hrf_pressure", "value_hrf_humidity", "value_hrf_temperature_bmp180"]]
results = coint_johansen(johan_test_temp,-1,1)
print(results.eig)


jh_results = coint_johansen(johan_test_temp, 0, 1)             # 0 - constant term; 1 - log 1
print(jh_results.lr1)                           # dim = (n,) Trace statistic
print(jh_results.cvt)                           # dim = (n,3) critical value table (90%, 95%, 99%)
print(jh_results.evec)                          # dim = (n, n), columnwise eigen-vectors
v1 = jh_results.evec[:, 0]
v2 = jh_results.evec[:, 1]



train = johan_test_temp[:int(0.8*(len(johan_test_temp)))]
valid = johan_test_temp[int(0.8*(len(johan_test_temp))):]

model = VAR(endog=train)
print(model.select_order())

model_fit = model.fit(maxlags=15, ic="aic")
print("result lag order: ",model_fit.k_ar, model_fit.y)

#results = model_fit.forecast(johan_test_temp.values[-model_fit.k_ar:], 5)
#print(results)

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))
#converting predictions to dataframe
#results.forecast(data.values[-lag_order:], 5)

cols = johan_test_temp.columns

pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,3):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#print(pred)

for i in cols:
    print('rmse value for', i, 'is : ', mean_squared_error(pred[i], valid[i]))
