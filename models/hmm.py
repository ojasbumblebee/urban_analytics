import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go

from hmmlearn import hmm
from sklearn.model_selection import train_test_split


os.chdir("/media/ojas/elements/AoT_Chicago.complete.2019-02-26/results")

def load_csv_data(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

dataframe = load_csv_data("combined.csv")
dataframe.drop(['Unnamed: 0'], axis=1, inplace=True)


nodes = dataframe.node_id.unique()
dataframe = dataframe[dataframe["node_id"] == nodes[1]]

start = '2018-09-01 00:00:00'
end = '2019-02-25 12:00:00'
dataframe = dataframe[dataframe['timestamp'].between(start, end)]
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

train_data, test_data = train_test_split(
    dataframe["value_hrf_temperature_bmp180"], test_size=0.1, shuffle=False)

def reshape_data(data):
    X = list(data)
    #print(len(X))
    X = np.reshape(X, (-1, 1))
    return X

latency = 50
#ncomponents = [2,3,4,5,6,7,8,9,10]
ncomponents = [10]
possible_outcomes = [i for i in np.arange(-40, 30, 1)]
day_indexes = [i for i in range(len(test_data))]
print(possible_outcomes)
def generate_test_data(day_index):
    previous_data_start_index = max(0, day_index - latency)
    previous_data_end_index = max(0, day_index - 1)
    previous_data = test_data[previous_data_start_index: previous_data_end_index]
    return previous_data

y_values = []
x_values = [i for i in range(len(test_data))]
for ncomponent in ncomponents:
    model = hmm.GaussianHMM(n_components=ncomponent, covariance_type="full", n_iter=1000)
    train_data = reshape_data(train_data)
    model.fit(train_data)

    most_probable_outcome = []
    for day_index in day_indexes:
        outcome_score = []
        for possible_outcome in possible_outcomes:
            previous_data = generate_test_data(day_index)
            previous_data= reshape_data(previous_data)
            previous_data = np.row_stack((previous_data, possible_outcome))
            outcome_score.append(model.score(previous_data))
        most_probable_outcome.append(possible_outcomes[np.argmax(outcome_score)])

    y_values.append(most_probable_outcome)
    print(most_probable_outcome)

#plt.plot(x_values, y_values[0], y_values[1], y_values[2], list(test_data))
plt.plot(x_values, y_values[0], list(test_data))
plt.show()
print(list(test_data))
