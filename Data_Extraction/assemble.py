import pandas as pd
import os

os.chdir("/media/ojas/elements/AoT_Chicago.complete.2019-02-26")
required_sensors = [ "htu21d", "bmp180", "pr103j2", "tmp112", "tsys01"]

def load_csv_data(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

path_name = os.path.split(os.getcwd())
node_data = load_csv_data("nodes.csv")
sensor_data = load_csv_data("sensors.csv")

os.chdir("ojas")

temperature_array_1 = []
temperature_array_2 = []
humidity_array = []
pressure_array = []
filenames = os.listdir(os.getcwd())
for filename in filenames:
    if "htu21d" in filename:
        dataframe = load_csv_data(filename)
        dataframe = dataframe[dataframe["parameter"]== "humidity"]
        dataframe.rename(columns={'value_hrf': 'value_hrf_' + "humidity",
                                  'parameter': 'parameter_'+'humidity'}, inplace=True)
        humidity_array.append(dataframe)

    if "htu21d" in filename:
        dataframe = load_csv_data(filename)
        dataframe = dataframe[dataframe["parameter"] == "temperature"]
        dataframe.rename(columns={'value_hrf': 'value_hrf_' + "temperature_htu21d",
                                  'parameter': 'parameter_' + 'temperature_htu21d'}, inplace=True)
        temperature_array_1.append(dataframe)

    if "bmp180" in filename:
        dataframe = load_csv_data(filename)
        dataframe = dataframe[dataframe["parameter"]== "temperature"]
        dataframe.rename(columns={'value_hrf': 'value_hrf_' + "temperature_bmp180",
                                  'parameter': 'parameter_' + 'temperature_bmp180'}, inplace=True)
        temperature_array_2.append(dataframe)

    if "bmp180" in filename:
        dataframe = load_csv_data(filename)
        dataframe = dataframe[dataframe["parameter"]== "pressure"]
        dataframe.rename(columns={'value_hrf': 'value_hrf_' + "pressure",
                                  'parameter': 'parameter_' + 'temperature_bmp180'}, inplace=True)
        pressure_array.append(dataframe)

os.chdir("../results")
pd.concat(humidity_array).to_csv("humidity.csv")
pd.concat(pressure_array).to_csv("pressure.csv")
pd.concat(temperature_array_1).to_csv("temperature_htu21d.csv")
pd.concat(temperature_array_2).to_csv("temperature_bmp180.csv")


humidity = load_csv_data("humidity.csv").drop(['Unnamed: 0'],axis=1)
pressure = load_csv_data("pressure.csv").drop(['Unnamed: 0'],axis=1)
temperature_htu21d = load_csv_data("temperature_htu21d.csv").drop(['Unnamed: 0'],axis=1)
temperature_bmp180 = load_csv_data("temperature_bmp180.csv").drop(['Unnamed: 0'],axis=1)


def inner_join_on_data_node_frame(entire_data, node_data):

    result = pd.merge(entire_data,
                      node_data[['node_id', 'lat', 'lon']],
                      on='node_id',
                      how='inner')
    return result

def inner_join_on_data_sensor_frame(combined_data_frame, sensor_data):

    result = pd.merge(combined_data_frame,
                      sensor_data,
                      on=['timestamp', 'node_id'],
                      how='inner')
    return result

combined_frame = inner_join_on_data_sensor_frame(humidity, pressure)
combined_frame = inner_join_on_data_sensor_frame(combined_frame, temperature_bmp180)
combined_frame = inner_join_on_data_sensor_frame(combined_frame, temperature_htu21d)
combined_frame.timestamp = pd.to_datetime(combined_frame.timestamp)
#print(combined_frame.timestamp.dtype)
combined_frame.sort_values(by ="timestamp", axis=0, ascending=True, inplace=True)
combined_frame = inner_join_on_data_node_frame(combined_frame, node_data)
combined_frame.to_csv("combined.csv")
