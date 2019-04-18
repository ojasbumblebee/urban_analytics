import pandas as pd
import os
import time

def load_csv_data(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

def inner_join_on_data_node_frame(entire_data, node_data):

    result = pd.merge(entire_data,
                      node_data[['node_id', 'lat', 'lon']],
                      on='node_id',
                      how='inner')
    return result


def inner_join_on_data_sensor_frame(combined_data_frame, sensor_data):

    result = pd.merge(combined_data_frame,
                      sensor_data[['sensor', 'parameter', 'hrf_minval', 'hrf_maxval']],
                      on=['sensor', 'parameter'],
                      how='inner')
    return result


def filter_corrupt_entries(combined_data_frame):
    # filter data values for corrupt data
    combined_data_frame = combined_data_frame[(combined_data_frame['value_hrf'].astype(float)
                                               >= combined_data_frame['hrf_minval'].astype(float)[0]) &
                                              (combined_data_frame['value_hrf'].astype(float)
                                               <= combined_data_frame['hrf_maxval'].astype(float)[0])]
    return combined_data_frame


def filter_data_for_sensor_and_parameter(combined_data_frame):
    # Filter the combined data frame for the desired sensor values
    combined_data_frame = combined_data_frame.loc[combined_data_frame['sensor'] == args.sensor]
    combined_data_frame = combined_data_frame.loc[combined_data_frame['parameter'] == args.parameter]
    combined_data_frame = combined_data_frame.dropna(how="all")
    return combined_data_frame

def adapter(current_data_chunk, sensor_data, node_data):
    #Call filtering on sensor and parameter
    current_data_chunk = filter_data_for_sensor_and_parameter(current_data_chunk)
    combined_data_frame = inner_join_on_data_node_frame(current_data_chunk, node_data)
    combined_data_frame = inner_join_on_data_sensor_frame(combined_data_frame, sensor_data)
    # Call filtering for sensor range and NaN values
    combined_data_frame = filter_corrupt_entries(combined_data_frame)
    return combined_data_frame


"""Here starts refactoring"""
os.chdir("/media/ojas/elements/AoT_Chicago.complete.2019-02-26")

required_sensors = [ "htu21d", "bmp180", "pr103j2", "tmp112", "tsys01"]

path_name = os.path.split(os.getcwd())
node_data = load_csv_data("nodes.csv")
sensor_data = load_csv_data("sensors.csv")

chunksize = 10 * 10 ** 6
i = 0
start = time.time()
for dataframe in pd.read_csv("data.csv", chunksize=chunksize):

    dataframe['timestamp'] = dataframe['timestamp'].astype('datetime64[h]')
    dataframe = dataframe.drop(['subsystem', 'value_raw'], axis=1)
    dataframe.set_index("sensor", inplace=True)

    for sensor_ in required_sensors:
        tmp = dataframe.loc[sensor_]
        tmp = tmp.dropna()
        tmp.value_hrf.astype(float)
        tmp["value_hrf"] = pd.to_numeric(tmp["value_hrf"])
        tmp = tmp.groupby(['timestamp', 'node_id', 'parameter'], as_index=False)["value_hrf"].mean()
        tmp.to_csv("ojas/output_{}_{}.csv".format(sensor_, i), index=False)

    end = time.time()
    print(end - start)
    print(i)
    start = time.time()
    i+=1

