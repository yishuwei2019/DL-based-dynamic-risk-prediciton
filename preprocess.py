import numpy as np
import pandas as pd


def data_truncate(data, truncate_time):
    """truncate data so that:
        1. only subjects with event time > truncate_time left
    :param data: long format
    """
    label = data[data.time > truncate_time]
    data = data[data.id.isin(label.id)]
    return data


def data_short_formatting(data, baseline_list, marker_list, truncate_time):
    """This function has three purposes
    1. enhance marker_list features
    2. make data into short format
    3. change data type into float
    :param data: long format, must have a column 'id', 'time'
    :param baseline_list: use tail(1), include labels
    :param marker_list: continuous covariates, use summary statistics
    :param truncate_time: only observations before truncate_time will be used
    """
    # only consider subjects who survive up to truncate_time
    label = data[data.time > truncate_time]
    data = data[(data.id.isin(label.id)) & (data.time <= truncate_time)]
    if 'id' not in baseline_list:
        baseline_list = ['id'] + baseline_list
    baseline = data.groupby('id')[baseline_list].tail(1).values
    markers = [np.array([
        data.groupby('id')[f].min().values,
        data.groupby('id')[f].max().values,
        data.groupby('id')[f].median().values,
        data.groupby('id')[f].mean().values,
    ]) for f in marker_list]
    markers = np.transpose(np.concatenate(markers))

    enhanced_data = pd.DataFrame(
        np.concatenate((baseline, markers), axis=1),
        columns=baseline_list + [i + j for i in marker_list for j in ["_min", "_max", "_med", "_mean"]]
    )
    for col in enhanced_data.columns[1:]:
        enhanced_data[col] = enhanced_data[col].astype(float)
    return enhanced_data
