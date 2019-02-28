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


def survival_preprocess(data, baseline_list, marker_list, truncate_time):
    """enhance marker_list features and make data into short format
    :param data: long format
    :param baseline_list: use tail(1), include event, event_time
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
        # data.groupby('id')[f].min().values,
        data.groupby('id')[f].max().values,
        data.groupby('id')[f].median().values,
        # data.groupby('id')[f].mean().values,
    ]) for f in marker_list]
    markers = np.transpose(np.concatenate(markers))

    enhanced_data = np.concatenate((baseline, markers), axis=1)
    col_names = [ii + jj for ii in marker_list for jj in ["_max", "_med"]]
    col_names = baseline_list + col_names
    return pd.DataFrame(enhanced_data, columns=col_names)
