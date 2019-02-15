import numpy as np


def train_test_split(data, p=.3):
    """split data of long format into training and testing set

    :param data: long format, must have a column named id
    :param p: proportion of test sample
    :return: train_data, test_data
    """
    ids = data.id.unique()
    mask = np.random.binomial(1, p, len(ids))
    train_ids = ids[mask == 0]
    test_ids = ids[mask == 1]
    return data.loc[data.id.isin(train_ids)], data.loc[data.id.isin(test_ids)]


