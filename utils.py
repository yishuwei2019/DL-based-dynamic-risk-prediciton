import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence


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


# noinspection PyIncorrectDocstring
def id_loaders(ids, batch_size, shuffle=True):
    """return list id batches
    mimic torch.utils.data.dataloader

    :param ids: pd.series
    :return: list of array of size batch_size
    """
    if shuffle:
        ids = ids.sample(frac=1).values
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def prepare_seq(data, ids):
    """prepare packed sequence for rnn input

    :param data: long format, must have a column named id, rest are input features
    :param ids: list of ids
    :return:
    """
    dd = [data[data.id == ii] for ii in ids]
    dd = sorted(dd, key=lambda d: d.shape[0], reverse=True)
    dd = [torch.tensor(d.loc[:, d.columns != 'id'].values.tolist()) for d in dd]

    return pack_sequence(dd)


