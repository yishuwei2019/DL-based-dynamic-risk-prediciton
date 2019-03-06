import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


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
    ids = ids.unique().flatten()
    if shuffle:
        ids = np.random.choice(ids, len(ids), replace=False)
    ids_list = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    return ids_list[:-1]


# noinspection PyIncorrectDocstring
def prepare_seq(data, ids, feature_names, label_name):
    """prepare packed sequence for rnn input
        data must have a column of id

    :param data: long format, all ids must have at least 2 rows
    :param ids: list of ids
    :param feature_names: list of feature names
    :return:
    """
    dd = [data[data.id == ii] for ii in ids]
    dd = sorted(dd, key=lambda d: d.shape[0], reverse=True)
    feature = pack_sequence(
        [torch.tensor(d.loc[:, feature_names].values.tolist()[:-1]) for d in dd],
    )
    label = pack_sequence(
        [torch.tensor(d.loc[:, label_name].values[:-1], dtype=torch.float32) for d in dd]
    )
    label = pad_packed_sequence(label, batch_first=True, padding_value=0)[0]

    return feature, label


def param_change(param, model):
    """calculate model's parameter change since last time

    :param param: a deepcopy of original model parameter
    :return:
    """
    cc = 0
    for key, value in param.items():
        cc = torch.norm(torch.add(value, torch.neg(model.state_dict()[key]))) + cc
    return cc


def plot_loss(train_loss, test_loss):
    f, axes = plt.subplots(2, 1)
    axes[0].plot(train_loss, 'b')
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("train loss")
    axes[1].plot(test_loss, 'r')
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("test loss")
    plt.show()

