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
        data must have a column of id

    :param data: long format, must have a column named id, rest are input features
    :param ids: list of ids
    :return:
    """
    dd = [data[data.id == ii] for ii in ids]
    dd = sorted(dd, key=lambda d: d.shape[0], reverse=True)
    ids_n = [d.id.iloc[0] for d in dd]  # sort the ids to keep consistent with packed_sequence
    dd = pack_sequence(
        [torch.tensor(d.loc[:, d.columns != 'id'].values.tolist()) for d in dd]
    )

    return ids_n, dd


def prepare_label(data_s, code_pos, target_len):
    """ prepare label for this discrete survival time problem
        data_s needs columns names and format fixed

    :param data_s: batch_size * 2 (event, event_time)
        event_time is index instead of actual target time (start with 0)
    :param code_pos: dictionary {event_code: column index}
    :param target_len: length of target
    :return: tensor of batch_size * target_len * (event_num + 1)
    """
    label = torch.zeros(*[data_s.shape[0], target_len, len(code_pos) + 1])

    for ii in range(data_s.shape[0]):
        t = data_s.iloc[ii, :]

        if t.event != 0:
            label[ii][t.event_time][code_pos[str(t.event)]] = 1
        else:
            label[ii][t.event_time][-1] = 1  # censor case

    return label

