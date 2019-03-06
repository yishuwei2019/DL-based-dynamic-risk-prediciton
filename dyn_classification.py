import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from common import *
from models import CNet
from preprocess import data_short_formatting
from utils import train_test_split, param_change, plot_loss
"""Treat dynamic risk prediction as a classification problem for a specific start time and horizon
"""


# noinspection PyShadowingNames
def train(batch_size=100):
    model.train()
    train_loss = []
    idx = np.random.permutation(x_train.shape[0])

    count = 0
    while count < len(idx) / batch_size:
        x = x_train[idx[count * batch_size: (count + 1) * batch_size], :]
        label = label_train[idx[count * batch_size: (count + 1) * batch_size]]
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, label)
        loss.backward()
        train_loss += [loss.data]
        optimizer.step()
        count += 1
        # if count % 10 == 0:
        #     print("train loss", sum(train_loss[:-9:-1]))
    return train_loss


# noinspection PyShadowingNames
def test(batch_size=200):
    model.eval()
    test_loss = []
    idx = np.random.permutation(x_test.shape[0])

    count = 0
    while count < x_test.shape[0] / batch_size:
        x = x_test[idx[count * batch_size: (count + 1) * batch_size], :]
        label = label_test[idx[count * batch_size: (count + 1) * batch_size]]
        output = model(x)
        loss = F.nll_loss(output, label)
        test_loss += [loss.data]
        count += 1
        # if count % 10 == 0:
        #     print("test_loss:", sum(test_loss[-9:-1]))
    return test_loss


if __name__ == "__main__":
    TRUNCATE_TIME = 10
    TARGET_END = 20
    data = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'data', 'data.pkl'))
    data = data[(data.ttocvd >= 0)]
    data['label'] = data.ttocvd <= TARGET_END
    data.label = data.label.astype('int32')
    data.loc[(data.ttocvd <= TARGET_END) & (data.cvd == 0), 'label'] = 2
    data = data_short_formatting(
        data, ['label'] + BASE_COVS + INDICATORS, MARKERS, TRUNCATE_TIME
    )
    FEATURE_LIST = data.columns[2:]

    d_in, h, d_out = 35, 64, 16
    model = CNet(35, 128, 64, 3)
    param = deepcopy(model.state_dict())

    batch_size = 200
    n_epochs = 30
    learning_rate = .001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=.1)
    train_set, test_set = train_test_split(data, .25)

    x_train, label_train = \
        torch.from_numpy(train_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor), \
        torch.from_numpy(train_set['label'].values).type(torch.LongTensor)
    x_test, label_test = \
        torch.from_numpy(test_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor), \
        torch.from_numpy(test_set['label'].values).type(torch.LongTensor)

    for epoch in range(n_epochs):
        print("*************** new epoch ******************")
        scheduler.step()
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
        train_loss = train(batch_size=batch_size)
        test_loss = test(batch_size=batch_size)

        print("train loss:", sum(train_loss) / len(train_loss))
        print("test loss:", sum(test_loss) / len(test_loss))
        print("parameter change:", param_change(param, model))
        param = deepcopy(model.state_dict())

        # plot_loss(train_loss, test_loss)

