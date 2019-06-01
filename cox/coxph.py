import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from copy import deepcopy
from common import *
from models import SurvDl
from preprocess import data_short_formatting
from loss_original import log_parlik
from loss import coxph_logparlk, c_index, auc_jm
from utils import train_test_split, param_change, plot_loss
"""use cox proportional hazard model to predict hazard and thus c-index 
1. maximizing partial likelihood 
2. constant hazard
"""


# noinspection PyShadowingNames
def train(batch_size=100):
    model.train()
    train_loss = []
    idx = np.random.permutation(x_train.shape[0])

    count = 0
    while count < len(idx) / batch_size:
        x = x_train[idx[count * batch_size: (count + 1) * batch_size], :]
        lifetime = lifetime_train[idx[count * batch_size: (count + 1) * batch_size]]
        censor = censor_train[idx[count * batch_size: (count + 1) * batch_size]]
        optimizer.zero_grad()
        score1, score2 = model(x)
        loss = log_parlik(lifetime, censor, score1)
        # loss = coxph_logparlk(lifetime, censor, score1)
        # loss2 = sum(
        #     [rank_loss(lifetime, censor, score2, t + 1, time_bin) for t in range(num_time_units)])
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
        lifetime = lifetime_test[idx[count * batch_size: (count + 1) * batch_size]]
        censor = censor_test[idx[count * batch_size: (count + 1) * batch_size]]
        score1, score2 = model(x)
        loss = log_parlik(lifetime, censor, score1)
        # loss = coxph_logparlk(lifetime, censor, score1)
        # loss2 = sum(
        #     [rank_loss(lifetime, censor, score2, t + 1, time_bin) for t in range(num_time_units)])
        test_loss += [loss.data]
        count += 1
        # if count % 10 == 0:
        #     print("test_loss:", sum(test_loss[-9:-1]))
    return test_loss


if __name__ == '__main__':
    TRUNCATE_TIME = 10
    TARGET_END = 30
    data = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'data', 'data.pkl'))
    data.ttocvd = data.ttocvd.round() - 1
    data = data_short_formatting(
        data,
        ['cvd', 'ttocvd'] + BASE_COVS + INDICATORS,
        MARKERS,
        TRUNCATE_TIME
    )
    # data = data[(data.ttocvd >= 0) & (data.ttocvd < TARGET_END)]
    FEATURE_LIST = data.columns[3:-3]

    d_in, h, d_out = 32, 64, 16
    time_bin, num_time_units = 30, 24  # 24 months
    model = SurvDl(d_in, h, d_out, num_time_units)
    param = deepcopy(model.state_dict())

    batch_size = 200
    n_epochs = 20
    learning_rate = .001  # should be small
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 4, gamma=.1)
    train_set, test_set = train_test_split(data, .3)

    x_train, lifetime_train, censor_train = \
        torch.from_numpy(train_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor), \
        torch.from_numpy(train_set['ttocvd'].values).type(torch.IntTensor),\
        torch.from_numpy(train_set['cvd'].values).type(torch.IntTensor)
    x_test, lifetime_test, censor_test = \
        torch.from_numpy(test_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor), \
        torch.from_numpy(test_set['ttocvd'].values).type(torch.IntTensor),\
        torch.from_numpy(test_set['cvd'].values).type(torch.IntTensor)

    train_loss = []
    test_loss = []
    for epoch in range(n_epochs):
        print("*************** new epoch ******************")
        score1_total, _ = model(x_test)
        loss_total = log_parlik(lifetime_test, censor_test, score1_total)

        # print("ten year auc", auc_jm(censor_test, lifetime_test, score1_total, 10))
        # print("total loss in test sample:", loss_total)
        cindex = c_index(censor_test, lifetime_test, score1_total)
        print("cindex in test sample:", cindex)

        if epoch < 15:
            scheduler.step()
        for param_group in optimizer.param_groups:
            print("learning rate:", param_group['lr'])
        train_loss = train_loss + train(batch_size=batch_size)
        test_loss = test_loss + test(batch_size=batch_size)

        # print("train loss:", sum(train_loss) / len(train_loss))
        # print("test loss:", sum(test_loss) / len(test_loss))
        print("parameter change:", param_change(param, model))
        param = deepcopy(model.state_dict())

    plot_loss(train_loss, test_loss)




