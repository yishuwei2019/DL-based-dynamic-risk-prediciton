# noinspection SpellCheckingInspection
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.nn.utils.rnn import pack_sequence
from common import *
from models import CoxPH
from loss import coxph_logparlk, sigmoid_concordance_loss, c_index
from utils import (
    train_test_split,
    id_loaders,
    prepare_seq,
)

TARGET_START = 1
TARGET_END = 61
FEATURE_LIST = MARKERS + COVS + BASE_COVS
FILE_DIR = os.path.dirname(__file__)
data = pd.read_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
# this part is very problematic!!
data.event_time = data.event_time.round().astype('int32')
data = data[data.event_time.isin(range(TARGET_START, TARGET_END))]
data.loc[:, FEATURE_LIST] = (
    data.loc[:, FEATURE_LIST] - data.loc[:, FEATURE_LIST].mean()
) / data.loc[:, FEATURE_LIST].std()
data.event = data.event.apply(func=lambda x: 1 if x > 0 else 0)  # combine all events into one


# noinspection PyShadowingNames
def train(model, train_set, batch_size=20):
    train_loss = []
    train_ids = id_loaders(train_set.id, batch_size)
    print("train batch number:", len(train_ids))

    count = 0
    for ids in train_ids:
        feature_b = torch.tensor([data.loc[data.id == ii, FEATURE_LIST].mean(axis=0) for ii in ids])
        hazard_ratio, preds = model(feature_b)
        event = torch.tensor([data.loc[data.id == ii, 'event'].tolist()[-1] for ii in ids])
        event_time = torch.tensor([data.loc[data.id == ii, 'event_time'].tolist()[-1] for ii in ids])

        l1 = coxph_logparlk(event_time, event, hazard_ratio)
        l2 = sigmoid_concordance_loss(event_time, event, preds)
        loss = torch.add(l1, 10 * l2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.tolist()
        count += 1
        if count % 10 == 0:
            print("10 batches trained:", sum(train_loss[-9:-1]))
            # print(model.state_dict())
    return train_loss


# noinspection PyShadowingNames
def test(model, test_set, batch_size=20):
    test_loss = []
    test_ids = id_loaders(test_set.id, batch_size)
    print("test batch number:", len(test_ids))

    for ids in test_ids:
        feature_b, marker_b = prepare_seq(
            data=data,
            ids=ids,
            feature_names=FEATURE_LIST,
            label_name='SBP_y'
        )
        marker_output = model(feature_b)
        marker_b = marker_b.contiguous().view(-1)
        # delete the padding part
        marker_output = marker_output.contiguous().view(-1)[torch.nonzero(marker_b)]
        marker_b = marker_b[torch.nonzero(marker_b)]
        test_loss += [mse_loss(marker_output, marker_b).tolist()]
        # print(c_index(event_time, event, hazard_ratio))

    return test_loss


if __name__ == '__main__':
    batch_size = 100
    n_epochs = 10
    learning_rate = .01
    model = CoxPH(
        input_size=len(FEATURE_LIST),
        size_1=128,
        output_size=5,
        n_time_units=TARGET_END - TARGET_START,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=.1)
    train_set, test_set = train_test_split(data, .3)

    for epoch in range(n_epochs):
        if epoch < 5:
            scheduler.step()
        print("*************** new epoch ******************")
        for param_group in optimizer.param_groups:
            print("learning rate:")
            print(param_group['lr'])
            train_loss = train(model, train_set, batch_size=batch_size)
            # test_loss = test(model, test_set, batch_size=batch_size)
            print("test loss:")
            # print(test_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(FILE_DIR, 'longitudinal_model.pth'))
        print("********************************************")

