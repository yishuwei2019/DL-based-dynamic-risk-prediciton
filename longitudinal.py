# noinspection SpellCheckingInspection
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from common import *
from models import LongRNN
from utils import (
    train_test_split,
    id_loaders,
    prepare_seq,
    plot_loss,
)

TARGET_START = 20
TARGET_END = 50
FEATURE_LIST = MARKERS + INDICATORS + BASE_COVS
FILE_DIR = os.path.dirname(__file__)
data = pd.read_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
data = data[data.event_time > TARGET_START]
data = data[data.event_time <= TARGET_END]
data.loc[:, FEATURE_LIST] = (
    data.loc[:, FEATURE_LIST] - data.loc[:, FEATURE_LIST].mean()
) / data.loc[:, FEATURE_LIST].std()
data.SBP_y = (data.SBP_y - data.SBP_y.mean()) / data.SBP_y.std()


# noinspection PyShadowingNames
def train(model, train_set, batch_size=20):
    train_loss = []
    train_ids = id_loaders(train_set.id, batch_size)
    print("train batch number:", len(train_ids))

    count = 0
    for ids in train_ids:
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

        optimizer.zero_grad()
        loss = mse_loss(marker_output, marker_b)
        loss.backward()
        optimizer.step()

        train_loss += [loss.tolist()]
        count += 1

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

    return test_loss


if __name__ == '__main__':
    batch_size = 50
    n_epochs = 10
    learning_rate = .01
    model = LongRNN(
        input_size=len(FEATURE_LIST),
        hidden_size=3,
        num_layers=1,
        batch_size=batch_size,
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
            test_loss = test(model, test_set, batch_size=batch_size)
            plot_loss(train_loss, test_loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(FILE_DIR, 'longitudinal_model.pth'))
        print("********************************************")

