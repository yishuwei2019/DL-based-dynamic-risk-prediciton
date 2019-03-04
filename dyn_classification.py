import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from common import *
from models import SurvDl
from preprocess import data_short_formatting
from loss_original import (
    c_index,
    log_parlik,
    rank_loss,
)
from loss import coxph_logparlk
from utils import train_test_split, param_change
"""Treat dynamic risk prediction as a classification problem for a specific start time and horizon
"""

if __name__ == "__main__":
    TRUNCATE_TIME = 10
    TARGET_END = 30
    data = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'data', 'data.pkl'))
    data.event_time = data.ttocvd.round() - 1
    data.event = data.cvd
    data = data_short_formatting(data, ['event', 'event_time'] + BASE_COVS + INDICATORS, MARKERS,
                                 TRUNCATE_TIME)
    data = data[(data.event_time >= 0) & (data.event_time < TARGET_END)]
    FEATURE_LIST = data.columns[3:]

    d_in, h, d_out = 35, 64, 16
    batch_size = 50
    num_time_units = 24  # 24 months
    time_bin = 30
    n_epochs = 20
    learning_rate = .1
    model = SurvDl(d_in, h, d_out, num_time_units)
    param = deepcopy(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=.1)
    train_set, test_set = train_test_split(data, .25)

    x_train = torch.from_numpy(train_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor)
    lifetime_train = torch.from_numpy(train_set.event_time.values).type(torch.IntTensor)
    censor_train = torch.from_numpy(train_set.event.values).type(torch.IntTensor)
    x_test = torch.from_numpy(test_set.loc[:, FEATURE_LIST].values).type(torch.FloatTensor)
    lifetime_test = torch.from_numpy(test_set.event_time.values).type(torch.IntTensor)
    censor_test = torch.from_numpy(test_set.event.values).type(torch.IntTensor)