import os
import torch
import numpy as np
import pandas as pd
from models import (
    DynamicDeepHit,
    loss_1
)
from data import (
    FILE_DIR,
    MARKERS
)
from utils import (
    train_test_split,
    id_loaders,
    prepare_label,
    prepare_seq,
)

# please do integer based
# it's assumed that all subjects experiences an event before end point
TARGET_START = 20
TARGET_END = 50
target_time = np.arange(TARGET_START, TARGET_END + 1)
CODE_POS = {
    '3': 0,  # stroke
    '4': 1,  # chf
    '5': 2,  # mi
}

data = pd.read_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
data = data[data.event_time > TARGET_START]
# round event time and record the index with respect to target_time
data.event_time = (data.event_time.round() - TARGET_START).astype('int32')
data_s = data[['id', 'event', 'event_time']].groupby('id').head(1).set_index('id')

batch_size = 20

train_set, test_set = train_test_split(data, .3)
train_ids = id_loaders(train_set.id, batch_size)
test_ids = id_loaders(test_set.id, batch_size)

ids_b, x = prepare_seq(data[MARKERS + ['id']], train_ids[0])
d_model = DynamicDeepHit(num_event=3, rnn_param=[7, 1, 1, 20], cs_param=[3, 3],
                         target_len=len(target_time))
marker_output, cs_output = d_model(x)
data_s_b = data_s.loc[ids_b, :]

label_b = prepare_label(data_s_b, CODE_POS, len(target_time))
l1 = loss_1(cs_output, label_b)
print(l1)

