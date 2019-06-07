import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch

from common import *
from loss import auc_jm
from preprocess import data_short_formatting
from utils import train_test_split
"""Simple logistic regression
"""

if __name__ == "__main__":
    TRUNCATE_TIME = 10
    TARGET_END = 40
    data = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'data', 'data.pkl'))
    data = data[(data.ttocvd >= 0)]

    # label = 0: alive after TARGET_END, 1: dead before TARGET_END, 2: censored before TARGET_END
    data['label'] = data.ttocvd <= TARGET_END
    data.label = data.label.astype('int32')
    # delete unclear subjects
    data = data.loc[(data.ttocvd > TARGET_END) | (data.cvd == 1), :]
    # data.loc[(data.ttocvd <= TARGET_END) & (data.cvd == 0), 'label'] = 2

    data = data_short_formatting(
        data, ['label', 'cvd', 'ttocvd'] + BASE_COVS + INDICATORS, MARKERS, TRUNCATE_TIME
    )
    FEATURE_LIST = data.columns[4:-3]
    train_set, test_set = train_test_split(data, .3)

    clf = LogisticRegression(solver='lbfgs').fit(
        train_set.loc[:, FEATURE_LIST].values,
        train_set.loc[:, 'label'].values
    )

    result = np.vstack(
        (clf.predict_proba(test_set.loc[:, FEATURE_LIST].values)[:, 1], test_set.loc[:, 'label'])
    ).T
    print("death ratio", sum(test_set.loc[:, 'label'] == 1) / test_set.shape[0])

    print("accuracy", clf.score(test_set.loc[:, FEATURE_LIST].values,
                    test_set.loc[:, 'label'].values))

    print("aucJM", auc_jm(
        torch.Tensor(test_set.loc[:, 'label'].values),
        torch.Tensor(test_set.loc[:, 'ttocvd'].values),
        torch.Tensor(clf.predict_proba(test_set.loc[:, FEATURE_LIST].values)[:, 1]),
        TARGET_END
    ))








