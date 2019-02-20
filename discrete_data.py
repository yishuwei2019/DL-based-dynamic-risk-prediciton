import os
import numpy as np
import pandas as pd
from common import FILE_DIR

# please do integer based
# it's assumed that all subjects experiences an event before end point
TARGET_START = 20
TARGET_END = 50
target_time = np.arange(TARGET_START, TARGET_END + 1)
# event code to column position
CODE_POS = {
    '3': 0,  # stroke
    '4': 1,  # chf
    '5': 2,  # mi
}

data = pd.read_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
data = data[data.event_time > TARGET_START]
data = data[data.event_time <= TARGET_END]
# round event time and record the index with respect to target_time
data.event_time = (data.event_time.round() - TARGET_START).astype('int32')


