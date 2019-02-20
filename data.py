import pandas as pd
import os
from common import *

data = pd.read_csv(os.path.join(FILE_DIR, 'data', 'LRPP_updated.csv'), delimiter=',')
data = data.rename(columns={'ID_d': 'id'})

# code string variable to numeric
data.loc[:, ['RACE', 'EDU_G', 'COHORT']] = pd.DataFrame.from_dict({
    "RACE": pd.to_numeric(data.RACE.replace(RACE_CODE)),
    "EDU_G": pd.to_numeric(data.EDU_G.replace(EDUCATION_CODE)),
    "COHORT": pd.to_numeric(data.COHORT.replace(COHORT_CODE)),
})

data['delta'] = data.groupby('id')['age'].diff().fillna(0)  # delta time
data['time'] = data.groupby('id')['delta'].cumsum()  # time since come to study

# next marker in the sequence as response
for marker in MARKERS:
    data[marker + '_y'] = data.groupby('id')[marker].shift(-1)
data = data.fillna(method="ffill")

# frame into multiple event setting
# event: event code for the first incident / 0 for censoring
# event_time: event time for the first event / time of censoring
data_s = data[['id'] + MULTI_EVENTS + MULTI_TIMES].groupby('id').head(1)
data_s['event_time'] = data_s[MULTI_TIMES].min(axis='columns')
data_s['event'] = data_s[MULTI_EVENTS].any(axis='columns').astype('int64')
data_s.loc[data_s.event == 1, 'event'] = data_s.loc[
    data_s.event == 1, MULTI_TIMES].idxmin(
    axis='columns').map(EVENT_CODE)
data = data.merge(data_s[['id', 'event', 'event_time']], on='id', how='left')
data = data[data.time <= data.event_time]

data.to_pickle(os.path.join(FILE_DIR, 'data', 'data.pkl'))
