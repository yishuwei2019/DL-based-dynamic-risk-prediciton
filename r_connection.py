import os
import numpy as np
import pandas as pd
import torch
from loss import coxph_logparlk, c_index, auc_jm

"""This file collects results from R outputs
"""

DIR = '/users/yishu/documents/dl_medical/Rcode/data'

# cox_simple_res = pd.read_csv(os.path.join(DIR, 'cox_simple_res.csv'), delimiter=',')
# cox_complex_res = pd.read_csv(os.path.join(DIR, 'cox_complex_res.csv'), delimiter=',')
# print(c_index(
#     torch.tensor(cox_simple_res.event),
#     torch.tensor(round(cox_simple_res.event_time)),
#     torch.tensor(cox_simple_res.hazard_ratio)
# ))
# print(c_index(
#     torch.tensor(cox_complex_res.event),
#     torch.tensor(round(cox_complex_res.event_time)),
#     torch.tensor(cox_complex_res.hazard_ratio)
# ))


jm_simple_res = pd.read_csv(os.path.join(DIR, 'jm_simple_res.csv'), delimiter=',')
jm_complex_res = pd.read_csv(os.path.join(DIR, 'jm_complex_res.csv'), delimiter=',')
for horizon in [20, 25, 30]:
    print("horizon is", horizon)
    print(
        "simple",
        # fitted result from JM package is survival probability
        1 - auc_jm(
            torch.tensor(jm_simple_res.event),
            torch.tensor(round(jm_simple_res.event_time)),
            torch.tensor(jm_simple_res[str(horizon)]),
            horizon
        )
    )
    print(
        "complex",
        1 - auc_jm(
            torch.tensor(jm_complex_res.event),
            torch.tensor(round(jm_complex_res.event_time)),
            torch.tensor(jm_complex_res[str(horizon)]),
            horizon
        )
    )