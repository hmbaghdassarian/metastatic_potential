#!/usr/bin/env python
# coding: utf-8

# In[47]:


import os
import itertools
import pandas as pd
import numpy as np
import json


import sys
sys.path.insert(1, './')
from BorutaShap import BorutaShap
from sklearn.ensemble import RandomForestRegressor


# emulating: https://medium.com/analytics-vidhya/is-this-the-best-feature-selection-algorithm-borutashap-8bc238aa1677

# In[50]:


seed = 42
n_cores = 30
data_path = '/nobackup/users/hmbaghda/metastatic_potential/'


# In[3]:


X_train_val = pd.read_csv(os.path.join(data_path, 'interim', 'X_train_val.csv'), index_col = 0)
y_train_val = pd.read_csv(os.path.join(data_path, 'interim', 'y_train_val.csv'), index_col = 0)


# In[41]:


res = {}
pvals = [0.1,0.05]
percentiles = [0.25, 0.5, 0.75, 1]

combs = list(itertools.product(pvals, percentiles))

for idx, comb in enumerate(combs):
    pval, percentile = comb
    default_rf_model = RandomForestRegressor(n_jobs = n_cores, 
                                             random_state = seed # gives variety per model fit
                                            )

    boruta_selector = BorutaShap(model = default_rf_model,
                                 importance_measure='shap',
                                 classification=False,
                                 percentile = percentile, 
                                pvalue = pval)

    boruta_selector.fit(X=X_train_val, 
                        y=y_train_val, 
                        n_trials=100,
                        random_state=seed,
                        train_or_test='train')
    res[idx] = {'pval': pval, 
           'percentile': percentile, 
           'selected_features': boruta_selector.accepted}
    with open(os.path.join(data_path, 'interim', 'depr_boruta_features.json'), "w") as json_file:
        json.dump(res, json_file, indent=4)  

