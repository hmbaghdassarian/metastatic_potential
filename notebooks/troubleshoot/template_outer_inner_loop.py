#!/usr/bin/env python
# coding: utf-8

# Here we do nested 10-fold CV. We have an outter loop split into (train + validation, test)  where we have unseen test data. We then have an inner loop splitting the outter train + validation into (train, validation). In the inner loop we do the following:
# 
# 1. Feature selection: This is conducted only on the train data (to allow for generalization to validation during hyperparameter tuning). We use 5-fold elastic net CV on the train data, selecting those features that were retained 80% of the time across 100 iterations. CV chooses the best alpha, while we set the L1 ratio to 0.1 to help ensure that approximately 10-20% of the features are selected for. 
# 2. Hyperparameter tuning: The model is fit on the training data for the selected features, and assessed on the validation data. Across the inner 10-folds, we identify the set of hyperparameters that minimize the MSE. 
# 
# 

# In[46]:


import os
from multiprocessing import Pool
from collections import Counter

from tqdm import trange 

import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[47]:


seed = 42
data_path = '/nobackup/users/hmbaghda/metastatic_potential/'
res_all_fn = os.path.join(data_path, 'interim', 'iteration_all_res.json')


# In[48]:


n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# In[49]:


n_splits = 10

# feature selection elastic_net
n_alphas = 50 # fewer numbers is less accurate but faster computationally, default is 100
n_splits_elastic = 5 # fewer is faster but less rigorous
n_iter_elastic = int(1e2) # fewer is faster but less rigorous
feature_thresh = 0.8
par_feature = True # parallelization on feature selection


# In[50]:


def write_res(res_all):
    with open(res_all_fn, "w") as json_file:
        json.dump(res_all, json_file, indent=4)  

def elastic_net_iteration(X, y, seed_, n_splits, n_cores, n_alphas):
    """
    Perform a single iteration of ElasticNetCV for feature selection.

    Parameters:
        args (tuple): A tuple containing (X_train, y_train, random_seed).

    Returns:
        np.array: Binary mask indicating selected features.
    """
    elastic_net = ElasticNetCV(l1_ratio = 0.1, 
                               cv=n_splits, 
                               n_alphas = n_alphas, 
                               random_state=seed_, 
                               n_jobs = n_cores)
    elastic_net.fit(X, y.values.ravel())
    selected_features = np.where(elastic_net.coef_ != 0)[0]
    return selected_features


# In[51]:


y = pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential.csv'), index_col = 0)
X = pd.read_csv(os.path.join(data_path, 'processed', 'expr.csv'), index_col = 0).transpose()

if os.path.isfile(res_all_fn):
    with open(res_all_fn, 'r') as file:
        res_all = json.load(file)
else:
    res_all = {}


# In[52]:


# outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
# for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
#     res_all[outer_idx] = {}
#     X_outer_train, X_outer_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
#     y_outer_train, y_outer_test = y.iloc[train_idx,:], y.iloc[test_idx,:]

#     # set up inner loop
#     inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     inner_selected_features = []
#     inner_best_params = []
#     break


# In[53]:


# for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train)):
#     print('outer: {}, inner: {}'.format(outer_idx, inner_idx))
#     res_all[outer_idx][inner_idx] = {}
#     X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx,:], X_outer_train.iloc[inner_val_idx,:]
#     y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx,:], y_outer_train.iloc[inner_val_idx,:]
#     break


# In[45]:


# for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train)):
#     print('outer: {}, inner: {}'.format(outer_idx, inner_idx))
#     res_all[outer_idx][inner_idx] = {}
#     X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx,:], X_outer_train.iloc[inner_val_idx,:]
#     y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx,:], y_outer_train.iloc[inner_val_idx,:]

#     # FEATURE SELECTION - on inner train
#     if 'selected_features' not in res_all[outer_idx][inner_idx]:
#         ec_seeds = range(n_iter_elastic)
#         if not par_feature:
#             selected_features_res = []
#             for feature_iter in trange(n_iter_elastic):
#                 selected_features = elastic_net_iteration(X = X_inner_train, 
#                                                           y = y_inner_train, 
#                                                           seed_ = ec_seeds[feature_iter], 
#                                                           n_splits = n_splits_elastic,
#                                                           n_cores = n_cores, 
#                                                          n_alphas = n_alphas)
#                 selected_features_res.append(selected_features)
#         else:
#             pool = Pool(processes = min(n_cores, n_iter_elastic))
#             par_inputs = [(X_inner_train, y_inner_train, seed, n_splits_elastic, 1, n_alphas) for seed in ec_seeds]
#             selected_features_res = pool.starmap(elastic_net_iteration, par_inputs)
#             del par_inputs
#             pool.close()
#             pool.join()
#             gc.collect()


#         feature_counter = dict(zip(list(range(X_inner_train.shape[1])), [0]*X_inner_train.shape[1]))
#         for selected_features in selected_features_res:
#             for sf_idx in selected_features:
#                 feature_counter[sf_idx] += 1
#         selected_features = [k for k,v in feature_counter.items() if v >= (n_iter_elastic * feature_thresh)]
#         selected_features = X_inner_train.columns[selected_features].tolist()

#         res_all[outer_idx][inner_idx]['selected_features'] = selected_features


# In[43]:


# len(res_all[outer_idx][0]['selected_features'])


# In[44]:


# len(res_all[outer_idx][1]['selected_features'])


# In[ ]:


outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    res_all[outer_idx] = {}
    X_outer_train, X_outer_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_outer_train, y_outer_test = y.iloc[train_idx,:], y.iloc[test_idx,:]

    # set up inner loop
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_selected_features = []
    inner_best_params = []
    
    for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train)):
        print('outer: {}, inner: {}'.format(outer_idx, inner_idx))
        res_all[outer_idx][inner_idx] = {}
        X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx,:], X_outer_train.iloc[inner_val_idx,:]
        y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx,:], y_outer_train.iloc[inner_val_idx,:]
        
        # FEATURE SELECTION - on inner train
        if 'selected_features' not in res_all[outer_idx][inner_idx]:
            ec_seeds = range(n_iter_elastic)
            if not par_feature:
                selected_features_res = []
                for feature_iter in trange(n_iter_elastic):
                    selected_features = elastic_net_iteration(X = X_inner_train, 
                                                              y = y_inner_train, 
                                                              seed_ = ec_seeds[feature_iter], 
                                                              n_splits = n_splits,
                                                              n_cores = n_cores, 
                                                             n_alphas = n_alphas)
                    selected_features_res.append(selected_features)
            else:
                pool = Pool(processes = min(n_cores, n_iter_elastic))
                par_inputs = [(X_inner_train, y_inner_train, seed, n_splits, 1, n_alphas) for seed in ec_seeds]
                selected_features_res = pool.starmap(elastic_net_iteration, par_inputs)
                del par_inputs
                pool.close()
                pool.join()
                gc.collect()


            feature_counter = dict(zip(list(range(X_inner_train.shape[1])), [0]*X_inner_train.shape[1]))
            for selected_features in selected_features_res:
                for sf_idx in selected_features:
                    feature_counter[sf_idx] += 1
            selected_features = [k for k,v in feature_counter.items() if v >= (n_iter_elastic * feature_thresh)]
            selected_features = X_inner_train.columns[selected_features].tolist()

            res_all[outer_idx][inner_idx]['selected_features'] = selected_features
            write_res(res_all) # checkpoint 1
        # HYPERPARAMETER TUNING
