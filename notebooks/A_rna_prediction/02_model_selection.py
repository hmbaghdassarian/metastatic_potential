#!/usr/bin/env python
# coding: utf-8

# In notebook 02A we did a preliminary model search to see which models performed best. We saw that overall, linear models seemed to perform atleast as well if not better than nonlinear models. For example, in SVR gridsearch, the linear kernel was consistently chosen. Furthermore, elasticNet, linear SVM, and PLSR were the top 3 performing models, whereas random forest, KNN, and xgboost were the bottom 3. 
# 
# Here, we comprehensively address whether linear models perform atleast as well as nonlinear ones for this specific prediction task as follows: we do a separate comprehensive optuna hyperparameter tuning (inner 5-fold CV) across the same set of 10-folds for each of the following models separately: 
# 
# - Linear models: PLSR, Ridge, Lasso, ElasticNet with a 0.5 L1 ratio, SVM with a linear kernel
# - Nonlinear models: SVM with a poly kernel, SVM with a rbf kernel, KNN, XGBoost, and an ensemble of neural networks. 
# 
# We then compare the performance as measured by the Pearson correlation of the selected best hyperparameter for each fold across models. 

# In[1]:


import os
import pickle
import pathlib

from tqdm import tqdm

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import CmaEsSampler, TPESampler, RandomSampler
from optuna.distributions import CategoricalDistribution

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import pearsonr
from sklearn.utils import shuffle

import xgboost as XGB

import sys
sys.path.insert(1, '../')
from utils import write_pickled_object
from utils import (MeanCenterer, HybridSampler, RandomTPESampler, pearson_corr_scorer,
                   PLSRegression_X)
from utils import RNAFeatureSelector as FeatureSelector


# In[2]:


data_path = '/home/hmbaghda/orcd/pool/metastatic_potential/'
random_state = 888

# n_cores = 64
# os.environ["OMP_NUM_THREADS"] = str(n_cores)
# os.environ["MKL_NUM_THREADS"] = str(n_cores)
# os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
# os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
# os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)

from threadpoolctl import threadpool_limits
n_cores = 64
for v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[v] = "1" # 1 thread per process; CV handles parallelism



# In[106]:


def optuna_objective(trial, X, y, inner_cv, n_cores, random_state, model_type):
    # Define feature reduction/selection method

    steps = [
        ("feature_reduction", FeatureSelector(method="top_residuals", 
                                              n_features=trial.suggest_categorical("FeatureSelector__n_features", [250, 500, 1000, 5000, X.shape[1]]))),
        ("mean_centering", MeanCenterer()),
    ]


    # Define model
    if model_type == "SVR_linear":
        steps.append(("model", SVR(
            kernel='linear',
            C=trial.suggest_float(model_type + "__C", 1e-4, 1e2, log = True),
            epsilon=trial.suggest_float(model_type + "__epsilon", 1e-3, 10, log=True)
        )))
    elif model_type == 'PLS':
        steps.append(
            ("model", PLSRegression_X(n_components=trial.suggest_int(model_type + "__n_components", 2, 100, step = 1))), 
        )
    elif model_type == 'Ridge':
        steps.append(
            ('model', Ridge(alpha=trial.suggest_float(model_type + "__alpha", 1e-3, 1e2, log = True), 
                                             random_state=random_state))
        )
    elif model_type == 'Lasso':
        steps.append(
            ('model', Lasso(alpha=trial.suggest_float(model_type + "__alpha", 1e-3, 1e2, log = True), 
                                             random_state=random_state))
        )
    elif model_type == 'ElasticNet':
        steps.append(
            ('model', ElasticNet(alpha=trial.suggest_float(model_type + "__alpha", 1e-3, 1e2, log = True),
                                 random_state=random_state, 
                                l1_ratio = trial.suggest_float(model_type + "__l1_ratio", 0.3, 0.7, step = 0.1)))
        )
    elif model_type == "SVR_poly":
        steps.append(("model", SVR(
            kernel='poly',
            C=trial.suggest_float(model_type + "__C", 1e-4, 1e2, log = True),
            epsilon=trial.suggest_float(model_type + "__epsilon", 1e-3, 10, log=True),
            degree=trial.suggest_int(model_type + "__degree", 2, 5, step=1),
            coef0=trial.suggest_float(model_type + "__coef0", 0, 2, step=0.1), 
            gamma=trial.suggest_categorical(model_type + "__gamma", ['scale', 'auto'])
        )))
    elif model_type == "SVR_rbf":
        steps.append(("model", SVR(
            kernel='rbf',
            C=trial.suggest_float(model_type + "__C", 1e-4, 1e2, log = True),
            epsilon=trial.suggest_float(model_type + "__epsilon", 1e-3, 10, log=True),
            gamma=trial.suggest_categorical(model_type + "__gamma", ['scale', 'auto'])
        )))
    elif model_type == "RFR":
        steps.append(("model", RandomForestRegressor(
            n_estimators=trial.suggest_int(model_type + "__n_estimators", 300, 1600, step=400),
            max_features=trial.suggest_categorical(model_type + "__max_features", ["sqrt", "log2", 0.5, 0.75, 1]),
            max_samples=trial.suggest_categorical(model_type + "__max_samples", [0.25, 0.5, 0.75, None]),
            max_depth=trial.suggest_categorical(model_type + "__max_depth", [None, 10, 25, 50, 100, 200]),
            random_state=random_state,
            n_jobs=n_cores
        )))
    elif model_type == "XGBoost":
        steps.append(("model", XGB.XGBRegressor(
            n_estimators=trial.suggest_int(model_type + "__n_estimators", 300, 1600, step=400),
            max_depth=trial.suggest_categorical(model_type + "__max_depth", [10, 25, 50, 100, 200]),
            learning_rate=trial.suggest_float(model_type + "__learning_rate", 1e-3, 1, log=True),
            subsample=trial.suggest_float(model_type + "__subsample", 0.25, 1.0, step=0.05),
            reg_alpha=trial.suggest_float(model_type + "__reg_alpha", 0, 10, step=0.1),
            reg_lambda=trial.suggest_float(model_type + "__reg_lambda", 0, 10, step=0.1),
            random_state=random_state,
            n_jobs=n_cores
        )))
    elif model_type == 'KNN':
        steps.append(("model",  KNeighborsRegressor(
            n_neighbors=trial.suggest_int(model_type + "__n_neighbors", 15, 25, step=1), 
            weights=trial.suggest_categorical(model_type + "__weights", ['uniform', 'distance']),
            metric=trial.suggest_categorical(model_type + "__metric", ['minkowski', 'l1', 'l2', 'cosine']),
            n_jobs = n_cores)))

    # Create the pipeline
    pipeline = Pipeline(steps)

    # Evaluate with cross-validation
    with threadpool_limits(1):  # 1 BLAS thread per worker
        mse = -cross_val_score(pipeline, X, y, 
                               cv=inner_cv, 
                               scoring="neg_mean_squared_error", 
                               n_jobs=inner_cv.n_splits).mean()

    return mse


def generate_best_pipeline(study):
    best_params = study.best_params
    steps = []
    steps.append(("feature_reduction", FeatureSelector(method="top_residuals", n_features=best_params["FeatureSelector__n_features"])))
    steps.append(("mean_centering", MeanCenterer()))

    if model_type == 'SVR_linear':
        steps.append(("model", SVR(
            kernel='linear',
            C=best_params[model_type + "__C"],
            epsilon=best_params[model_type + '__epsilon']
        )))
    elif model_type == 'PLS':
        steps.append(
            ("model", PLSRegression_X(n_components=best_params[model_type + '__n_components'])), 
        )
    elif model_type == 'Ridge':
        steps.append(
            ('model', Ridge(alpha=best_params[model_type + '__alpha'], 
                                             random_state=random_state))
        )
    elif model_type == 'Lasso':
        steps.append(
            ('model', Lasso(alpha=best_params[model_type + '__alpha'], 
                                             random_state=random_state))
        )
    elif model_type == 'ElasticNet':
        steps.append(
            ('model', ElasticNet(alpha=best_params[model_type + '__alpha'],
                                 random_state=random_state, 
                                l1_ratio = best_params[model_type + '__l1_ratio']))
        )
    elif model_type == "SVR_poly":
        steps.append(("model", SVR(
            kernel='poly',
            C=best_params[model_type + '__C'],
            epsilon=best_params[model_type + '__epsilon'],
            degree=best_params[model_type + '__degree'],
            coef0=best_params[model_type + '__coef0'], 
            gamma=best_params[model_type + '__gamma']
        )))
    elif model_type == "SVR_rbf":
        steps.append(("model", SVR(
            kernel='rbf',
            C=best_params[model_type + '__C'],
            epsilon=best_params[model_type + '__epsilon'],
            gamma=best_params[model_type + '__gamma']
        )))
    elif model_type == "RFR":
        steps.append(("model", RandomForestRegressor(
            n_estimators=best_params[model_type + '__n_estimators'],
            max_features=best_params[model_type + '__max_features'],
            max_samples=best_params[model_type + '__max_samples'],
            max_depth=best_params[model_type + '__max_depth'],
            random_state=random_state,
            n_jobs=int(n_cores/inner_cv.n_splits)
        )))
    elif model_type == "XGBoost":
        steps.append(("model", XGB.XGBRegressor(
            n_estimators=best_params[model_type + '__n_estimators'],
            max_depth=best_params[model_type + '__max_depth'],
            learning_rate=best_params[model_type + '__learning_rate'],
            subsample=best_params[model_type + '__subsample'],
            reg_alpha=best_params[model_type + '__reg_alpha'],
            reg_lambda=best_params[model_type + '__reg_lambda'],
            random_state=random_state,
            n_jobs=int(n_cores/inner_cv.n_splits)
        )))
    elif model_type == 'KNN':
        steps.append(("model",  KNeighborsRegressor(
            n_neighbors=best_params[model_type + '__n_neighbors'], 
            weights=best_params[model_type + '__weights'],
            metric=best_params[model_type + '__metric'],
            n_jobs = int(n_cores/inner_cv.n_splits)
        )))
    best_pipeline = Pipeline(steps)
    return best_pipeline


# In[107]:


X = pd.read_csv(os.path.join(data_path, 'processed',  'expr.csv'), index_col = 0).values
y = pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential.csv'), index_col = 0)['mean'].values.ravel()


# In[3]:


outer_folds=10
inner_folds=5
n_trials = 100


# In[110]:


cmaes_sampler = CmaEsSampler(seed=random_state, 
                             warn_independent_sampling=False, 
                            restart_strategy='bipop')

exploration_sampler = RandomSampler(seed=random_state)
tpe_sampler = RandomTPESampler(seed=random_state, 
                               n_startup_trials = 15,
                               exploration_sampler = exploration_sampler, 
                               exploration_freq=20 # randomly sample every n trials
                              )
# tpe_sampler = TPESampler(seed=random_state, 
#                         n_startup_trials = 20)


# In[114]:


outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

if os.path.isfile(os.path.join(data_path, 'interim', 'pipeline_model_selection_transcriptomics_individual.csv')):
    res_df = pd.read_csv(os.path.join(data_path, 'interim', 'pipeline_model_selection_transcriptomics_individual.csv'), 
                     index_col = 0)
    results = res_df.to_dict(orient='records')
else:
    results = []
    res_df = None

for model_type in ['SVR_linear', 'PLS', 'Ridge', 'Lasso', 'ElasticNet', 
                   'SVR_poly', 'SVR_rbf', 'RFR', 'KNN']: #'XGBoost', 
    for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        if res_df is not None and res_df[(res_df.fold == k) & (res_df.model_type == model_type)].shape[0] != 0:
            pass
        else:
            print(model_type + ': ' + str(k))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]


            pruner = optuna.pruners.SuccessiveHalvingPruner()
            study = optuna.create_study(direction="minimize", 
                                        sampler=HybridSampler(primary_sampler=cmaes_sampler, fallback_sampler=tpe_sampler), 
                                       pruner = pruner, 
                                       study_name = '{}_optuna'.format(k))
            study.optimize(
                lambda trial: optuna_objective(trial, X_train, y_train, inner_cv, n_cores, random_state, model_type),
                n_trials=n_trials, 
                catch=(ValueError,)
            )
    #         write_pickled_object(study, os.path.join(data_path, 'interim', study.study_name + '.pickle'))

            best_pipeline = generate_best_pipeline(study)
            best_pipeline.fit(X_train, y_train)

            y_train_pred = best_pipeline.predict(X_train)
            y_test_pred = best_pipeline.predict(X_test)

            train_corr = pearsonr(y_train, y_train_pred)[0]
            test_corr = pearsonr(y_test, y_test_pred)[0]
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)

            results.append({
                "model_type": model_type,
                "fold": k,
                "train_corr": train_corr,
                "test_corr": test_corr,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "best_params": study.best_params,
                "inner_cv": study.trials_dataframe()
                })
            res_df = pd.DataFrame(results)
            res_df.to_csv(os.path.join(data_path, 'interim', 
                                       'pipeline_model_selection_transcriptomics_individual.csv'))


