#!/usr/bin/env python
# coding: utf-8

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

import sys
sys.path.insert(1, '../')
from utils import write_pickled_object
from utils import FeatureSelector, MeanCenterer


# In[2]:


data_path = '/nobackup/users/hmbaghda/metastatic_potential/'
random_state = 888

n_cores = 80
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# In[3]:


def pearson_corr_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

class PLSRegression_X(PLSRegression):
    def transform(self, X, y=None):
        X_transformed = super().transform(X, y)
        if isinstance(X_transformed, tuple):
            X_transformed = X_transformed[0]
        return X_transformed
    
class HybridSampler(optuna.samplers.BaseSampler):
    def __init__(self, primary_sampler, fallback_sampler):
        self.primary_sampler = primary_sampler  # e.g., CmaEsSampler
        self.fallback_sampler = fallback_sampler  # e.g., TPESampler

    def infer_relative_search_space(self, study, trial):
        # Let the primary sampler define the relative search space
        return self.primary_sampler.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        # Let the primary sampler handle relative sampling
        return self.primary_sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Use the fallback sampler for unsupported parameter types
        if isinstance(param_distribution, CategoricalDistribution):
            return self.fallback_sampler.sample_independent(study, trial, param_name, param_distribution)
        # Default to the primary sampler
        return self.primary_sampler.sample_independent(study, trial, param_name, param_distribution)

class RandomTPESampler(TPESampler):
    def __init__(self, exploration_sampler, exploration_freq=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_sampler = exploration_sampler
        self.exploration_freq = exploration_freq

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Use the exploration_sampler periodically
        if trial.number % self.exploration_freq == 0:
            return self.exploration_sampler.sample_independent(study, trial, param_name, param_distribution)
        # Default to TPE
        return super().sample_independent(study, trial, param_name, param_distribution)


# In[4]:


def optuna_objective(trial, X, y, inner_cv, n_cores, random_state, model_type):
    # Define feature reduction/selection method
        
    steps = [
        ("feature_reduction", FeatureSelector(method="top_n_cv", 
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
    mse = -cross_val_score(pipeline, X, y, 
                           cv=inner_cv, 
                           scoring="neg_mean_squared_error", 
                           n_jobs=inner_cv.n_splits).mean()

    return mse


def generate_best_pipeline(study):
    best_params = study.best_params
    steps = []
    steps.append(("feature_reduction", FeatureSelector(method="top_n_cv", n_features=best_params["FeatureSelector__n_features"])))
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
            max_depth=best_params[model_type + '__n_estimators'],
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
            n_jobs = int(n_cores/inner_cv.n_splits))))

    best_pipeline = Pipeline(steps)
    return best_pipeline


# In[11]:


X = pd.read_csv(os.path.join(data_path, 'processed',  'expr_protein.csv'), index_col = 0).values
y = pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential_protein.csv'), index_col = 0)['mean'].values.ravel()


# In[5]:


outer_folds=10
inner_folds=5
n_trials = 100


# In[6]:


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

results = []
res_df = None

for model_type in ['SVR_linear', 'PLS', 'Ridge', 'Lasso', 'ElasticNet', 
                   'SVR_poly', 'SVR_rbf', 'RFR', 'KNN']:#'XGBoost', ]:
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
            res_df.to_csv(os.path.join(data_path, 'interim', 'pipeline_model_selection_proteomics_individual.csv'))

