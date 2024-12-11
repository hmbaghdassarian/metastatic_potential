#!/usr/bin/env python
# coding: utf-8

# To do: load in non feature selected, non mean-centered data

# In[109]:


import os
import pickle
import pathlib

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import CmaEsSampler, TPESampler, RandomSampler
from optuna.distributions import CategoricalDistribution

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import pearsonr
from sklearn.utils import shuffle


# In[110]:


data_path = '/nobackup/users/hmbaghda/metastatic_potential/'
random_state = 42

n_cores = 80
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# In[111]:


def write_pickled_object(object_, file_name: str) -> None:
    if '.' in file_name:
        p = pathlib.Path(file_name)
        extensions = "".join(p.suffixes)
        file_name = str(p).replace(extensions, '.pickle')
    else:
        file_name = file_name + '.pickle'

    with open(file_name, 'wb') as handle:
        pickle.dump(object_, handle)


# In[112]:


# Feature selection transformer
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method='top_n_cv', n_features=None):
        if method not in ['top_n_cv']:#, 'all_features']:
            raise ValueError('Incorrect feature selection method implemented')
        self.method = method
        self.n_features = n_features

    def fit(self, X, y=None):
        if self.method == 'top_n_cv':
            self.coefficient_of_variation_ = np.std(X, axis=0) / np.mean(X, axis=0)
            self.top_indices_ = np.argsort(self.coefficient_of_variation_)[::-1][:self.n_features]
#         elif self.method == 'all_features':
#             self.top_indices_ = range(X.shape[1])
        return self
    def transform(self, X, y=None):
        return X[:, self.top_indices_]
    
class MeanCenterer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
#         self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        return X - np.mean(X, axis=0)
    
def pearson_corr_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

class PLSRegression_X(PLSRegression):
    def transform(self, X, y=None):
        X_transformed = super().transform(X, y)
        if isinstance(X_transformed, tuple):
            X_transformed = X_transformed[0]
        return X_transformed


# In[118]:


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


def optuna_objective(trial, X, y, inner_cv, n_cores, random_state):
    # Define feature reduction/selection method
    feature_step = trial.suggest_categorical("feature_step", ["PLS", "PCA", "FeatureSelector"])

    if feature_step == "PLS":
        steps = [
            ("mean_centering", MeanCenterer()),
            ("feature_reduction", PLSRegression_X(n_components=trial.suggest_categorical("PLS__n_components", [2, 5, 10, 25, 50, 100]))),
        ]
    elif feature_step == "PCA":
        steps = [
            ("mean_centering", MeanCenterer()),
            ("feature_reduction", PCA(n_components=trial.suggest_categorical("PCA__n_components", [2, 5, 10, 25, 50, 100]), random_state=random_state)),
        ]
    elif feature_step == "FeatureSelector":
        steps = [
            ("feature_reduction", FeatureSelector(method="top_n_cv", n_features=trial.suggest_categorical("FeatureSelector__n_features", [250, 500, 1000, 5000, 17879]))),
            ("mean_centering", MeanCenterer()),
        ]
    # Define model
    model_type = trial.suggest_categorical("model_type", ["SVR", "RFR"])
    if model_type == "SVR":
        steps.append(("model", SVR(
            kernel=trial.suggest_categorical("SVR__kernel", ["rbf", "poly"]),
            C=trial.suggest_float("SVR__C", 1e-4, 1e2, log = True),
            degree=trial.suggest_int("SVR__degree", 2, 4),
#             coef0=trial.suggest_uniform("SVR__coef0", 0, 2),
            gamma=0.001
        )))
    elif model_type == "RFR":
        steps.append(("model", RandomForestRegressor(
            n_estimators=trial.suggest_int("RFR__n_estimators", 300, 1600, step=400),
            max_features=trial.suggest_categorical("RFR__max_features", ["sqrt", "log2", 0.5, 0.75, 1]),
            max_samples=trial.suggest_categorical("RFR__max_samples", [0.25, 0.5, 0.75, None]),
            max_depth=trial.suggest_categorical("RFR__max_depth", [None, 10, 25, 50, 100, 200]),
            random_state=random_state,
            n_jobs=int(n_cores/inner_cv.n_splits)
        )))

    # Create the pipeline
    pipeline = Pipeline(steps)

    # Evaluate with cross-validation
    mse = -cross_val_score(pipeline, X, y, 
                           cv=inner_cv, 
                           scoring="neg_mean_squared_error", 
                           n_jobs=inner_cv.n_splits).mean()

#     for fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X, y)):
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         # Train and evaluate the pipeline on the current fold
#         pipeline.fit(X_train, y_train)
#         y_val_pred = pipeline.predict(X_val)
#         mse = mean_squared_error(y_val, y_val_pred)

#         # Store the MSE for this fold
#         mse_scores.append(mse)

#         # Report intermediate result to Optuna
#         trial.report(np.mean(mse_scores), step=fold_idx)

#         # Check if the trial should be pruned
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
    
#     return np.mean(mse_scores)

    return mse


def generate_best_pipeline(study):
    best_params = study.best_params
    steps = []
    if best_params["feature_step"] == "PLS":
        steps.append(("mean_centering", MeanCenterer()))
        steps.append(("feature_reduction", PLSRegression_X(n_components=best_params["PLS__n_components"])))
    elif best_params["feature_step"] == "PCA":
        steps.append(("mean_centering", MeanCenterer()))
        steps.append(("feature_reduction", PCA(n_components=best_params["PCA__n_components"], random_state=random_state)))
    elif best_params["feature_step"] == "FeatureSelector":
        steps.append(("feature_reduction", FeatureSelector(method="top_n_cv", n_features=best_params["FeatureSelector__n_features"])))
        steps.append(("mean_centering", MeanCenterer()))

    if "SVR__kernel" in best_params:
        steps.append(("model", SVR(
            kernel=best_params["SVR__kernel"],
            C=best_params["SVR__C"],
            degree=best_params["SVR__degree"],
#             coef0=best_params["SVR__coef0"],
            gamma=0.001
        )))
    elif "RFR__n_estimators" in best_params:
        steps.append(("model", RandomForestRegressor(
            n_estimators=best_params["RFR__n_estimators"],
            max_features=best_params["RFR__max_features"],
            max_samples=best_params["RFR__max_samples"],
            max_depth=best_params["RFR__max_depth"],
            random_state=random_state,
            n_jobs=n_cores
        )))

    best_pipeline = Pipeline(steps)
    return best_pipeline


# In[57]:


X = pd.read_csv(os.path.join(data_path, 'processed',  'expr.csv'), index_col = 0).T.values
y = pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential.csv'), index_col = 0).values.ravel()


# In[119]:


outer_folds=10
inner_folds=5
n_trials = 150


# In[124]:


# cmaes_sampler = CmaEsSampler(seed=random_state, 
#                              warn_independent_sampling=False, 
#                             restart_strategy='bipop')

# exploration_sampler = RandomSampler(seed=random_state)
# tpe_sampler = RandomTPESampler(seed=random_state, 
#                                n_startup_trials = 25,
#                                exploration_sampler = exploration_sampler, 
#                                exploration_freq=20)


# def optuna_objective_toy(trial, X, y, inner_cv, n_cores, random_state):
#     # Define feature reduction/selection method
#     feature_step = trial.suggest_categorical("feature_step", ["PLS", "FeatureSelector"])

#     if feature_step == "PLS":
#         steps = [
#             ("mean_centering", MeanCenterer()),
#             ("feature_reduction", PLSRegression_X(n_components=trial.suggest_categorical("PLS__n_components", [2, 3]))),
#         ]
#     elif feature_step == "FeatureSelector":
#         steps = [
#             ("feature_reduction", FeatureSelector(method="top_n_cv", n_features=trial.suggest_categorical("FeatureSelector__n_features", [5,10]))),
#             ("mean_centering", MeanCenterer()),
#         ]
#     # Define model
#     model_type = trial.suggest_categorical("model_type", ["SVR", "RFR"])
#     if model_type == "SVR":
#         steps.append(("model", SVR(
#             kernel=trial.suggest_categorical("SVR__kernel", ["rbf", "poly"]),
#             gamma=0.001
#         )))
#     elif model_type == "RFR":
#         steps.append(("model", RandomForestRegressor(
#             n_estimators=trial.suggest_int("RFR__n_estimators", 5, 7),
#             random_state=random_state,
#             n_jobs=int(n_cores/inner_cv.n_splits)
#         )))

#     # Create the pipeline
#     pipeline = Pipeline(steps)

#     # Evaluate with cross-validation
#     mse = -cross_val_score(pipeline, X, y, 
#                            cv=inner_cv, 
#                            scoring="neg_mean_squared_error", 
#                            n_jobs=inner_cv.n_splits).mean()

#     return mse


# def generate_best_pipeline_toy(study):
#     best_params = study.best_params
#     steps = []
#     if best_params["feature_step"] == "PLS":
#         steps.append(("mean_centering", MeanCenterer()))
#         steps.append(("feature_reduction", PLSRegression_X(n_components=best_params["PLS__n_components"])))
#     elif best_params["feature_step"] == "FeatureSelector":
#         steps.append(("feature_reduction", FeatureSelector(method="top_n_cv", n_features=best_params["FeatureSelector__n_features"])))
#         steps.append(("mean_centering", MeanCenterer()))

#     if "SVR__kernel" in best_params:
#         steps.append(("model", SVR(
#             kernel=best_params["SVR__kernel"],
#             gamma=0.001
#         )))
#     elif "RFR__n_estimators" in best_params:
#         steps.append(("model", RandomForestRegressor(
#             n_estimators=best_params["RFR__n_estimators"],
#             random_state=random_state,
#             n_jobs=n_cores
#         )))

#     best_pipeline = Pipeline(steps)
#     return best_pipeline

# X = np.random.randn(20, 100)
# y = np.random.randn(20,)
# outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
# inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

# results = []
# for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
#     print(str(k))
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
    
    
#     pruner = optuna.pruners.SuccessiveHalvingPruner()
#     study = optuna.create_study(direction="minimize", 
#                                 sampler=PeriodicHybridSampler(primary_sampler=cmaes_sampler, 
#                                                               fallback_sampler=tpe_sampler, 
#                                                              exploration_sampler=exploration_sampler, ), 
#                                pruner = pruner, 
#                                study_name = '{}_optuna'.format(k))
#     break


# In[115]:


cmaes_sampler = CmaEsSampler(seed=random_state, 
                             warn_independent_sampling=False, 
                            restart_strategy='bipop')

exploration_sampler = RandomSampler(seed=random_state)
tpe_sampler = RandomTPESampler(seed=random_state, 
                               n_startup_trials = 25,
                               exploration_sampler = exploration_sampler, 
                               exploration_freq=20 # randomly sample every n trials
                              )
# tpe_sampler = TPESampler(seed=random_state, 
#                         n_startup_trials = 20)


# In[ ]:


outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

results = []
for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(str(k))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(direction="minimize", 
                                sampler=HybridSampler(primary_sampler=cmaes_sampler, fallback_sampler=tpe_sampler), 
                               pruner = pruner, 
                               study_name = '{}_optuna'.format(k))
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, inner_cv, n_cores, random_state),
        n_trials=n_trials, 
        catch=(ValueError,)
    )
    write_pickled_object(study, os.path.join(data_path, 'interim', study.study_name + '.pickle'))
        
    best_pipeline = generate_best_pipeline(study)
    best_pipeline.fit(X_train, y_train)

    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    train_corr = pearsonr(y_train, y_train_pred)[0]
    test_corr = pearsonr(y_test, y_test_pred)[0]

    results.append({
        "fold": k,
        "train_corr": train_corr,
        "test_corr": test_corr,
        "best_params": study.best_params,
        "inner_cv": study.trials_dataframe()
        })
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(data_path, 'interim', 'pipeline.csv'))


# # Start

# In[43]:


# steps = [
#     ("mean_centering", MeanCenterer()),
#     ("feature_reduction", PLSRegression_X(n_components=10)),
# ]
# steps.append(("model", RandomForestRegressor(
#     n_estimators=300,
#     max_features='sqrt',
#     max_samples=None,
#     max_depth=25,
#     random_state=random_state,
#     n_jobs=n_cores
# )))
# best_pipeline = Pipeline(steps)


# In[44]:


# best_pipeline.fit(X_train, y_train)

# y_train_pred = best_pipeline.predict(X_train)
# y_test_pred = best_pipeline.predict(X_test)

# train_corr = pearsonr(y_train, y_train_pred)[0]
# test_corr = pearsonr(y_test, y_test_pred)[0]

# y_test_pred = pd.cut(y_test_pred, bins=3, labels=['Low', 'Medium', 'High'])
# y_test = pd.cut(y_test, bins = 3, labels=['Low', 'Medium', 'High'])
# print(test_corr)
# print(f1_score(y_test_pred, y_test, average='weighted'))
# print((1/3) + (test_corr**2))


# In[93]:


# k = 0


# In[95]:


# with open(os.path.join(data_path, 'interim', study.study_name + '.pickle'), 'rb') as handle:
#     study = pickle.load(handle)


# In[128]:


# study.sampler.primary_sampler

