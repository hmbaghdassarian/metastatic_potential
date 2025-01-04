#!/usr/bin/env python
# coding: utf-8

# Since linear models performed best in 02A and was selected in 02B for proteomics and transcriptomics separately, we will just run the comprehensive hyperparameter tuning on the top 3 linear models (no RF this time). The other change is that the PLS n_components search starts at 6 instead of 2 because nothing lower than 6 was selected in 02A.

# In[24]:


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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge
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


# In[25]:


data_path = '/nobackup/users/hmbaghda/metastatic_potential/'
random_state = 42 + 2

n_cores = 80
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# In[27]:


def pearson_corr_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]
    
class ModalitySelector(BaseEstimator, TransformerMixin):
    def __init__(self, modality):
        if modality not in ['protein', 'rna']:
            raise ValueError("modality must be 'protein' or 'rna'")
        self.modality = modality

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a tuple: (X_protein, X_rna)
        if self.modality == 'protein':
            return X[0]  # Return X_protein
        elif self.modality == 'rna':
            return X[1]  # Return X_rna
        
class PLSRegression_X(PLSRegression):
    def transform(self, X, y=None):
        X_transformed = super().transform(X, y)
        if isinstance(X_transformed, tuple):
            X_transformed = X_transformed[0]
        return X_transformed
        

# class TupleCV:
#     def __init__(self, base_cv):
#         self.base_cv = base_cv

#     def split(self, X, y=None):
#         if not isinstance(X, tuple):
#             raise ValueError("Input to TupleCV must be a tuple (X_protein, X_rna).")
#         X_protein, X_rna = X
#         for train_idx, test_idx in self.base_cv.split(X_protein, y):  # Use X_protein for indexing
#             # Yield consistent splits for both modalities
#             yield (
#                 (X_protein[train_idx], X_rna[train_idx]),  # Training data tuple
#                 (X_protein[test_idx], X_rna[test_idx]),    # Testing data tuple
#             ), (y[train_idx], y[test_idx])  # Training and testing targets

#     def get_n_splits(self, X=None, y=None):
#         return self.base_cv.get_n_splits()


# In[28]:


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


def optuna_objective(trial, X_protein, X_rna, y, inner_folds, #inner_cv, 
                     n_cores, random_state):
    # Suggest parameters for feature selection
    n_features_protein = trial.suggest_categorical("FeatureSelector__n_features_protein", [250, 500, 1000, 5000, X_protein.shape[1]])
    n_features_rna = trial.suggest_categorical("FeatureSelector__n_features_rna", [250, 500, 1000, 5000, X_rna.shape[1]])

    # Protein-specific pipeline
    protein_pipeline = Pipeline([
        ("select_protein", ModalitySelector(modality="protein")),
        ("feature_selection_protein", FeatureSelector(method="top_n_cv", n_features=n_features_protein)),
        ("mean_centering_protein", MeanCenterer()),  # Mean centering for protein data
    ])

    # RNA-specific pipeline
    rna_pipeline = Pipeline([
        ("select_rna", ModalitySelector(modality="rna")),
        ("feature_selection_rna", FeatureSelector(method="top_n_cv", n_features=n_features_rna)),
        ("mean_centering_rna", MeanCenterer()),  # Mean centering for RNA data
    ])

    # Combine both pipelines
    combined_pipeline = FeatureUnion([
        ("protein_pipeline", protein_pipeline),
        ("rna_pipeline", rna_pipeline),
    ])

    # Add the model
    steps = [
        ("feature_processing", combined_pipeline),
    ]
    
    model_type = trial.suggest_categorical("model_type", ["SVR", 'PLS', 'Ridge'])
    
    if model_type == "SVR":
        steps.append(("model", SVR(
            kernel='linear',
            C=trial.suggest_float("SVR__C", 1e-4, 1e2, log=True),
            epsilon=trial.suggest_float("SVR__epsilon", 1e-3, 10, log=True)
        )))
    elif model_type == 'PLS':
        steps.append(
            ("model", PLSRegression_X(n_components=trial.suggest_int("PLS__n_components", 6, 100, step = 3))), 
        )
    elif model_type == 'Ridge':
        steps.append(
            ('model', Ridge(alpha=trial.suggest_float("Ridge__alpha", 1, 250, step = 10), 
                                             random_state=random_state))
        )

    pipeline = Pipeline(steps)

    # Evaluate with cross-validation
    X_combined = (X_protein, X_rna)  # Combine datasets as tuple
    
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    mse_scores = []

    for train_idx, test_idx in kf.split(X_protein):
        X_train = (X_protein[train_idx], X_rna[train_idx])
        X_test = (X_protein[test_idx], X_rna[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit and evaluate the pipeline
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
        
    return np.mean(mse_scores)
#     mse = -cross_val_score(pipeline, X_combined, y, 
#                            cv=inner_cv, 
#                            scoring="neg_mean_squared_error", 
#                            n_jobs=n_cores).mean()

#     return mse

def generate_best_pipeline(study):
    best_params = study.best_params

    # Protein-specific pipeline
    protein_pipeline = Pipeline([
        ("select_protein", ModalitySelector(modality="protein")),
        ("feature_selection_protein", FeatureSelector(method="top_n_cv", n_features=best_params["FeatureSelector__n_features_protein"])),
        ("mean_centering_protein", MeanCenterer()),  # Mean centering for protein data
    ])

    # RNA-specific pipeline
    rna_pipeline = Pipeline([
        ("select_rna", ModalitySelector(modality="rna")),
        ("feature_selection_rna", FeatureSelector(method="top_n_cv", n_features=best_params["FeatureSelector__n_features_rna"])),
        ("mean_centering_rna", MeanCenterer()),  # Mean centering for RNA data
    ])

    # Combine both pipelines
    combined_pipeline = FeatureUnion([
        ("protein_pipeline", protein_pipeline),
        ("rna_pipeline", rna_pipeline),
    ])

    # Add the model
    steps = [
        ("feature_processing", combined_pipeline),
    ]

    steps.append(("model", SVR(
        kernel='linear',
        C=best_params["SVR__C"],
        epsilon=best_params['SVR__epsilon']
    )))
    
    # Create the full pipeline
    best_pipeline = Pipeline(steps)
    return best_pipeline


# In[44]:


X = pd.read_csv(os.path.join(data_path, 'processed',  'expr_joint.csv'), index_col = 0)
y = pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential_joint.csv'), index_col = 0)['mean'].values.ravel()

expr_protein = pd.read_csv(os.path.join(data_path, 'processed',  'expr_protein.csv'), index_col = 0)
expr_rna = pd.read_csv(os.path.join(data_path, 'processed',  'expr.csv'), index_col = 0)

protein_cols = expr_protein.columns
rna_cols = expr_rna.columns

X_protein = X[protein_cols].values
X_rna = X[rna_cols].values


# In[31]:


outer_folds=10
inner_folds=5
n_trials = 200


# In[12]:


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


# In[13]:


outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
# inner_cv = TupleCV(KFold(n_splits=inner_folds, shuffle=True, random_state=random_state))

if os.path.isfile(os.path.join(data_path, 'interim', 'pipeline_model_selection_joint.csv')):
    res_df = pd.read_csv(os.path.join(data_path, 'interim', 'pipeline_model_selection_joint.csv'), 
                     index_col = 0)
    results = res_df.to_dict(orient='records')
else:
    results = []
    res_df = None
    
for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    if res_df is not None and res_df[res_df.fold == k].shape[0] != 0:
        pass
    else:
        print(str(k))
        X_train_rna, X_test_rna = X_rna[train_idx], X_rna[test_idx]
        X_train_protein, X_test_protein = X_protein[train_idx], X_protein[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        pruner = optuna.pruners.SuccessiveHalvingPruner()
        study = optuna.create_study(direction="minimize", 
                                    sampler=HybridSampler(primary_sampler=cmaes_sampler, fallback_sampler=tpe_sampler), 
                                   pruner = pruner, 
                                   study_name = '{}_optuna'.format(k))
        study.optimize(
            lambda trial: optuna_objective(trial, X_train_protein, X_train_rna, y_train, inner_folds, #inner_cv, 
                                           n_cores, random_state),
            n_trials=n_trials, 
            catch=(ValueError,)
        )
        write_pickled_object(study, os.path.join(data_path, 'interim', study.study_name + '.pickle'))

        X_train = (X_train_protein, X_train_rna)
        X_test = (X_test_protein, X_test_rna)
        
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
        res_df.to_csv(os.path.join(data_path, 'interim', 'pipeline_model_selection_joint.csv'))

