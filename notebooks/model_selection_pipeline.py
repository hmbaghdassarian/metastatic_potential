#!/usr/bin/env python
# coding: utf-8

# To do: load in non feature selected, non mean-centered data

# In[212]:


import os

import numpy as np
import pandas as pd

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


# In[213]:


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
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        return X - self.mean_
    
def pearson_corr_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


# In[214]:


def create_pipeline(n_cores, random_state):
    # Step 1: Feature reduction/selection
    feature_reduction = [
        ('PLS', PLSRegression(scale = False)),
        ('PCA', PCA(random_state=random_state)),
        ('FeatureSelector', FeatureSelector(method='top_n_cv'))
    ]
    feature_reduction_params = [
        {'PLS__n_components': [2, 5, 10, 25, 50, 100, 200]},
        {'PCA__n_components': [2, 5, 10, 25, 50, 100, 200]},
        {'FeatureSelector__n_features': [250, 500, 1000, 5000, 17879]} # last one is no selecting features
    ]
    
    # Step 2: Modeling
    models = [
        ('SVR', SVR(gamma=0.001)),
        ('RFR', RandomForestRegressor(random_state=random_state, n_jobs=n_cores))  # Pass random_state and n_jobs
    ]
    model_params = [
        {
            'SVR__kernel': ['rbf', 'poly'],
            'SVR__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'SVR__degree': [2, 3, 4],
            'SVR__coef0': [0, 0.1, 0.5, 1.0, 1.2, 2.0]
        },
        {
            'RFR__n_estimators': range(100, 1001, 250),
            'RFR__max_features': ['sqrt', 'log2', 0.5, 0.75, 1],
            'RFR__max_samples': [0.25, 0.5, 0.75, None],
            'RFR__max_depth': [None, 10, 25, 50, 100, 200]
        }
    ]

    return feature_reduction, feature_reduction_params, models, model_params


# In[215]:


data_path = '/nobackup/users/hmbaghda/metastatic_potential/'
random_state = 42

n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# In[216]:


X = pd.read_csv(os.path.join(data_path, 'processed',  'expr.csv'), index_col = 0).T.values
y = pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential.csv'), index_col = 0).values


# In[217]:


outer_folds=10
inner_folds=5


# In[ ]:


mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
feature_reduction, feature_reduction_params, models, model_params = create_pipeline(n_cores, random_state)


results = []
for feature_step, feature_params in zip(feature_reduction, feature_reduction_params):
    for model_step, model_param in zip(models, model_params):
        
        # set up pipeline
        steps = []
        if feature_step[0] in ['PLS', 'PCA']:
            steps.append(('mean_centering', MeanCenterer()))
            steps.append(feature_step)
        elif feature_step[0] == 'FeatureSelector':
            steps.append(feature_step)
            steps.append(('mean_centering', MeanCenterer()))

        steps.append(model_step)

        pipeline = Pipeline(steps)

        param_grid = {**feature_params}
        param_grid.update({k: v for k, v in model_param.items()})



        grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, 
                            return_train_score = True,
                            scoring=mse_scorer, n_jobs=n_cores)
        for k, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print('feature: ' + feature_step[0] + ' | ' + 'model: ' + model_step[0] + ' | k: {}'.format(k))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_corr = pearsonr(y_train, y_train_pred)[0]
            test_corr = pearsonr(y_test, y_test_pred)[0]

            results.append({
                'outer_fold': k,
                'feature_selection': feature_step[0],
                'model': model_step[0],
                'train_corr': train_corr,
                'test_corr': test_corr,
                'best_params': grid.best_params_,
                'cv_results': grid.cv_results_
            })
            res_df = pd.DataFrame(results)
            res_df.to_csv(os.path.join(data_path, 'interim', 'pipeline.csv'))


# In[211]:





# In[208]:


k=0


# In[ ]:




