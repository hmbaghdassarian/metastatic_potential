#!/usr/bin/env python
# coding: utf-8

# We run the data on many different models to get a set of candidate models that have high performance. We do some grid search to ensure that poor performance isn't due to hyperparameter constraints, however the complete hyperparameter tuning is implemented in Notebook 02B. 

# In[1]:


# import torch
import argparse
import joblib
from pathlib import Path
import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.model_selection import GridSearchCV
import xgboost as XGB

import plotnine as p9


# Specify parameters and inputs:

# In[2]:


data_path = '/nobackup/users/hmbaghda/metastatic_potential/'
res_dir = os.path.join(data_path, 'interim')

X_path = os.path.join(data_path, 'processed',  'expr_joint.csv')
Y_path = os.path.join(data_path, 'processed', 'metastatic_potential_joint.csv')

seed = 42

num_folds = 10
grid_search = True
cv_folds = 5
n_cores = 80

model_types = ['PLSR','elasticNet', 'svm', 
               'rf', 'xgboost', 'knn']


# In[3]:


os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# In[13]:


X = pd.read_csv(X_path,index_col=0)
Y = pd.DataFrame(pd.read_csv(Y_path,index_col=0)['mean'])


# In[14]:


# separately mean center protein/rna

X_protein = X[[col for col in X.columns if col.startswith('sp')]]
X_rna = X[[col for col in X.columns if not col.startswith('sp')]]
X_protein = X_protein.subtract(X_protein.mean(axis=0), axis=1)
X_rna = X_rna.subtract(X_rna.mean(axis=0), axis=1)

X = pd.concat([X_protein, X_rna], axis = 1, ignore_index = False)


# In[5]:


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym,dim=0)
    x_square_sum = torch.sum(xm * xm,dim=0)
    y_square_sum = torch.sum(ym * ym,dim=0)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return r #torch.mean(r)

def pair_pearsonr(x, y, axis=0):
    mx = np.mean(x, axis=axis, keepdims=True)
    my = np.mean(y, axis=axis, keepdims=True)
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym, axis=axis)
    r_den = np.sqrt((xm*xm).sum(axis=axis) * (ym*ym).sum(axis=axis))
    r = r_num / r_den
    return r

def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList

def L2Regularization(deepLearningModel, L2):
    weightLoss = 0.
    biasLoss = 0.
    for layer in deepLearningModel:
        if isinstance(layer, torch.nn.Linear):
            weightLoss = weightLoss + L2 * torch.sum((layer.weight)**2)
            biasLoss = biasLoss + L2 * torch.sum((layer.bias)**2)
    L2Loss = biasLoss + weightLoss
    return(L2Loss)


# In[6]:


n_estimators = range(100, 1001, 250)
svm_c = [10**i for i in range(-3, 3)]
svm_gamma = [10**i for i in range(-3, 2)]
# svm_epsilon = [10**i for i in range(-2, 1)]
alpha = [10**i for i in range(-3, 3)]

grid_search_params = {
    'knn': {
        'n_neighbors': range(5, 41, 5)
    },
    'plsr': {
        'n_components': range(2, 16, 2)
    }, 
    'rf': {
        'n_estimators': n_estimators
            },
    'xgboost': {
        'n_estimators': n_estimators
    },
    'svm': {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': svm_c, 
        'gamma': svm_gamma,  # only poly and rbf
        'degree': [2,3,4,5],  # only for poly
#         'coef0':[0,0.1,0.5,1.,1.2,2.] # only for poly, 
        
    },
#     'svmRBF': {
#         'C': svm_c,
#         'gamma': svm_gamma,
#         'epsilon': svm_epsilon
#     }, 
#     'svmPoly':{
#         'gamma': svm_gamma,
#         'C': svm_c,
#         'degree':[2,3,4,5],
#         'coef0':[0,0.1,0.5,1.,1.2,2.]
#     }, 
#     'lasso':{
#         'alpha': alpha
#     }, 
#     'ridge': {
#         'alpha': alpha,
#     }, 
    'elasticNet': {
        'alpha': alpha,
        'l1_ratio': np.arange(0, 1.01, 0.25) # with 0 and 1 inclusive, this also does ridge and lasso
    }
}


# In[7]:


models = {}
for mdl in model_types:
    if mdl == 'knn':
        if grid_search:
            model = GridSearchCV(estimator=KNN(),
                                 param_grid = grid_search_params['knn'], 
                                 cv=cv_folds, 
                                 n_jobs=cv_folds)
        else:
            model = KNN(n_neighbors=5) # default value
    elif mdl=='PLSR':
        if grid_search:
            model = GridSearchCV(estimator=PLSRegression(scale=False),
                                 param_grid = grid_search_params['plsr'], cv=cv_folds, n_jobs=cv_folds)
        else:
            model = PLSRegression(n_components=4,scale=False)
    elif mdl == 'rf':
        if grid_search:
            model = GridSearchCV(estimator=RandomForestRegressor(n_jobs = int(n_cores/cv_folds)),
                                 param_grid = grid_search_params['rf'], cv=cv_folds, n_jobs=cv_folds)
        else:
            model = RandomForestRegressor(n_estimators=800, n_jobs = int(n_cores/cv_folds))
    elif mdl == 'xgboost':
        if grid_search:
            model = GridSearchCV(estimator=XGB.XGBRegressor(n_jobs = int(n_cores/cv_folds)),
                                 param_grid = grid_search_params['xgboost'], cv=cv_folds, n_jobs=cv_folds)
        else:
            model = XGB.XGBRegressor(n_estimators=800, n_jobs = int(n_cores/cv_folds))
    elif mdl == 'svm':
        if grid_search:
            model = GridSearchCV(estimator=SVR(),
                                 param_grid = grid_search_params['svm'], cv=cv_folds, n_jobs=cv_folds)
        else:
            model = SVR(kernel='rbf')
#     elif mdl == 'svmLinear':
#         if grid_search:
#             model = GridSearchCV(estimator=LinearSVR(),
#                                  param_grid = grid_search_params['svmLinear'], cv=cv_folds, n_jobs=cv_folds)
#         else:
#             model = LinearSVR()
#     elif mdl == 'svmRBF':
#         if grid_search:
#             model = GridSearchCV(estimator=SVR(kernel='rbf'),
#                                  param_grid = grid_search_params['svmRBF'], 
#                                  cv=cv_folds, n_jobs=cv_folds)
#         else:
#             model = SVR(kernel='rbf')
#     elif mdl == 'svmPoly':
#         if grid_search:
#             model = GridSearchCV(estimator=SVR(kernel='poly'),
#                                  param_grid = grid_search_params['svmPoly'], 
#                                  cv=cv_folds, n_jobs=cv_folds)
#         else:
#             model = SVR(kernel='poly')
#     elif mdl == 'lasso':
#         if grid_search:
#             model = GridSearchCV(estimator=Lasso(),
#                                  param_grid = grid_search_params['lasso'], cv=cv_folds, n_jobs=cv_folds)
#         else:
#             model = Lasso(alpha=0.1)
#     elif mdl == 'ridge':
#         if grid_search:
#             model = GridSearchCV(estimator=Ridge(),param_grid = grid_search_params['ridge'], 
#                                  cv=cv_folds, n_jobs=cv_folds)
#         else:
#             model = Ridge(alpha=0.1)
    elif mdl == 'elasticNet':
        if grid_search:
            model = GridSearchCV(estimator=ElasticNet(),
                                 param_grid = grid_search_params['elasticNet'], cv=cv_folds, n_jobs=cv_folds)
        else:
            model = ElasticNet(alpha=0.1)
#     elif mdl == 'neuralNet':
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         model = 'define the ANN just before it is trained'
#         epochs = 100
#         l2_reg  = 0.01
#         bs = 20
#         criterion = torch.nn.MSELoss(reduction='mean')
#     models.append(model)
    models[mdl] = model


# Run the iterations:

# In[ ]:


cv = KFold(n_splits=num_folds,shuffle=True,random_state=seed)

if os.path.isfile(os.path.join(data_path, 'processed', 'coarse_model_tests_joint.csv')):
    res = pd.read_csv(os.path.join(data_path, 'processed', 'coarse_model_tests_joint.csv'), 
                     index_col = 0)
else:
    res = pd.DataFrame(columns = ['model_type', 'fold', 'train_pearson', 'test_pearson', 'best_params'])

for model_type, model in tqdm(models.items()):
    print('Begun fitting and evaluation for model: %s'%model_type)
    for k, (train_index, test_index) in enumerate(cv.split(X)):
        if res[(res.model_type == model_type) & (res.fold == k)].shape[0] == 0:
            x_train = X.iloc[train_index,:].values
            x_test = X.iloc[test_index,:].values
            y_train = Y.iloc[train_index,:].values.ravel()
            y_test = Y.iloc[test_index,:].values.ravel()


            # fit model and evaluate in validation set
            model.fit(x_train,y_train)
            yhat_train = model.predict(x_train)
            yhat_test = model.predict(x_test)

            train_pearson=pair_pearsonr(y_train, yhat_train, axis=0).mean()
            test_pearson=pair_pearsonr(y_test, yhat_test, axis=0).mean()

            res.loc[res.shape[0], :] = [model_type, k, train_pearson, test_pearson, model.best_params_]
            res.to_csv(os.path.join(data_path, 'processed', 'coarse_model_tests_joint.csv'))
        else:
            pass


# In[3]:


res = pd.read_csv(os.path.join(data_path, 'processed', 'coarse_model_tests.csv'), index_col = 0)


# In[4]:


res.groupby('model_type').test_pearson.mean().sort_values(ascending = False)


# Since our 3rd best model is very similar in performance to our 4th best, we will now more comprehensively sweep all of these. We also see that 
# - elasticNet consitenly chose a ridge penalty, so we use ridge rather than elasticNet.  
# - svm was always the linear kernel, so we proceed with that.
