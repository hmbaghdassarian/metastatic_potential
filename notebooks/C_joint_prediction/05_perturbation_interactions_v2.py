#!/usr/bin/env python
# coding: utf-8

# Since linear models don't capture interactions between features, here, we test for this. This can reveal synergies or antagonisms that have outsized effects on metastatic potential. 

# In[106]:


import os
import itertools
import copy

from tqdm import tqdm 

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats

import sys
sys.path.insert(1, '../')
from utils import read_pickled_object, get_stats, cohen_d


# In[2]:


data_path = '/nobackup/users/hmbaghda/metastatic_potential/'
random_state = 42 + 3

n_cores = 30
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)


# Load data:

# In[3]:


X = pd.read_csv(os.path.join(data_path, 'processed',  'expr_joint.csv'), index_col = 0)
expr_joint = X.copy()

mp_joint=pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential_joint.csv'), index_col = 0)['mean']
y = mp_joint.values.ravel()

expr_protein = pd.read_csv(os.path.join(data_path, 'processed',  'expr_protein.csv'), index_col = 0)
expr_rna = pd.read_csv(os.path.join(data_path, 'processed',  'expr.csv'), index_col = 0)

protein_cols = expr_protein.columns
rna_cols = expr_rna.columns

X_protein = X[protein_cols].values
X_rna = X[rna_cols].values

# model results from 04
model_coefs = pd.read_csv(os.path.join(data_path, 'interim', 'joint_features.csv'), 
                          index_col = 0)
model_coefs.set_index('feature_name', inplace = True)

best_pipeline = read_pickled_object(os.path.join(data_path, 'processed', 
                                                 'best_model_svr_linear_joint.pickle'))


# Fit the full data on the original model:

# In[66]:


# best_pipeline = read_pickled_object(os.path.join(data_path, 'processed', 
#                                                  'best_model_svr_linear_joint.pickle'))
# X = (X_protein, X_rna)
# best_pipeline.fit(X, y)


# In[16]:


# model_coefs = pd.read_csv(os.path.join(data_path, 'interim', 'joint_features.csv'), 
#                           index_col = 0)
# if not np.allclose(model_coefs['SVM coefficient'].values, 
#                    best_pipeline.named_steps['model'].coef_.flatten()):
#     raise ValueError('Inconsitency between Notebook 04 and 05')
# # model_coefs.sort_values(by='SVM coefficient', key=lambda x: x.abs(), ascending=False, inplace=True)
# model_coefs.set_index('feature_name', inplace = True)


# In[20]:


# protein_indices = best_pipeline.named_steps['feature_processing'].transformer_list[0][1].named_steps['feature_selection_protein'].top_indices_
# selected_protein_cols = [protein_cols[i] for i in protein_indices]

# rna_indices = best_pipeline.named_steps['feature_processing'].transformer_list[1][1].named_steps['feature_selection_rna'].top_indices_
# selected_rna_cols = [rna_cols[i] for i in rna_indices]

# selected_indices = {'Transcriptomics': rna_indices, 
#                    'Proteomics': protein_indices}


# ## Set up the pipeline manually this time:
# 
# 1) Center the data for each modality:

# In[45]:


X_map_untransformed = {'Transcriptomics': X_rna, 'Proteomics': X_protein}
X_map = {k: X_ - np.mean(X_, axis=0) for k, X_ in X_map_untransformed.items()} # center the data


# 2) Get the selected features from the fit on the full dataset and union of modalities. 
# This is different from the 10-fold CV, where feature selection is run each time. We will use 10-fold CV, but on this consistent set of features which we are interested in testing interactions for. 

# In[8]:


X_map_selected = {}
for modality, X in X_map.items():
    modality_indices = model_coefs[model_coefs.Modality == modality]['feature_index'].values # selected_indices[modality]
    X_map_selected[modality] = X[:, modality_indices]

X_selected = np.concatenate([X_map_selected['Proteomics'], X_map_selected['Transcriptomics']], 
               axis = 1)


# 3. Sanity check -- does fitting the model on this manual pipeline match the automated best_pipeline?

# In[11]:


svr_model = copy.deepcopy(best_pipeline.named_steps['model'])


# In[12]:


svr_model.fit(X_selected, y)
model_coefs_check = pd.DataFrame(data = {'SVM coefficient': svr_model.coef_.ravel()})
model_coefs_check['feature_name'] = model_coefs.index.tolist()#selected_protein_cols + selected_rna_cols
model_coefs_check.set_index('feature_name', inplace = True)

if not np.allclose(model_coefs_check['SVM coefficient'].values, #model_coefs_check.loc[model_coefs.index,:]['SVM coefficient'].values, 
                   model_coefs['SVM coefficient'].values):
    raise ValueError('Something in the manual processing pipeline does not match the automated best_pipeline')
    
    
model_coefs = pd.read_csv(os.path.join(data_path, 'interim', 'joint_features.csv'), 
                          index_col = 0)
model_coefs.set_index('feature_name', inplace = True)
model_coefs['feature_index_selected'] = range(model_coefs.shape[0])
model_coefs.sort_values(by='SVM coefficient', key=lambda x: x.abs(), ascending=False, inplace=True)


# ## 10-fold CV
# Do the 10-fold CV on the X-block and the X-block with an added interaction term, and see if prediction improves:

# Get results for model with no interaction:

# In[97]:


np.random.seed(random_state)
outer_cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

results_nointeraction = {'test_corr': [], 'test_mse': []}
for k, (train_idx, test_idx) in tqdm(enumerate(outer_cv.split(X_selected, y))):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    res = get_stats(svr_model, y_train, y_test, X_train, X_test)
    train_corr, test_corr, train_mse, test_mse = res
    
    results_nointeraction['test_corr'].append(test_corr)
    results_nointeraction['test_mse'].append(test_mse)
    
    


# For each interaction, runt the same 10-fold CV:

# In[124]:


def get_interaction_value(feature_1, feature_2, 
                          X_selected, model_coefs, outer_cv, results_nointeraction):
    X_interaction = X_selected.copy()

    feature_1_index = model_coefs.loc[feature_1, 'feature_index_selected']
    feature_2_index = model_coefs.loc[feature_2, 'feature_index_selected']
    interaction = X_selected[:, feature_1_index]*X_selected[:, feature_2_index]

    X_interaction = np.concatenate([X_interaction, interaction.reshape(-1, 1)], axis = 1)

    results = {'test_corr': [], 'test_mse': []}
    for (train_idx, test_idx) in outer_cv.split(X_interaction, y):
        X_train, X_test = X_interaction[train_idx], X_interaction[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        res = get_stats(svr_model, y_train, y_test, X_train, X_test)
        train_corr, test_corr, train_mse, test_mse = res

        results['test_corr'].append(test_corr)
        results['test_mse'].append(test_mse)

    # compare to model with no interaction
    pval_corr = stats.mannwhitneyu(results['test_corr'], results_nointeraction['test_corr'], 
                      alternative = 'greater').pvalue
    cd_corr = cohen_d(results['test_corr'], results_nointeraction['test_corr'])

    pval_mse = stats.mannwhitneyu(results['test_mse'], results_nointeraction['test_mse'], 
                      alternative = 'less').pvalue
    cd_mse = cohen_d(results['test_mse'], results_nointeraction['test_mse'])
    
    return {'feature_1': feature_1,
            'feature_2': feature_2,
            'pval_corr': pval_corr, 'cd_corr': cd_corr, 'pval_mse': pval_mse, 'cd_mse': cd_mse}


# In[135]:


top_n = 50
features = model_coefs.index.tolist()[:top_n]
feature_combs = list(itertools.combinations(features, 2))

res_all = []
for feature_comb in tqdm(feature_combs):
    res = get_interaction_value(feature_1 = feature_comb[0], 
                                feature_2 = feature_comb[1],
                                X_selected = X_selected,
                                model_coefs = model_coefs, 
                                outer_cv = outer_cv, 
                                results_nointeraction = results_nointeraction)
    res_all.append(res)
    
res_all = pd.DataFrame(res_all)

_, bh_fdr, _, _ = multipletests(res_all.pval_corr, method='fdr_bh')
res_all['bh_fdr_corr'] = bh_fdr

_, bh_fdr, _, _ = multipletests(res_all.pval_mse, method='fdr_bh')
res_all['bh_fdr_mse'] = bh_fdr
res_all.to_csv(os.path.join(data_path, 'processed', 'joint_interaction_CV.csv'))

