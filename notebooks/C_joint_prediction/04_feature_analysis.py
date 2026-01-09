#!/usr/bin/env python
# coding: utf-8

# In[156]:


import os
import json

from tqdm import tqdm
from tqdm import trange

import numpy as np
import pandas as pd

from scipy.stats import hypergeom
from scipy.stats import fisher_exact
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import r2_score


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm as cm_matplot
from matplotlib.colorbar import make_axes

import mygene

import sys
sys.path.insert(1, '../')
from utils import write_pickled_object, read_pickled_object, ModalitySelector


# In[157]:


data_path = '/home/hmbaghda/orcd/pool/metastatic_potential/'
random_state = 42 + 3

n_cores = 45
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1' #str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = '1' #str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = '1' #str(n_cores)
os.environ["NUMEXPR_NUM_THREADS"] = '1' #str(n_cores)


# In[213]:


X = pd.read_csv(os.path.join(data_path, 'processed',  'expr_joint.csv'), index_col = 0)
expr_joint = X.copy()

mp_joint=pd.read_csv(os.path.join(data_path, 'processed', 'metastatic_potential_joint.csv'), index_col = 0)['mean']
y = mp_joint.values.ravel()

expr_protein = pd.read_csv(os.path.join(data_path, 'processed',  'expr_protein.csv'), index_col = 0)
expr_rna = pd.read_csv(os.path.join(data_path, 'processed',  'expr.csv'), index_col = 0)

protein_cols = expr_protein.columns
rna_cols = expr_rna.columns

# with open("protein_cols.txt", "w") as f: f.writelines(f"{item}\n" for item in protein_cols)
# with open("rna_cols.txt", "w") as f: f.writelines(f"{item}\n" for item in rna_cols)


X_protein = X[protein_cols].values
X_rna = X[rna_cols].values

best_pipeline = read_pickled_object(os.path.join(data_path, 'processed', 
                                                 'best_model_svr_linear_joint.pickle'))


# In[159]:


# fns = [os.path.join(data_path, 'processed', 
#                                                  'best_model_svr_linear_joint.pickle'), 
#        os.path.join(data_path, 'processed',  'expr_joint.csv'), 
#        os.path.join(data_path, 'processed', 'metastatic_potential_joint.csv'), 
#        os.path.join(data_path, 'processed',  'expr_protein.csv'), 
#        os.path.join(data_path, 'processed',  'expr.csv'), 
#       os.path.join(data_path, 'processed', 'uniprot_mapper.json')]

# print('scp hmbaghda@satori-login-002.mit.edu:' + ' hmbaghda@satori-login-002.mit.edu:'.join(fns) + ' Downloads/mp_ronit/.')
# # for fn in fns:
# #     print('scp hmbaghda@satori-login-002.mit.edu:' + fn + ' Downloads/mp_ronit')
# #     print(' ')


# ID mapping between protein and RNA:

# In[160]:


# mapper generated in notebook B/04
with open(os.path.join(data_path, 'processed', 'uniprot_mapper.json'), "r") as json_file:
    uid_mapper = json.load(json_file)

# manually mapped some that failed to map using uniprot ID
manual_map = {'Q9TNN7': 'HLA-C',
'P16189': 'HLA-A',
'P30456': 'HLA-A',
'P30443': 'HLA-A',
'P05534': 'HLA-A',
'P18462': 'HLA-A',
'P01892': 'HLA-A',
'P13746': 'HLA-A',
'P01891': 'HLA-A',
'P30483': 'HLA-B',
'P30484': 'HLA-B',
'P03989': 'HLA-B',
'P30460': 'HLA-B',
'P30461': 'HLA-B',
'Q95365': 'HLA-B',
'P16188': 'HLA-A',
'Q95604': 'HLA-C',
'Q07000': 'HLA-C',
'P30499': 'HLA-C',
'P30501': 'HLA-C',
'P30504': 'HLA-C',
'Q95IE3': 'HLA-DRB1',
'P04229': 'HLA-DRB1',
'P20039': 'HLA-DRB1',
'P13760': 'HLA-DRB1',
'Q5Y7A7': 'HLA-DRB1',
'Q9GIY3': 'HLA-DRB1',
'Q9TQE0': 'HLA-DRB1',
'Q30134': 'HLA-DRB1'}

protein_names = []
for protein_id in protein_cols:
    uniprot_id = protein_id.split('|')[1].split('-')[0]
    if pd.isna(uid_mapper[uniprot_id]):
        gene_name = protein_id.split('|')[-1].split('_HUMAN')[0]
        if gene_name[0].isdigit():
            gene_name = manual_map[uniprot_id]
    else:
        gene_name = uid_mapper[uniprot_id]
    protein_names.append(gene_name)


# In[161]:


rna_names = [rna_id.split(' (')[0] for rna_id in rna_cols]
# protein_names = [protein_id.split('|')[-1].split('_HUMAN')[0] for protein_id in protein_cols]
intersect_names = set(rna_names).intersection(protein_names)

n_features = [len(rna_names), len(protein_names), len(intersect_names)]
print('Of the {} RNA features and {} protein features, there are {} features in common'.format(*n_features))


# Fit on the full dataset:

# In[162]:


X = (X_protein, X_rna)
best_pipeline = clone(best_pipeline)
best_pipeline.fit(X, y)
model_coefs = best_pipeline.named_steps['model'].coef_.flatten()

# write_pickled_object(best_pipeline, 
#                      os.path.join(data_path, 'interim', 'best_linearSVR_joint_fitted_allsamples.pickle'))


# In[163]:


fig, ax = plt.subplots(figsize = (5,5))

sns.kdeplot(model_coefs, ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha = 'center')
ax.set_xlabel('Fitted Linear SVM Feature Coefficients')
# ax.set_title('Transcriptomics + Proteomics')

plt.savefig(os.path.join(data_path, 'figures', 'joint_feature_dist.png'), 
            dpi=300, 
            bbox_inches="tight")  

("")


# Let's take a look at the top 500 features:

# In[164]:


protein_indices = best_pipeline.named_steps['feature_processing'].transformer_list[0][1].named_steps['feature_selection_protein'].top_indices_
selected_protein_cols = [protein_cols[i] for i in protein_indices]

rna_indices = best_pipeline.named_steps['feature_processing'].transformer_list[1][1].named_steps['feature_selection_rna'].top_indices_
selected_rna_cols = [rna_cols[i] for i in rna_indices]

selected_cols = selected_protein_cols + selected_rna_cols

# X_selected = np.concatenate((X_protein[:,protein_indices], X_rna[:,rna_indices]),axis=1)
# np.savetxt(os.path.join(data_path, 'interim', 'joint_selected_X.txt'), 
#            X_selected, delimiter=',', fmt='%d')

model_coefs = best_pipeline.named_steps['model'].coef_.flatten()
model_coefs = pd.DataFrame(data = {'SVM coefficient': model_coefs, 
                                  'Modality': ['Proteomics']*len(selected_protein_cols) + ['Transcriptomics']*len(selected_rna_cols)}, 
                           index = selected_cols)

model_coefs_ = model_coefs.copy()
model_coefs_['feature_index'] = list(protein_indices) + list(rna_indices)
model_coefs_.reset_index(names = 'feature_name', inplace = True)


top_n = 500
model_coefs.sort_values(by='SVM coefficient', key=lambda x: x.abs(), ascending=False, inplace=True)
model_coefs.reset_index(names = 'Feature Name', inplace = True)
model_coefs['Modality'] = pd.Categorical(model_coefs['Modality'], 
                                         categories = ['Transcriptomics', 'Proteomics'], 
                                        ordered = True)
top_model_coefs = model_coefs.iloc[:top_n, :]

model_coefs.to_csv(os.path.join(data_path, 'processed', 'rank_ordered_joint_features.csv'))



# In[165]:


fig, ax = plt.subplots(figsize = (15, 5))

viz_df = top_model_coefs.sort_values(by = 'SVM coefficient', ascending = True)
sns.barplot(data = viz_df, x = 'Feature Name', y = 'SVM coefficient', hue = 'Modality', 
            ax = ax)
ax.set_xticklabels([])
ax.set_title('Top 500 Features by Absolute Value of SVM Coefficient')
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax.set_xlabel('Features')

plt.savefig(os.path.join(data_path, 'figures', 'joint_feature_top_barplot.png'), 
            dpi=300, 
            bbox_inches="tight")

("")


# Let's look at the relative usage of proteomics vs transcriptomic features:

# In[169]:


def get_shared_genes(curr_coefs, get_prot_difference = False):
    rna_genes = set(curr_coefs[curr_coefs.Modality == 'Transcriptomics']['Gene Name'])
    protein_genes = set(curr_coefs[curr_coefs.Modality == 'Proteomics']['Gene Name'])
    common_genes = rna_genes.intersection(protein_genes)

    if get_prot_difference:
        diff_genes = protein_genes.difference(rna_genes)
        return rna_genes, protein_genes, common_genes, diff_genes

    return rna_genes, protein_genes, common_genes


# In[170]:


model_coefs = pd.read_csv(os.path.join(data_path, 'processed', 'rank_ordered_joint_features.csv'), index_col = 0)


# In[171]:


# proteomics fraction
model_coefs['Protein Feature Fraction'] = np.nan
for i in trange(model_coefs.shape[0]):
    n_protein = model_coefs.iloc[:(i+1), :]['Modality'].value_counts()['Proteomics']    
    model_coefs.iloc[i, 3] = n_protein/(i+1)

model_coefs.reset_index(names = 'Feature Rank', inplace = True)
model_coefs['Feature Rank'] += 1

top_prot_feature = model_coefs.loc[model_coefs['Protein Feature Fraction'].idxmax(),'Feature Rank']

# shared features
rna_mapper = dict(zip(rna_cols, rna_names))
protein_mapper = dict(zip(protein_cols, protein_names))
gene_mapper = {**rna_mapper, **protein_mapper}
model_coefs['Gene Name'] = model_coefs['Feature Name'].map(gene_mapper)
assert not model_coefs['Gene Name'].isna().any(), 'Some features did not map to gene name'

model_coefs_['gene_name'] = model_coefs_['feature_name'].map(gene_mapper)
model_coefs_.to_csv(os.path.join(data_path, 'interim', 'joint_features.csv'))

model_coefs['Shared Genes Fraction - Protein'] = np.nan
model_coefs['Shared Genes Fraction - RNA'] = np.nan
for i in trange(model_coefs.shape[0]):
    curr_coefs=model_coefs.iloc[:(i+1), :]
    rna_genes, protein_genes, common_genes = get_shared_genes(curr_coefs)

    model_coefs.iloc[i, 6] = 0 if len(protein_genes) == 0 else len(common_genes)/len(protein_genes)
    model_coefs.iloc[i, 7] = 0 if len(rna_genes) == 0 else len(common_genes)/len(rna_genes)


# # AB correlation
# 
# Next, let's get the correlation between protein and corresponding transcript:
# 
# This code was written by Arjana Begzati.

# In[172]:


import scipy
# from sklearn.metrics import normalized_mutual_info_score as nmi_score
from dcor import distance_correlation


# For duplicated gene names within a modality, keep the highest ranking occurence. 

# In[173]:


data_df = pd.read_csv(os.path.join(data_path, 'processed',  'expr_joint.csv'), index_col = 0)
ranked_fts_df = (
    model_coefs
    .drop_duplicates(subset=['Modality', 'Gene Name'], keep='first')
)
ranked_fts_df.loc[:, 'Feature Rank'] = range(1, ranked_fts_df.shape[0] + 1)

print('Of the {} total features, {} are retained when accounting for duplicate gene names.'.format(
    model_coefs.shape[0], ranked_fts_df.shape[0]))


# In[174]:


protein_names_dict = {} 
for protein_id in protein_cols:
    uniprot_id = protein_id.split('|')[1].split('-')[0]
    if pd.isna(uid_mapper[uniprot_id]):
        gene_name = protein_id.split('|')[-1].split('_HUMAN')[0]
        if gene_name[0].isdigit():
            gene_name = manual_map[uniprot_id]
    else:
        gene_name = uid_mapper[uniprot_id]
    protein_names_dict[protein_id] = gene_name

rna_names_dict = {rna_id:rna_id.split(' (')[0] for rna_id in rna_cols}

assert sorted(ranked_fts_df['Modality'].unique())==['Proteomics', 'Transcriptomics']
ranked_fts_df['Gene Name'] = np.where(ranked_fts_df['Modality']=='Transcriptomics',
                                 ranked_fts_df['Feature Name'].map(rna_names_dict), 
                                 ranked_fts_df['Feature Name'].map(protein_names_dict))
assert sum(ranked_fts_df['Gene Name'].isna())==0


# In[175]:


# number of proteins
print('The total # of protein features selected after accounting for duplicate mappings is: {}'.format(len(ranked_fts_df[ranked_fts_df['Modality']=='Proteomics']['Feature Name'].to_list())))


# number of proteins with rna feature
ranked_fts_subdf = ranked_fts_df.groupby('Gene Name').filter(lambda g: 
                                                      {'Transcriptomics', 'Proteomics'}.issubset(set(g['Modality'])))
n_shared = ranked_fts_subdf.shape[0]
per_modality = ranked_fts_subdf.Modality.value_counts()

msg = 'There are {} total features present in both transcriptomcis and proteomics, '.format(n_shared)
msg += '{} proteomics and {} transcriptomics'.format(per_modality.Proteomics, per_modality.Transcriptomics)
print(msg)

prots_in_rankedfts_w_rna = ranked_fts_subdf[ranked_fts_subdf['Modality']=='Proteomics']['Feature Name'].to_list()


# This tells us that nearly all the proteomics features included have a corresponding transcriptomics feature. 

# In[176]:


from typing import Literal
def pearsonr_omit_nan(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return stats.pearsonr(x[mask], y[mask])



def calc_corr(values_df, ft1_id, ft2_id, 
              corr_type: Literal['spearman', 'pearson', 'dcor'] = 'spearman', 
#               control_mean: bool = False,
              ft1_thresh = None, 
              ft2_thresh = None,
              min_samples = 10, # minimum non-nan samples
             ):

    ft1 = values_df.loc[:,ft1_id].values
    ft2 = values_df.loc[:,ft2_id].values

    if ft1_thresh is not None:
        ft1[ft1 < ft1_thresh] = np.nan
    if ft2_thresh is not None:
        ft2[ft2 < ft2_thresh] = np.nan

    mask = ~np.isnan(ft1) & ~np.isnan(ft2)
    if len(np.where(mask)[0]) < min_samples:
        return np.nan
    else:
        ft1 = ft1[mask]
        ft2 = ft2[mask]



    if corr_type == 'spearman':
        return stats.spearmanr(ft1, ft2).statistic
    elif corr_type == 'pearson':
        return stats.pearsonr(ft1, ft2).statistic
    elif corr_type == 'dcor':
        return distance_correlation(ft1, ft2)

#     if control_mean:
#         # covariate = per-sample mean abundance of the two features
#         temp_df = pd.DataFrame({
#             'ft1': ft1,
#             'ft2': ft2,
#             'mean_expr': 0.5 * (ft1 + ft2)
#         })


#         r,p = pg.partial_corr(data = temp_df, x='ft1', y='ft2',covar='mean_expr', method=corr_type).loc[
#     corr_type, ['r', 'p-val']].tolist()
#         return r, p

    else:
        if corr_type=='pearson':
            pearson_func = pearsonr_omit_nan if ft1_thresh is not None or ft2_thresh is not None else stats.pearsonr
            r, p = pearson_func(ft1, ft2) 
        elif corr_type=='spearman':
            nan_policy = 'omit' if ft1_thresh is not None or ft2_thresh is not None else 'propagate'
            r, p = scipy.stats.spearmanr(ft1, ft2, nan_policy = nan_policy)
        return r,p

def calc_overlapping_vs_not_mean_correlations(top_n, cor_df, cor_col, groupby = 'mean'):
    top_n_fts_df = ranked_fts_df.iloc[:top_n, :]

    rna_genes = set(top_n_fts_df[top_n_fts_df['Modality'] == 'Transcriptomics']['Gene Name'])
    protein_genes = set(top_n_fts_df[top_n_fts_df['Modality'] == 'Proteomics']['Gene Name'])

    common_genes = protein_genes.intersection(rna_genes)
    only_prot_genes = protein_genes.difference(rna_genes)

    numb_overlapping_genes = len(common_genes)
    numb_nonoverlapping_genes = len(only_prot_genes)

    overlapping_cor_df = cor_df[cor_df['Gene Name'].isin(common_genes)]
    nonoverlapping_cor_df = cor_df[cor_df['Gene Name'].isin(only_prot_genes)]

    if overlapping_cor_df.shape[0]>0:
        if groupby == 'mean':
            mean_overlapping_cor = overlapping_cor_df[cor_col].mean()
        elif groupby == 'median':
            mean_overlapping_cor = overlapping_cor_df[cor_col].median()

    else: 
        mean_overlapping_cor = np.nan
    if nonoverlapping_cor_df.shape[0]>0:
        if groupby == 'mean':
            mean_nonoverlapping_cor = nonoverlapping_cor_df[cor_col].mean()
        elif groupby == 'median':
            mean_nonoverlapping_cor = nonoverlapping_cor_df[cor_col].median()
    else: 
        mean_nonoverlapping_cor = np.nan
    return top_n, numb_overlapping_genes, mean_overlapping_cor, numb_nonoverlapping_genes, mean_nonoverlapping_cor


# Get correlation between features:

# In[177]:


# percentile thresholding prevents values by being driven by very low - expression features
percentile_thresh = 0.1
rna_thresh = expr_rna.quantile(percentile_thresh).median() #np.quantile(expr_rna.values.flatten(), 0.1)
protein_thresh = expr_protein.quantile(percentile_thresh).median() # np.quantile(expr_protein.values.flatten(), 0.1)


# In[178]:


pearson_corr_df = pd.DataFrame(columns=['prot_ft', 'rna_ft', 'r', 'p'])
for prot_ft in tqdm(prots_in_rankedfts_w_rna):
    gene_name = protein_names_dict[prot_ft]
    rna_df = ranked_fts_df[
        (ranked_fts_df['Gene Name']==gene_name) & (ranked_fts_df['Modality']=='Transcriptomics')
    ]
    assert rna_df.shape[0]==1
    rna_ft = rna_df['Feature Name'].values[0]
    r = calc_corr(values_df=data_df, ft1_id=prot_ft, ft2_id=rna_ft, 
                     corr_type='spearman', 
#                      control_mean = False, 
                     ft1_thresh = protein_thresh, 
                     ft2_thresh = rna_thresh,
                     min_samples = 10, 
                    )
    new_row = pd.DataFrame({'prot_ft': [prot_ft],  'rna_ft':[rna_ft], 'r': [r]})#, 'p': [p]})
    pearson_corr_df = pd.concat([pearson_corr_df, new_row], ignore_index=True)
pearson_corr_df['Gene Name'] = pearson_corr_df['rna_ft'].map(rna_names_dict)

# nan_mask = pearson_corr_df['r'].isna()
# print('{} of {} feature correlations are dropped because not enough samples were present after thresholds'.format(
#     len(nan_mask[nan_mask]), pearson_corr_df.shape[0]))
# pearson_corr_df = pearson_corr_df[~nan_mask].reset_index(drop = True)


# Add partial correlations from AB's analysis of expr_joint:

# In[179]:


# pc = pd.read_csv(os.path.join(data_path, 'interim', 'partial_correlations.csv'), index_col = None)
# pc_node_map =  pd.read_csv(os.path.join(data_path, 'interim', 'pc_node_map.csv'), index_col = None)
# pc_node_map = dict(zip(pc_node_map.node, pc_node_map.feature))

# pc.node1 = pc.node1.map(pc_node_map)
# pc.node2 = pc.node2.map(pc_node_map)

# def get_pcor(x):
#     mask = (pc.node1 == x.prot_ft) & (pc.node2 == x.rna_ft) | (pc.node1 == x.rna_ft) & (pc.node2 == x.prot_ft)

#     tot_hits = len(np.where(mask)[0])
#     if tot_hits == 0:
#         return np.nan
#     elif tot_hits != 1:
#         raise ValueError('Two correlation values present')
#     else:
#         return pc.loc[mask, 'pcor'].values[0]

# pc_keys = dict(zip(list(zip(pc.node1, pc.node2)), pc.pcor))
# pcor = pd.Series(zip(pearson_corr_df.prot_ft, pearson_corr_df.rna_ft)).map(pc_keys)
# pcor2 = pd.Series(zip(pearson_corr_df.rna_ft, pearson_corr_df.prot_ft)).map(pc_keys)
# assert pcor2.isna().all(), 'Need to account for both directions of nodes from partial correlation'
# pearson_corr_df['partial_correlation'] = pcor


# Split correlation as a rank order between overlapping and not:

# In[180]:


ranked_split_mean_pearson_corr_df = pd.DataFrame(columns=['Feature Rank', 
                                                          'Number of Overlapping Genes', 
                                                          'Overlapping Mean Pearson Correlation', 
                                                          'Number of Non-overlapping Genes',
                                                          'Non-overlapping Mean Pearson Correlation'])

for rank in tqdm(range(1, ranked_fts_df.shape[0]+1)):
    top_n, numb_o_genes, mean_o_cor, numb_no_genes, mean_no_cor = calc_overlapping_vs_not_mean_correlations(
        top_n=rank, cor_df=pearson_corr_df, 
        cor_col='r', 
        groupby = 'median') 
    ranked_split_mean_pearson_corr_df = pd.concat([ranked_split_mean_pearson_corr_df, 
                                                   pd.DataFrame(
                                                       [[top_n, numb_o_genes, mean_o_cor, numb_no_genes, mean_no_cor]], 
                                                        columns=ranked_split_mean_pearson_corr_df.columns)], 
                                                   ignore_index=True)
corr_rank = ranked_split_mean_pearson_corr_df.copy()



# In[181]:


# corr_rank['Difference'] = corr_rank['Overlapping Mean Pearson Correlation'] - corr_rank['Non-overlapping Mean Pearson Correlation']
# corr_rank['Normalized_Difference'] = corr_rank.Difference/corr_rank['Difference'].mean()




# In[182]:


fig, ax = plt.subplots(figsize = (6,5))
sns.lineplot(data = corr_rank, x = 'Feature Rank', y = 'Overlapping Mean Pearson Correlation',
             label='Overlapping Genes', ax = ax)
sns.lineplot(data = corr_rank, x = 'Feature Rank', y = 'Non-overlapping Mean Pearson Correlation',
             label='Non-overlapping Genes', ax = ax)
ax.axvline(x=500, color='grey', linestyle='--', label='Rank=500')
ax.set_ylabel('Median Spearman Correlation')
ax.set_xscale('log')
ax.legend(loc = 'lower right')

plt.savefig(os.path.join(data_path, 'figures', 'overlapping_pearson_running_rank.png'), 
            dpi=300, 
            bbox_inches="tight")
("")


# Across all feature ranks, those overlapping genes have a higher median spearman correlation than their non-overlapping counterpart. This tells us that the model is actually prioritizing features that reinforce similar biological signal across modalities, rather than selecting features that provide divergent, non-redundant information. 
# 
# This is even with the linear SVR regularization parameter, C, as freely tunable in the model. The pipeline selected C = 5.74, which does not enforce stringent regularization of model weights, indicating that the model chose a regularization that allows for multi-collinearity, and consequently, our observations. 
# 
# This result shows that, when predicting a y-block phenotype, non-redundancy defined purely by cross-modal correlation is not the dominant factor driving feature importance. In other words, a gene can have high positive correlation across modalities and yet still provide non-redundant information with regards to a y-block phenotype. In the next section, we consider exactly whether features contribute non-redundantly to y.  
# 
# In terms of multi-modal analysis, when considering a y-block phenotype, this analysis indicates that feature selection should not necessarily prioritize divergent features. 

# In[253]:


import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Pipeline instance is not fitted yet.*",
)

def fit_and_get_r2(X, y, best_pipeline):
    best_pipeline_remove = clone(best_pipeline)
    best_pipeline_remove.fit(X, y)
    y_pred = best_pipeline_remove.predict(X)
    return r2_score(y, y_pred)


def dominance_analysis_pair(prot_ft, rna_ft, r2_full):

    protein_cols_sub = [c for c in protein_cols if c != prot_ft]
    rna_cols_sub = [c for c in rna_cols if c != rna_ft]
    X_protein_sub = expr_joint[protein_cols_sub].values
    X_rna_sub = expr_joint[rna_cols_sub].values

    assert X_protein_sub.shape[1] == expr_protein.shape[1] - 1, 'Incorrect feature removal'
    assert X_rna_sub.shape[1] == expr_rna.shape[1] - 1, 'Incorrect feature removal'

    X_p = (X_protein, X_rna_sub)
    X_r = (X_protein_sub, X_rna)
    X_b = (X_protein_sub, X_rna_sub)


    r2_p = fit_and_get_r2(X_p, y, best_pipeline)
    r2_r = fit_and_get_r2(X_r, y, best_pipeline)
    r2_b = fit_and_get_r2(X_b, y, best_pipeline)

    u_r = r2_full - r2_p # rna contribution
    u_p = r2_full - r2_r # protein contribution
    s = r2_p + r2_r - r2_b
    return {
        'prot_ft': prot_ft, 
        'rna_ft': rna_ft,
            'R2_full': r2_full,
            'unique_rna_contribution': u_r,
            'unique_protein_contribution': u_p,
            'shared_contribution': s,
            'total pair contribution': r2_full - r2_b, #u_r + u_p + s
        }


# In[254]:


X = (X_protein, X_rna)
r2_full = fit_and_get_r2(X, y, best_pipeline)


# In[ ]:


X = (X_protein, X_rna)
r2_full = fit_and_get_r2(X, y, best_pipeline)


# In[271]:


# res_all = []
# for prot_ft in tqdm(prots_in_rankedfts_w_rna):
#     gene_name = protein_names_dict[prot_ft]
#     rna_df = ranked_fts_df[
#         (ranked_fts_df['Gene Name']==gene_name) & (ranked_fts_df['Modality']=='Transcriptomics')
#     ]
#     assert rna_df.shape[0]==1
#     rna_ft = rna_df['Feature Name'].values[0]

#     res = dominance_analysis_pair(prot_ft, rna_ft, r2_full)
#     res_all.append(res)


from joblib import Parallel, delayed
from tqdm import tqdm



def dominance_wrapper(prot_ft):
    gene_name = protein_names_dict[prot_ft]

    rna_df = ranked_fts_df[
        (ranked_fts_df['Gene Name'] == gene_name) &
        (ranked_fts_df['Modality'] == 'Transcriptomics')
    ]


    rna_ft = rna_df['Feature Name'].values[0]
    return dominance_analysis_pair(prot_ft, rna_ft, r2_full)

# --- Parallel execution with progress bar ---
res_all = Parallel(n_jobs=n_cores)(   # -1 = use all cores
    delayed(dominance_wrapper)(prot_ft)
    for prot_ft in tqdm(prots_in_rankedfts_w_rna)
)

res_all = pd.DataFrame(res_all)
res_all.to_csv(os.path.join(data_path, 'interim', 'dominance_analysis.csv'))


# # you are here:
# - if results hold, can just talk about high correlation reinforcing, and correlation alone not being a proxy for "information" re y-block, which will be interesting. will need to think about the analysis see [here](https://chatgpt.com/c/69316c59-e31c-832e-b69a-db4915540841)

# In[ ]:





# In[ ]:





# In[61]:


fig, ax = plt.subplots(figsize = (15, 3.75), ncols = 3)

sns.lineplot(data = model_coefs, x = 'Feature Rank', y = 'Protein Feature Fraction', 
            ax = ax[0])
ax[0].set_xlabel('Rank-Ordered Features')
ax[0].set_xscale('log')
ax[0].set_ylabel('Fraction of Features \n Derived from Proteomics')
ax[0].axvline(x=top_prot_feature, color='red', linestyle='--')


sns.lineplot(data = model_coefs, x = 'Feature Rank', y = 'Shared Genes Fraction - Protein', 
             ax = ax[1])
ax[1].set_xlabel('Rank-Ordered Features')
ax[1].set_xscale('log')
ax[1].set_ylabel('Fraction of Proteomic Features \n Also Present in Transcriptomics')


# sns.lineplot(data = model_coefs, x = 'Feature Rank', y = 'Shared Genes Fraction - RNA', 
#              ax = ax[2])
# ax[2].set_xlabel('Rank-Ordered Features')
# ax[2].set_xscale('log')
# ax[2].set_ylabel('Fraction of Transcriptomic Features \n Also Present in Proteomics')

sns.lineplot(data = corr_rank, x = 'Feature Rank', y = 'Normalized_Difference', ax = ax[2])
xmin, xmax = ax[2].get_xlim()
ymin, ymax = ax[2].get_ylim()
ax[2].hlines(1, xmin = xmin, xmax = xmax, color = 'gray', linestyle = '--')
ax[2].vlines(500, ymin = ymin, ymax = ymax, color = 'red', linestyle = '--')
ax[2].set_xscale('log')

ax[2].set_xlim(xmin, xmax)
ax[2].set_ylim(ymin, ymax)
ax[2].set_ylabel('Normalized Correlation Difference')
ax[2].set_xlabel('Rank-Ordered Features')


fig.tight_layout()
plt.savefig(os.path.join(data_path, 'figures', 'joint_feature_running_rank.png'), 
            dpi=300, 
            bbox_inches="tight")

("")


# In[20]:


msg = 'Whereas all {} RNA features were selected in the model, '.format(len(selected_rna_cols))
msg += 'only {} of the protein features were selected'.format(len(selected_protein_cols))
protein_frac_all = 100*len(selected_protein_cols)/len(selected_cols)
msg += '. Thus, protein features comprise {:.2f}% of all features used in the model'.format(protein_frac_all) 

protein_frac = 100*model_coefs[model_coefs['Feature Rank'] == top_n]['Protein Feature Fraction'].tolist()[0]
msg +='. They comprise {:.2f}% of the top 500 features'.format(protein_frac)

print(msg)
print('')
print('The fraction of features derived from protein peaks at the top {} feature at {:.2f}%'.format(top_prot_feature, 100 * model_coefs['Protein Feature Fraction'].max()))




# Regarding the first panel, this suggests a weighting of proteomics in the top features, which aligns with the fact that they help improve predictive performance.

# In[21]:


all_features_shared_prot = 100*model_coefs['Shared Genes Fraction - Protein'].tolist()[-1]
top_n_shared_prot = 100*model_coefs[model_coefs['Feature Rank'] == top_n]['Shared Genes Fraction - Protein'].tolist()[0]
max_frac_shared_prot = 100*model_coefs[model_coefs['Feature Rank'] == top_prot_feature]['Shared Genes Fraction - Protein'].tolist()[0]
all_shared_prot = [all_features_shared_prot, top_n_shared_prot, max_frac_shared_prot]

msg = 'The percent of proteomics features that are in common with transcriptomics is'
msg += ' {:.2f}%, {:.2f}%, {:.2f}%'.format(*all_shared_prot)
msg += ' for all features used by the model, the top 500 features, '
msg += 'and the top {} features (where the fraction of features that is dervied from proteomics is maximized)'.format(top_prot_feature)
msg += ' , respectively.'
print(msg)


# Regarding the second panel, this tells us that while features representing the same gene between transcriptomics and proteomics do add information,they are substantially less heavily favored in the more influential set of top-ranked features (fewer features are shared in the top 500 than in all features used in the dataset).

# Let's quantitate this using over-representation analysis:
# 
# For the fraction of proteomics, we can test whether there is an enrichment of proteomic features in the top 500 features. A contingency matrix testing for enrichment of Pathway A in a gene list will look as follows:
# 
# enrichment of missing/imputed values for the under-predicted genes. 
# 
# |                   | In Pathway A               | Not in Pathway A               |
# |-------------------|----------------------------|--------------------------------|
# | Pathway A         | Count of genes in both       | Count of genes in "Pathway A"  |
# |                   | "Pathway A" and your gene    | but not in your gene list      |
# |                   | list (observed)             | (expected under null hypothesis)|
# |-------------------|----------------------------|--------------------------------|
# | Not Pathway A     | Count of genes not in        | Count of genes not in          |
# |                   | "Pathway A" but in your      | "Pathway A" and not in your     |
# |                   | gene list (observed)         | gene list (expected under null  |
# |                   |                            | hypothesis)                     |
# 
# Here we test for over-representation of the proteomic modality in the top ranked features:
# 
# | Modality            | Top *n* Features | Outside Top *n* | Total |
# |---------------------|------------------|-----------------|-------|
# | Proteomics          | a                | b               | a + b |
# | Transcriptomics     | c                | d               | c + d |
# | Total               | a + c            | b + d           | N     |
# 
# 
# - a: Proteomic features within the top n ranks.
# - b: Proteomic features outside the top n ranks.
# - c: Transcriptomic features within the top n ranks.
# - d: Transcriptomic features outside the top n ranks.
# 
# 
# We also test for depletion of proteomic features that have a transcriptomic counterpart:
# | Proteomic features  | Top *n* Features | Outside Top *n* | Total |
# |---------------------|------------------|-----------------|-------|
# | Overlapping         | a                | b               | a + b |
# | Non-overlapping     | c                | d               | c + d |
# | Total               | a + c            | b + d           | N     |
# 
# 
# - a: Proteomic features overlapping with transcriptomic features within the top n ranks.
# - b: Proteomic features overlapping with transcriptomic features outside the top n ranks.
# - c: Proteomic features not overlapping with transcriptomic features within the top n ranks.
# - d: Proteomic features not overlapping with transcriptomic features outside the top n ranks.
# 

# In[22]:


def proteomic_ora(model_coefs, n, verbose = False):
    # Define contingency table
    top_n = model_coefs[model_coefs["Feature Rank"] <= n]
    outside_top_n = model_coefs[model_coefs["Feature Rank"] > n]

    # Calculate counts for contingency table
    a = (top_n["Modality"] == "Proteomics").sum() # proteomic features in top n
    c = (top_n["Modality"] == "Transcriptomics").sum() 
    b = (outside_top_n["Modality"] == "Proteomics").sum()
    d = (outside_top_n["Modality"] == "Transcriptomics").sum()

    N = model_coefs.shape[0] # total features
    K = a + b # total number of protein features
    k = a # total number of protein features in top n

#     hypergeom_p = hypergeom(M=N, n=K, N=n).sf(k-1)

    # Calculate odds ratio
    odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]], alternative = 'greater')
    if verbose: 
        msg = 'At a feature rank of {}'.format(n)
        msg += " the Fisher's exact test p-value for over-representation of proteomic features is"
        msg += " {:.3f}, with an odds ratio of {:.2f}".format(fisher_p, odds_ratio)
        print(msg)
    else: 
        return odds_ratio, fisher_p#, hypergeom_p


# def shared_ora(model_coefs, n, verbose = False):
#     # Define contingency table
#     top_n = model_coefs[model_coefs["Feature Rank"] <= n]
#     outside_top_n = model_coefs[model_coefs["Feature Rank"] > n]

#     _, prot_genes_top, common_genes_top, diff_genes_top = get_shared_genes(top_n, get_prot_difference = True)
#     _, prot_genes_out, common_genes_out, diff_genes_out = get_shared_genes(outside_top_n, get_prot_difference = True)


#     # Calculate counts for contingency table
#     a = len(diff_genes_top)
#     c = len(common_genes_top)
#     b = len(diff_genes_out)
#     d = len(common_genes_out)

#     # Calculate odds ratio
#     odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]], alternative = 'greater')


#     if verbose: 
#         msg = 'At a feature rank of {}'.format(n)
#         msg += " the Fisher's exact test p-value for over-representation of proteomic features that"
#         msg += " are not shared with transcriptomic features is"
#         msg += " {:.3e}, with an odds ratio of {:.2f}".format(fisher_p, odds_ratio)
#         print(msg)
#     else: 
#         return odds_ratio, fisher_p#, hypergeom_p



def shared_ora(model_coefs, n, verbose = False):
    # Define contingency table
    top_n = model_coefs[model_coefs["Feature Rank"] <= n]
    outside_top_n = model_coefs[model_coefs["Feature Rank"] > n]

    _, prot_genes_top, common_genes_top, diff_genes_top = get_shared_genes(top_n, get_prot_difference = True)
    _, prot_genes_out, common_genes_out, diff_genes_out = get_shared_genes(outside_top_n, get_prot_difference = True)


    # Calculate counts for contingency table
    a = len(common_genes_top) 
    c = len(diff_genes_top)
    b = len(common_genes_out)
    d = len(diff_genes_out)

    # Calculate odds ratio
    odds_ratio, fisher_p = fisher_exact([[a, b], [c, d]], alternative = 'less')


    if verbose: 
        msg = 'At a feature rank of {}'.format(n)
        msg += " the Fisher's exact test p-value for depletion of proteomic features that"
        msg += " are shared with transcriptomic features is"
        msg += " {:.3e}, with an odds ratio of {:.2f}".format(fisher_p, odds_ratio)
        print(msg)
    else: 
        return odds_ratio, fisher_p



# In[23]:


shared_ora(model_coefs, n = 500, verbose = True)


# In[24]:


proteomic_ora(model_coefs, n = top_prot_feature, verbose = True)


# In[25]:


proteomic_ora(model_coefs, n = 500, verbose = True)


# In[53]:


ora_proteomics_res = pd.DataFrame(columns = ['rank', 'odds_ratio', 'fisher_exact_p'])#, 'hypergeometric_p'])
for i, n in enumerate(range(650, 500, -1)):
    ora_proteomics_res.loc[i, :] = [n] + list(proteomic_ora(model_coefs, n))
ora_proteomics_res['-log(p-value)'] = -np.log(ora_proteomics_res.fisher_exact_p.astype(float))
ora_proteomics_res['odds_ratio'] = ora_proteomics_res.odds_ratio.apply(lambda x: float(x))
ora_proteomics_res['rank'] = ora_proteomics_res['rank'].astype(int)


# In[121]:


fig, ax = plt.subplots(ncols = 2, figsize = (10, 5))

# sns.scatterplot(data = ora_proteomics_res, x = 'rank', y = 'odds_ratio', 
#                 color = 'black',
#                 ax = ax[0])
sns.regplot(data = ora_proteomics_res, x = 'rank', y = 'odds_ratio', ci = None, 
            scatter = False, lowess = True, ax = ax[0])

ax[0].set_ylabel("Odds Ratio")
# sns.scatterplot(data = ora_proteomics_res, x = 'rank', y = '-log(p-value)', ax = ax[1])
sns.regplot(data = ora_proteomics_res, x = 'rank', y = '-log(p-value)', ci = None, 
            scatter = False, lowess = True, ax = ax[1])
# ax[1].axhline(y=-np.log(0.05), color='red', linestyle='dashed')


for i in range(2):
    ax[i].set_xlabel('Feature Rank')
    ax[i].invert_xaxis()


fig.suptitle("Fisher's Exact Test")
fig.tight_layout()
plt.savefig(os.path.join(data_path, 'figures', 'joint_feature_fisher_trend.png'), 
            dpi=300, 
            bbox_inches="tight")


# We can repeat this for the number of proteomic features shared with transcriptomic features:

# Finally, let's look at how the features selected by the joint omics model compare to that of the transcriptomics only model. The one caveat in this comparison is that both models were fit to their entire respective dataset, which is 247 samples for the join omics model and 483 samples for the transcriptomics-only model.

# # to do: comparison with transcriptomics
# - use full dataset, or identify consensus features in transcriptomics of a sample subset of equal size?
# - in the joint model when protein features are present that are not in common with transcriptomics features, are those transcriptomic features present in the transcriptomics only model? this tells us those genes are informative, but more i

# In[48]:


# fns = [os.path.join(data_path, 'processed',  'expr_joint.csv'),
# os.path.join(data_path, 'processed', 'uniprot_mapper.json'),
# os.path.join(data_path, 'processed', 'rank_ordered_joint_features.csv'),
# '/home/hmbaghda/Projects/metastatic_potential/notebooks/C_joint_prediction/*.txt']

# for fn in fns:
#     cmd = 'scp hmbaghda@satori-login-002.mit.edu:' + fn
#     cmd += ' Downloads/for_Arjana/.'
#     print(cmd)


# ## ORA

# We will run ORA on the positive and negative coefficients from the top 500 features separately. 

# In[12]:


top_model_coefs = model_coefs.iloc[:top_n].copy()
neg_top = top_model_coefs[top_model_coefs['SVM coefficient'] < 0]
pos_top = top_model_coefs[top_model_coefs['SVM coefficient'] > 0]

background = sorted(set(protein_names).union(rna_names))
fn_oras = []
# prepare for metascape input
for col_name, gene_list in {'negative': neg_top, 'positive': pos_top}.items():
    mi = pd.DataFrame(data = {'_BACKGROUND': background})

    goi = gene_list['Gene Name'].tolist()
    mi[col_name] = goi + [np.nan]*(len(background)- len(goi))

    mi.set_index('_BACKGROUND', inplace = True)

    fn_ora = os.path.join(data_path, 'interim/', col_name + '_joint_metascape_input.csv')
    fn_oras.append(fn_ora)
    mi.to_csv(fn_ora)


# In[56]:


# for fn in fn_oras:
#     cmd = 'scp hmbaghda@satori-login-002.mit.edu:' + fn
#     cmd += ' Downloads/.'
#     print(cmd)


# Load and format metascape output:

# In[18]:


ms = pd.read_excel(os.path.join(data_path, 'processed', key + '_joint_metascape_results.xlsx'), 
                       sheet_name = 'Enrichment',
                       index_col = None)
ms_all = pd.read_csv(os.path.join(data_path, 'processed', key + '_joint_GO_AllLists.csv'), 
                    index_col = 0)

# get average z-score per summary term from the individual members
ms_members = ms[ms['GroupID'].apply(lambda x: x.endswith('_Member'))].copy()
if ms_members.Term.nunique() != ms_members.shape[0]:
    raise ValueError('Expected unique terms')


# In[ ]:





# In[ ]:





# In[115]:


mss = {}
sort_by = 'pval' #'zscore'
for key in ['negative', 'positive']:
    ms = pd.read_excel(os.path.join(data_path, 'processed', key + '_joint_metascape_results.xlsx'), 
                           sheet_name = 'Enrichment',
                           index_col = None)
    ms_all = pd.read_csv(os.path.join(data_path, 'processed', key + '_joint_GO_AllLists.csv'), 
                        index_col = 0)

    # get average z-score per summary term from the individual members
    ms_members = ms[ms['GroupID'].apply(lambda x: x.endswith('_Member'))].copy()
    if ms_members.Term.nunique() != ms_members.shape[0]:
        raise ValueError('Expected unique terms')

    zscores = []
    for term in ms_members.Term:
        ms_term = ms_all[ms_all.GO == term]
        if ms_term.shape[0] > 1:
            raise ValueError('Expected 1 unique term')
        elif ms_term.shape[0] == 0:
            zscores.append(np.nan)
        else:
            zscores.append(ms_term['Z-score'].values.tolist()[0])
    ms_members['Z-score'] = zscores

    term_zscore = ms_members.groupby('GroupID')['Z-score'].mean().to_dict()
    term_zscore = {k.split('_')[0]:v for k,v in term_zscore.items()}

    ms = ms[ms['GroupID'].apply(lambda x: x.endswith('_Summary'))]
    ms.reset_index(drop = True, inplace = True)
    ms['Z-score'] = ms['GroupID'].apply(lambda x: term_zscore[x.split('_')[0]])


    # formatting
    ms['q-value'] = 10**ms['Log(q-value)']
    ms = ms[ms['q-value'] <= 0.1]
    ms['-Log10(q-value)'] = -ms['Log(q-value)']
    if ms['Z-score'].min() < 0:
        raise ValueError('Sorted according to assumption that all Z-scores to be positive')
    if sort_by == 'zscore':
        ms.sort_values(by = ['Z-score','Log(q-value)'], ascending=[False, True], inplace = True) 
    elif sort_by == 'pval':
        # sort by q-value, tie break with z-score
        ms.sort_values(by = ['Log(q-value)', 'Z-score'], ascending=[True, False], inplace = True) 
    else:
        raise ValueError('Specificy a sort column')

    ms.reset_index(drop = True, inplace = True)
    mss[key] = ms


# In[116]:


def middle_break(my_string, len_thresh):
    if len(my_string) >= len_thresh:
        middle_index = len(my_string) // 2

        space_before = my_string.rfind(' ', 0, middle_index)  
        space_after = my_string.find(' ', middle_index)       

        # Choose the nearest space
        if space_before == -1:  
            break_index = space_after
        elif space_after == -1:  
            break_index = space_before
        else:  
            break_index = space_before if middle_index - space_before <= space_after - middle_index else space_after

        new_string = my_string[:break_index] + '\n' + my_string[break_index + 1:]
    else:
        new_string = my_string

    return new_string


# In[117]:


fig, ax = plt.subplots(figsize = (10, 5))

key = 'negative'
top_n = 10
ms = mss[key]
ms_topn = ms.iloc[:top_n, :]

if sort_by == 'zscore':
    color_col = '-Log10(q-value)'
    y_col = 'Z-score'
elif sort_by == 'pval':
    color_col = 'Z-score' 
    y_col = '-Log10(q-value)'
norm = Normalize(vmin=ms_topn[color_col].min(), vmax=ms_topn[color_col].max())
cmap = cm_matplot.get_cmap('YlOrBr_r')
colors = [cmap(norm(val)) for val in ms[color_col]]
cbar_ax = fig.add_axes([0.92, 0.525, 0.02, 0.425])  # [left, bottom, width, height]
# pos = ax.get_position()  # Returns a Bbox object
# cbar_width = 0.02  # Width of the colorbar
# cbar_padding = 0.02  # Space between subplot and colorbar
# cbar_ax = fig.add_axes([pos.x1 + cbar_padding, pos.y0, cbar_width, pos.height])  # [left, bottom, width, height]
colorbar = plt.colorbar(cm_matplot.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
colorbar.set_label(color_col)


sns.barplot(data = ms_topn, x = 'Description', y = y_col, 
            palette = colors, ax = ax)

xlabels = []
for x_label in ax.get_xticklabels():
    xlabels.append(middle_break(x_label._text, len_thresh = 35))

ax.set_xticklabels(xlabels, 
                   rotation=30, va = 'top',
                   ha="right", rotation_mode="anchor")
# ax.get_legend().remove()
ax.set_title('Top Enriched Terms: ' + key.capitalize() + ' Features')
ax.set_xlabel('Enriched Term')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
plt.savefig(os.path.join(data_path, 'figures', 'joint_feature_negativeora.png'), 
            dpi=300, 
            bbox_inches="tight")


# In[ ]:


os.path.join(data_path, 'figures', 'joint_feature_negativeora.png')


# In[118]:


fig, ax = plt.subplots(figsize = (10, 5))

key = 'positive'
top_n = 10
ms = mss[key]
ms_topn = ms.iloc[:top_n, :]

if sort_by == 'zscore':
    color_col = '-Log10(q-value)'
    y_col = 'Z-score'
elif sort_by == 'pval':
    color_col = 'Z-score' 
    y_col = '-Log10(q-value)'
norm = Normalize(vmin=ms_topn[color_col].min(), vmax=ms_topn[color_col].max())
cmap = cm_matplot.get_cmap('YlOrBr_r')
colors = [cmap(norm(val)) for val in ms[color_col]]
cbar_ax = fig.add_axes([0.92, 0.525, 0.02, 0.425])  # [left, bottom, width, height]
# pos = ax.get_position()  # Returns a Bbox object
# cbar_width = 0.02  # Width of the colorbar
# cbar_padding = 0.02  # Space between subplot and colorbar
# cbar_ax = fig.add_axes([pos.x1 + cbar_padding, pos.y0, cbar_width, pos.height])  # [left, bottom, width, height]
colorbar = plt.colorbar(cm_matplot.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
colorbar.set_label(color_col)


sns.barplot(data = ms_topn, x = 'Description', y = y_col, 
            palette = colors, ax = ax)

xlabels = []
for x_label in ax.get_xticklabels():
    xlabels.append(middle_break(x_label._text, len_thresh = 35))

ax.set_xticklabels(xlabels, 
                   rotation=30, va = 'top',
                   ha="right", rotation_mode="anchor")
# ax.get_legend().remove()
ax.set_title('Top Enriched Terms: ' + key.capitalize() + ' Features')
ax.set_xlabel('Enriched Term')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
plt.savefig(os.path.join(data_path, 'figures', 'joint_feature_positiveora.png'), 
            dpi=300, 
            bbox_inches="tight")


# In[24]:


# fns = [os.path.join(data_path, 'figures', 'joint_feature_dist.png'), 
# os.path.join(data_path, 'processed', 'rank_ordered_joint_features.csv'),
# os.path.join(data_path, 'figures', 'joint_feature_running_rank.png'),
# os.path.join(data_path, 'figures', 'joint_feature_fisher_trend.png'),
# os.path.join(data_path, 'figures', 'joint_feature_negativeora.png'),
# os.path.join(data_path, 'figures', 'joint_feature_positiveora.png')]

fns = [os.path.join(data_path, 'figures', 'joint_feature_*'), 
      os.path.join(data_path, 'figures', 'overlapping_pearson_running_rank.png')]

for fn in fns:
    cmd = 'scp hmbaghda@satori-login-002.mit.edu:' + fn
    cmd += ' Downloads/figures/.'
    print(cmd)


# In[63]:


fns = [os.path.join(data_path, 'figures', 'overlapping_pearson_running_rank.png'),
os.path.join(data_path, 'figures', 'joint_feature_running_rank.png')]

for fn in fns:
    cmd = 'scp hmbaghda@satori-login-002.mit.edu:' + fn
    cmd += ' Downloads/figures/.'
    print(cmd)


# In[ ]:




