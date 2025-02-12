"""Runs power analysis on same folds as Notebook 05Ai using single ANN. Written by Nikos Meimetis."""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from pathlib import Path
import argparse
import plotnine as p9
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def set_seeds(seed: int=888):
    """Sets random seeds for torch operations.

    Parameters
    ----------
    seed : int, optional
        seed value, by default 888
    """
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ.keys():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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

def corr(x, y):
    r,_ = pearsonr(x, y)
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

### Initialize the parsed arguments
parser = argparse.ArgumentParser(description='Run power analysis')
parser.add_argument('--X_all_path', action='store', type=str,help='path to the input training data',default='new_files/expr.csv')
parser.add_argument('--Y_all_path', action='store', type=str,help='path to the output training data',default='new_files/metastatic_potential.csv')
parser.add_argument('--num_folds', action='store', type=int,help='number of folds',default=10)
parser.add_argument('--read_folds',action='store', type=str ,help='A location to read folds from a json file or None otherwise',default = 'new_files/transcriptomics_power_analysis_folds.json')
parser.add_argument('--res_dir', action='store', type=str,help='Results directory',default='new_results/MLresults/PoweAnalysisANN/')
parser.add_argument('--epochs', action='store', type=int,help='Number of epochs',default=100)
parser.add_argument('--l2_reg', action='store', type=float,help='L2 regularization',default=0.01)
parser.add_argument('--bs', action='store', type=int,help='Batch size',default=40)
parser.add_argument('--seed', action='store', type=int,help='seed',default=42)
args = parser.parse_args()
num_folds = args.num_folds
X_all_path = args.X_all_path
Y_all_path = args.Y_all_path
res_dir= args.res_dir
epochs = args.epochs
l2_reg = args.l2_reg
bs = args.bs
seed = args.seed
read_folds = args.read_folds
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print2log(device)
### Load the data
X = pd.read_csv(X_all_path,index_col=0)
Y = pd.read_csv(Y_all_path,index_col=0)
Y = Y.loc[X.index.values,["mean"]]
X = X - X.mean()

### Create 10-fold cross validation sets with KFold
with open(read_folds, "r") as json_file:
    cv = json.load(json_file)

### Initialize lists and matrices to save the results
Path(res_dir+'models/').mkdir(parents=True, exist_ok=True)
Path(res_dir+'performance/').mkdir(parents=True, exist_ok=True)
#res_all = pd.DataFrame({})
#res_all = pd.read_csv(res_dir+'performance/'+'df_res_fold%s_model%s.csv'%(1,7),index_col=0)
criterion = torch.nn.MSELoss(reduction='mean')
print2log('Begin cross %s-fold validation with power analysis'%num_folds)
for i, fold_ind in enumerate(cv):
        # train_index = cv[fold_ind]['train_idx']
        test_index = cv[fold_ind]['test_idx']
        x_val = X.iloc[test_index,:].values
        y_val = Y.iloc[test_index,:].values
        res_all = pd.DataFrame({})

        model_no = 0
        for train_key in list(cv[fold_ind].keys())[1:]:
            train_subset = cv[fold_ind][train_key]
            iteration = 0
            val_r = []
            train_r = []
            for selected_train_key in list(train_subset.keys()):
                train_index = np.array(train_subset[selected_train_key])
                x_train = X.iloc[train_index,:].values
                y_train = Y.iloc[train_index,:].values
                model_seed = int(model_no) * int(iteration)* (int(epochs) + 1)
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                set_seeds(seed = model_seed)
                #print2log(x_train.shape)
                model = torch.nn.Sequential(torch.nn.Dropout(0.5),
                                            torch.nn.Linear(x_train.shape[1],1024,bias=True,dtype=torch.double),
                                            torch.nn.BatchNorm1d(num_features=1024, momentum=0.2,dtype = torch.double),
                                            torch.nn.ELU(),
                                            torch.nn.Dropout(0.25),
                                            torch.nn.Linear(1024, 256,bias=True,dtype=torch.double),
                                            torch.nn.BatchNorm1d(num_features=256, momentum=0.2,dtype = torch.double),
                                            torch.nn.ELU(),
                                            torch.nn.Dropout(0.25),
                                            torch.nn.Linear(256, 32,bias=True,dtype=torch.double),
                                            torch.nn.BatchNorm1d(num_features=32, momentum=0.2,dtype = torch.double),
                                            torch.nn.ELU(),
                                            torch.nn.Dropout(0.25),
                                            torch.nn.Linear(32, 16,bias=True,dtype=torch.double),
                                            torch.nn.BatchNorm1d(num_features=16, momentum=0.2,dtype = torch.double),
                                            torch.nn.ELU(),
                                            torch.nn.Dropout(0.25),
                                            torch.nn.Linear(16, 1,bias=True,dtype=torch.double)).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                for e in range(epochs):
                    trainloader = getSamples(x_train.shape[0], bs)
                    all_losses=[]
                    all_r = []
                    for dataIndex in trainloader:
                        model.train()
                        optimizer.zero_grad()
                        dataIn = torch.tensor(x_train[dataIndex, :]).to(device)
                        dataOut = torch.tensor(y_train[dataIndex, :],dtype=torch.double).to(device)
                        Yhat = model(dataIn)
                        fitLoss = criterion(dataOut, Yhat)
                        RegLoss = L2Regularization(model,L2=l2_reg)
                        loss = fitLoss + RegLoss
                        loss.backward()
                        optimizer.step()
                        all_losses.append(loss.item())
                        r = torch.mean(pearson_r(dataOut, Yhat))
                        all_r.append(r.item())
                    if(e%40==0 or e==0 or e==epochs-1):
                        print2log('Fold {}, Model number {}, Iteration {}, Epoch {}/{} : Loss = {}, r = {}'.format(i,model_no,iteration,e+1,epochs,np.mean(all_losses),np.mean(all_r)))

                model.eval()
                yhat_train = model(torch.tensor(x_train).to(device))
                yhat_val = model(torch.tensor(x_val).to(device))
                # torch.save(model, res_dir+'models/'+'model'+str(model_no)+'_fold'+str(i)+'.pt')
                train_r.append(pearson_r(torch.tensor(y_train,dtype=torch.double).to(device), yhat_train).mean().item())
                val_r.append(pearson_r(torch.tensor(y_val,dtype=torch.double).to(device), yhat_val).mean().item())
                #torch.save(model, res_dir+'models/'+'model'+'_fold'+str(i)+'_subset'+str(train_key)+'_iteration'+str(iteration)+'.pt')
                iteration += 1
            df = pd.DataFrame({'train':train_r,'test':val_r})
            df['iteration'] = list(train_subset.keys())
            df['subset'] = train_key
            df = df.melt(id_vars=['iteration','subset'],var_name='set',value_name='r')
            res_all = pd.concat([res_all,df],axis=0)
            res_all.to_csv(res_dir+'performance/'+'df_res_fold%s_model%s.csv'%(i,model_no))
            model_no += 1
