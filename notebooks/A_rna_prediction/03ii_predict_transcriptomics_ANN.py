"""Runs prediction on same folds as Notebook 03, using single or ensemble ANNs."""

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

### Adapted from https://github.com/NickMeim/deepSIBA_pytorch/blob/main/utility/gaussian.py
class GaussianLayer(torch.nn.Module):
    def __init__(self, output_dim,input_shape, dtype):
        self.output_dim = output_dim
        self.input_shape=input_shape
        super(GaussianLayer, self).__init__()
        self.linear1=torch.nn.Linear(input_shape, output_dim, bias=True,dtype=dtype)
        self.linear2=torch.nn.Linear(input_shape, output_dim, bias=True,dtype=dtype)
        torch.nn.init.xavier_normal_(self.linear1.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.xavier_normal_(self.linear2.weight,gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(self.linear2.bias)
    def forward(self, x):
        output_mu  = self.linear1(x)
        output_sig = self.linear2(x)
        output_sig_pos = torch.log(1 + torch.exp(output_sig)) + 1e-06  
        return output_mu, output_sig_pos
    
def custom_loss(y_true, y_pred,sigma):
    return torch.mean(0.5*torch.log(sigma) + 0.5*torch.div(torch.square(y_true - y_pred), sigma)) + 1e-6

### Initialize the parsed arguments
parser = argparse.ArgumentParser(description='Run ensemble learning approach')
parser.add_argument('--number_of_ensemble_members', action='store', type=int,help='number of models to use in ensemble',default=100)
#parser.add_argument('--X_train_path', action='store', type=str,help='path to the input training data',default='X_train_val_mean_centered.csv')
#parser.add_argument('--Y_train_path', action='store', type=str,help='path to the output training data',default='y_train_val_mean_centered.csv')
#parser.add_argument('--X_test_path', action='store', type=str,help='path to the input test data',default='X_test_mean_centered.csv')
#parser.add_argument('--Y_test_path', action='store', type=str,help='path to the output test data',default='y_test_mean_centered.csv')
parser.add_argument('--X_all_path', action='store', type=str,help='path to the input training data',default='new_files/expr.csv')
parser.add_argument('--Y_all_path', action='store', type=str,help='path to the output training data',default='new_files/metastatic_potential.csv')
parser.add_argument('--num_folds', action='store', type=int,help='number of folds',default=10)
parser.add_argument('--read_folds',action='store', type=str ,help='A location to read folds from a json file or None otherwise',default = 'new_files/transcriptomics_consensus_folds.json')
parser.add_argument('--res_dir', action='store', type=str,help='Results directory',default='new_results/MLresults/EnsembleLearning/')
parser.add_argument('--epochs', action='store', type=int,help='Number of epochs',default=100)
parser.add_argument('--l2_reg', action='store', type=float,help='L2 regularization',default=0.01)
parser.add_argument('--bs', action='store', type=int,help='Batch size',default=40)
parser.add_argument('--seed', action='store', type=int,help='seed',default=42)
args = parser.parse_args()
num_folds = args.num_folds
X_all_path = args.X_all_path
Y_all_path = args.Y_all_path
res_dir= args.res_dir
number_of_ensemble_members = args.number_of_ensemble_members
epochs = args.epochs
l2_reg = args.l2_reg
bs = args.bs
seed = args.seed
read_folds = args.read_folds
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
Path(res_dir+'predictions/').mkdir(parents=True, exist_ok=True)
res_all = pd.DataFrame({})
df_predictions_cv = pd.DataFrame({})
val_r_ensemble = []
train_r_ensemble = []
print2log('Begin cross %s-fold validation with an ensemble of %s models'%(num_folds,number_of_ensemble_members))
for i, fold_ind in enumerate(cv):
        train_index = cv[fold_ind]['train_idx']
        test_index = cv[fold_ind]['test_idx']
        x_train = X.iloc[train_index,:].values
        x_val = X.iloc[test_index,:].values
        y_train = Y.iloc[train_index,:].values
        y_val = Y.iloc[test_index,:].values

        val_r = []
        train_r = []
        pred_val_mus = np.zeros((y_val.shape[0],number_of_ensemble_members))
        pred_val_sigmas = np.zeros((y_val.shape[0],number_of_ensemble_members))
        pred_train_mus = np.zeros((y_train.shape[0],number_of_ensemble_members))
        pred_train_sigmas = np.zeros((y_train.shape[0],number_of_ensemble_members))
        for model_no in range(number_of_ensemble_members):
            model_seed = int(model_no) * (int(epochs) + 1)
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            set_seeds(seed = model_seed)
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
                                        GaussianLayer(output_dim=1,input_shape=16,dtype=torch.double)).to(device)
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
                    Yhat_mu, Yhat_sigma = model(dataIn)
                    fitLoss = custom_loss(dataOut, Yhat_mu,Yhat_sigma)
                    RegLoss = L2Regularization(model,L2=l2_reg)
                    loss = fitLoss + RegLoss
                    loss.backward()
                    optimizer.step()
                    all_losses.append(loss.item())
                    r = torch.mean(pearson_r(dataOut, Yhat_mu))
                    all_r.append(r.item())
                if(e%40==0 or e==0 or e==epochs-1):
                    print2log('Fold {}, Ensemble member {}, Epoch {}/{} : Loss = {}, r = {}'.format(i,model_no,e+1,epochs,np.mean(all_losses),np.mean(all_r)))

            model.eval()
            yhat_train_mu,yhat_train_sigma = model(torch.tensor(x_train).to(device))
            yhat_val_mu,yhat_val_sigma = model(torch.tensor(x_val).to(device))
            # torch.save(model, res_dir+'models/'+'model'+str(model_no)+'_fold'+str(i)+'.pt')
            train_r.append(pearson_r(torch.tensor(y_train,dtype=torch.double).to(device), yhat_train_mu).mean().item())
            val_r.append(pearson_r(torch.tensor(y_val,dtype=torch.double).to(device), yhat_val_mu).mean().item())
            pred_train_sigmas[:,model_no] = yhat_train_sigma.detach().cpu().squeeze().numpy()
            pred_train_mus[:,model_no] = yhat_train_mu.detach().cpu().squeeze().numpy()
            pred_val_sigmas[:,model_no] = yhat_val_sigma.detach().cpu().squeeze().numpy()
            pred_val_mus[:,model_no] = yhat_val_mu.detach().cpu().squeeze().numpy()
    
        ## calculate ensemble performance and save predictions
        np.save(res_dir+'predictions/'+'pred_val_mus_fold'+str(i)+'.npy',pred_val_mus)
        np.save(res_dir+'predictions/'+'pred_val_sigmas_fold'+str(i)+'.npy',pred_val_sigmas)
        np.save(res_dir+'predictions/'+'pred_train_mus_fold'+str(i)+'.npy',pred_train_mus)
        np.save(res_dir+'predictions/'+'pred_train_sigmas_fold'+str(i)+'.npy',pred_train_sigmas)
        yhat_val = np.mean(pred_val_mus, axis=1)
        yhat_train = np.mean(pred_train_mus, axis=1)
        sigma_star_val = np.sqrt(np.mean(pred_val_sigmas + np.square(pred_val_mus), axis = 1) - np.square(yhat_val))
        cv_val = sigma_star_val/yhat_val
        sigma_star_train = np.sqrt(np.mean(pred_train_sigmas + np.square(pred_train_mus), axis = 1) - np.square(yhat_train))
        cv_train = sigma_star_train/yhat_train
        predictions_val = pd.DataFrame({'yhat':yhat_val,'sigma':sigma_star_val,'CV':cv_val,'y':y_val.squeeze()})
        predictions_val['fold'] = i
        predictions_val['set'] = 'test'
        predictions_train = pd.DataFrame({'yhat':yhat_train,'sigma':sigma_star_train,'CV':cv_train,'y':y_train.squeeze()})
        predictions_train['fold'] = i
        predictions_train['set'] = 'train'
        df_predictions_cv = pd.concat([df_predictions_cv,predictions_val,predictions_train])
        df_predictions_cv.to_csv(res_dir+'predictions/'+'df_predictions_CV.csv')
        rval = corr(y_val.squeeze(),yhat_val)
        rtrain = corr(y_train.squeeze(),yhat_train)
        train_r_ensemble = rtrain
        val_r_ensemble = rval

        print2log('Fold {}, train r = {}, val r = {}'.format(i,np.mean(train_r),np.mean(val_r)))
        print2log('Fold {}, ensemble train r = {}, ensemble val r = {}'.format(i,train_r_ensemble,val_r_ensemble))
        
        res_val = pd.DataFrame({'single model r':val_r,'model':[xx for xx in range(number_of_ensemble_members)]})
        res_val['ensemble_r'] = val_r_ensemble
        res_val['fold'] = i
        res_val['set'] = 'test'
        res_train =  pd.DataFrame({'single model r':train_r,'model':[xx for xx in range(number_of_ensemble_members)]})
        res_train['ensemble_r'] = train_r_ensemble
        res_train['fold'] = i
        res_train['set'] = 'train'
        df_res_cv = pd.concat([res_val,res_train])
        df_res_cv.to_csv(res_dir+'performance/'+'df_res_fold%s.csv'%(i))
        res_all = pd.concat([res_all,df_res_cv],axis=0)

res_all.to_csv(res_dir+'performance/'+'all_performance.csv')
df_predictions_cv.to_csv(res_dir+'predictions/'+'df_predictions_CV.csv')
# res_all = res_all.melt(id_vars=['fold','set'],var_name='type',value_name='r')