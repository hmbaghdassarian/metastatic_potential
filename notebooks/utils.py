"""Utility functions for metastatic potential predictive modeling"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression

import pathlib
import pickle

import optuna
from optuna.samplers import CmaEsSampler, TPESampler, RandomSampler
from optuna.distributions import CategoricalDistribution


def cohen_d(vector_1, vector_2):
    # Calculate the means of the two vectors
    mean1 = np.mean(vector_1)
    mean2 = np.mean(vector_2)

    # Calculate the standard deviations of the two vectors
    std1 = np.std(vector_1, ddof=1)
    std2 = np.std(vector_2, ddof=1)

    # Calculate the pooled standard deviation
    n1 = len(vector_1)
    n2 = len(vector_2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std

def mixup(X, y, n_synthetic, alpha=2, random_state=None):
    """
    Create synthetic samples using the mixup technique.

    Parameters:
    - n_synthetic (int): Number of synthetic samples to generate.
    - alpha (float): Parameter for the Beta distribution controlling the mixup ratio.
    - random_seed (int, optional): Random seed for reproducibility.

    Returns:
    - synthetic_data (np.ndarray): A 2D array of shape (n_synthetic, features) with synthetic samples.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    synthetic_X = np.zeros((n_synthetic, n_features))
    synthetic_y = np.zeros((n_synthetic, ))

    for i in range(n_synthetic):
        # Randomly select two samples to mix
        idx1, idx2 = np.random.choice(n_samples, size=2, replace=False)
        
        # Generate mixup coefficient from a Beta distribution
        lambda_ = np.random.beta(alpha, alpha)
        
        # Create a synthetic sample
        synthetic_X[i] = lambda_ * X[idx1] + (1 - lambda_) * X[idx2]
        synthetic_y[i] = lambda_ * y[idx1] + (1 - lambda_) * y[idx2]
    
    return synthetic_X, synthetic_y

def get_stats(model, y_train, y_test, X_train, X_test, get_cod = False):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_corr = pearsonr(y_train, y_train_pred)[0]
    test_corr = pearsonr(y_test, y_test_pred)[0]
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    if get_cod: # get coefficient of determination
        r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        return train_corr, test_corr, train_mse, test_mse, r2

    
    return train_corr, test_corr, train_mse, test_mse

def write_pickled_object(object_, file_name: str) -> None:
    if '.' in file_name:
        p = pathlib.Path(file_name)
        extensions = "".join(p.suffixes)
        file_name = str(p).replace(extensions, '.pickle')
    else:
        file_name = file_name + '.pickle'

    with open(file_name, 'wb') as handle:
        pickle.dump(object_, handle)
        
def read_pickled_object(file_name: str):
    # Ensure the file has the correct .pickle extension
    if '.' in file_name:
        p = pathlib.Path(file_name)
        extensions = "".join(p.suffixes)
        file_name = str(p).replace(extensions, '.pickle')
    else:
        file_name = file_name + '.pickle'

    # Read and deserialize the object
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)
        
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
    

def pearson_corr_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

class PLSRegression_X(PLSRegression):
    def transform(self, X, y=None):
        X_transformed = super().transform(X, y)
        if isinstance(X_transformed, tuple):
            X_transformed = X_transformed[0]
        return X_transformed