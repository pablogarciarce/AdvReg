import numpy as np
import torch

def get_toy_data(n_samples=1000):
    # Create toy dataset with 2 features and dependence y = beta0*x1 + beta1*x2 + epsilon with
    # epsilon ~ N(0, 1) and beta0 ~ N(1, 1/2), beta1 ~ N(2, 1/2)
    np.random.seed(0)

    # X has uniform distribution in [0, 10] for each feature
    X = np.random.uniform(0, 10, (n_samples, 2))

    beta0 = np.random.normal(1, 1/2, n_samples)
    beta1 = np.random.normal(2, 1/2, n_samples)

    y = beta0*X[:, 0] + beta1*X[:, 1] + np.random.normal(0, 1, n_samples)

    return X, y

def id(x, y):
    return y

def expy2(x, y):
    return torch.exp(y ** 2 / 100)