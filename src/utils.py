import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def plot_ppds(model, x, x_adv_distr, appd=None):
    y_samples = model.sample_predictive_distribution(x, num_samples=10000).numpy()
    y_adv_samples = model.sample_predictive_distribution(x_adv_distr, num_samples=10000).numpy()
    if appd is not None:
        y_appd_samples = appd.sample((10000,)).numpy()

    kde = gaussian_kde(y_samples.T)
    kde_adv = gaussian_kde(y_adv_samples.T)
    if appd is not None:
        kde_appd = gaussian_kde(y_appd_samples.T)

    plt.hist(y_samples, bins=50, alpha=0.5, label='Original', density=True)
    plt.hist(y_adv_samples, bins=50, alpha=0.5, label='Adversarial', density=True)
    if appd is not None:
        plt.hist(y_appd_samples, bins=50, alpha=0.5, label='Objective APPD', density=True)

    if appd is not None:
        ys = np.linspace(
        min(np.min(y_samples), np.min(y_adv_samples), np.min(y_appd_samples)), 
        max(np.max(y_samples), np.max(y_adv_samples), np.max(y_appd_samples)), 
        100)
    else:
        ys = np.linspace(min(np.min(y_samples), np.min(y_adv_samples)), max(np.max(y_samples), np.max(y_adv_samples)), 100)
    # plot with same color and label
    plt.plot(ys, kde(ys), color='C0')
    plt.plot(ys, kde_adv(ys), color='C1')
    if appd is not None:
        plt.plot(ys, kde_appd(ys), color='C2')
    plt.legend()
    plt.xlabel('y')
    plt.show()