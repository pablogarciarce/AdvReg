import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def get_toy_data(n_samples=1000):
    # Create toy dataset with 2 features and dependence y = beta0*x1 + beta1*x2 + epsilon with
    # epsilon ~ N(0, 1) and beta ~ N([1, 2], A) with A some random positive definite matrix
    np.random.seed(0)

    A = np.array([[1, 2], [3, 4]])
    Ad = A @ A.T
    X = np.random.multivariate_normal([0, 0], Ad, n_samples)

    beta = np.random.multivariate_normal([1, 2], np.eye(2))

    y = X @ beta + np.random.normal(0, 1, n_samples)

    return X, y

def id(x, y):
    return y

def expy2(x, y):
    return torch.exp(y ** 2 / 100)

def plot_ppds(model, x, x_adv_distr, appd=None, num_samples=100000, ax=None):  
    y_samples = model.sample_predictive_distribution(x, num_samples=num_samples).numpy()
    y_adv_samples = model.sample_predictive_distribution(x_adv_distr, num_samples=num_samples).numpy()
    if appd is not None:
        y_appd_samples = appd.sample((num_samples,)).numpy()

    kde = gaussian_kde(y_samples.T)
    kde_adv = gaussian_kde(y_adv_samples.T)
    if appd is not None:
        kde_appd = gaussian_kde(y_appd_samples.T)

    if ax is None:
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

    else:
        ax.hist(y_samples, bins=50, alpha=0.5, label='Original', density=True)
        ax.hist(y_adv_samples, bins=50, alpha=0.5, label='Adversarial', density=True)
        if appd is not None:
            ax.hist(y_appd_samples, bins=50, alpha=0.5, label='Objective APPD', density=True)

        if appd is not None:
            ys = np.linspace(
            min(np.min(y_samples), np.min(y_adv_samples), np.min(y_appd_samples)), 
            max(np.max(y_samples), np.max(y_adv_samples), np.max(y_appd_samples)), 
            100)
        else:
            ys = np.linspace(min(np.min(y_samples), np.min(y_adv_samples)), max(np.max(y_samples), np.max(y_adv_samples)), 100)
        # plot with same color and label
        ax.plot(ys, kde(ys), color='C0')
        ax.plot(ys, kde_adv(ys), color='C1')
        if appd is not None:
            ax.plot(ys, kde_appd(ys), color='C2')
        ax.legend()
        ax.set_xlabel('y')
        return ax