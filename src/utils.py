import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_ppds(model, x, x_adv_distr, rng, appd=None, num_samples=100000, ax=None):  
    y_samples = model.sample_predictive_distribution(x, num_samples=num_samples).squeeze()
    y_adv_samples = model.sample_predictive_distribution(x_adv_distr, num_samples=num_samples).squeeze()
    if appd is not None:
        y_appd_samples = appd.sample(rng, (num_samples,)).squeeze()
    kde = gaussian_kde(y_samples)
    kde_adv = gaussian_kde(y_adv_samples)
    if appd is not None:
        kde_appd = gaussian_kde(y_appd_samples)

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
    
def l2_projection(x, x_0, epsilon):
    """Realiza la proyección L2."""
    delta = x - x_0
    norm_delta = jnp.linalg.norm(delta, ord=2)
    delta = epsilon * delta / norm_delta
    return x_0 + delta


def l1_projection(x, x_0, epsilon):
    """Realiza la proyección L1."""
    delta = x - x_0
    abs_delta = jnp.abs(delta)
    if jnp.sum(abs_delta) > epsilon:
        sorted_delta = jnp.sort(abs_delta.ravel())[::-1]
        cumsum_delta = jnp.cumsum(sorted_delta)
        indices = jnp.arange(1, len(sorted_delta) + 1)
        rho = jnp.max(jnp.nonzero(sorted_delta * indices > (cumsum_delta - epsilon))[0])
        theta = (cumsum_delta[rho] - epsilon) / (rho + 1)
        delta = jnp.sign(delta) * jnp.maximum(abs_delta - theta, 0)
    return x_0 + delta


def get_toy_data(n_samples=1000):
    # Create toy dataset with 2 features and dependence y = beta0*x1 + beta1*x2 + epsilon with
    # epsilon ~ N(0, 1) and beta ~ N([1, 2], A) with A some random positive definite matrix
    np.random.seed(0)

    A = np.array([[1, 2], [3, 4]])
    Ad = A.T @ A
    X = np.random.multivariate_normal([0, 0], Ad, n_samples)

    beta = np.random.multivariate_normal([1, 2], np.eye(2))

    y = X @ beta + np.random.normal(0, 1, n_samples)

    return X, y

def get_toy_data_indep(n_samples=1000):
    # Create toy dataset with 2 features and dependence y = beta0*x1 + beta1*x2 + epsilon with
    # epsilon ~ N(0, 1) and beta ~ N([1, 2], A) with A some random positive definite matrix
    np.random.seed(0)

    Ad = np.eye(2)
    X = np.random.multivariate_normal([0, 0], Ad, n_samples)

    beta = np.array([-1, 2])

    y = X @ beta + np.random.normal(0, 1, n_samples)

    return X, y

def id(x, y):
    return y

def expy2(x, y):
    return jnp.exp(y ** 2 / 100)





############################################################################################################

def _torch_plot_ppds(model, x, x_adv_distr, appd=None, num_samples=100000, ax=None):  
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

def _torch_l2_projection(x, x_0, epsilon):
    return x_0 + epsilon * (x - x_0) / torch.norm(x - x_0, p=2)

def _torch_l1_projection(x, x_0, epsilon):
    delta = x - x_0  
    abs_delta = torch.abs(delta)  
    if torch.sum(abs_delta) > epsilon:
        sorted_delta, _ = torch.sort(abs_delta.view(-1), descending=True)
        cumsum_delta = torch.cumsum(sorted_delta, dim=0)
        rho = torch.nonzero(sorted_delta * torch.arange(1, len(sorted_delta)+1).float() > (cumsum_delta - epsilon)).max()
        theta = (cumsum_delta[rho] - epsilon) / (rho + 1)
        delta = torch.sign(delta) * torch.max(abs_delta - theta, torch.zeros_like(abs_delta))
    return x_0 + delta


