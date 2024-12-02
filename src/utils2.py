from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_ppds(model, x, x_adv_distr, rng, appd=None, num_samples=100000, ax=None):  
    y_samples = model.sample_predictive_distribution(x, num_samples=num_samples)
    y_adv_samples = model.sample_predictive_distribution(x_adv_distr, num_samples=num_samples)
    if appd is not None:
        y_appd_samples = appd.sample(rng, (num_samples, 1))
    print(y_samples.shape)
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