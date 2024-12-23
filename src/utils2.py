from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_ppds(model, x, x_adv_distr, rng, appd=None, num_samples=100000, ax=None):  
    y_samples = model.sample_predictive_distribution(x, num_samples=num_samples).squeeze()
    y_adv_samples = model.sample_predictive_distribution(x_adv_distr, num_samples=num_samples).squeeze()
    if appd is not None:
        y_appd_samples = appd.sample(rng, (num_samples, 1)).squeeze()
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
    if norm_delta > epsilon:
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
