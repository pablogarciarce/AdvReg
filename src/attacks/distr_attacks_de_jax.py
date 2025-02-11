import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax import jit
import optax
from jax.scipy.stats import norm
import numpyro
from functools import partial
import numpy as np

from flax.core.frozen_dict import FrozenDict
from jax.tree_util import tree_map
from time import time


def kl_div(mu_n, lam_n, sigma2, x, x_adv):
    sigma2_A = jnp.dot(x_adv.T, jnp.linalg.inv(lam_n)) @ x_adv + sigma2
    mu_A = jnp.dot(x_adv.T, mu_n)
    sigma2_D = jnp.dot(x.T, jnp.linalg.inv(lam_n)) @ x + sigma2
    mu_D = jnp.dot(x.T, mu_n)
    kl = 0.5 * (jnp.log(sigma2_A / sigma2_D) + (sigma2_D + (mu_D - mu_A)**2) / sigma2_A - 1)
    return kl


def kl_to_appd(mu_A, sigma2_A, mu_D, sigma2_D):
    kl = 0.5 * (jnp.log(sigma2_A / sigma2_D) + (sigma2_D + (mu_D - mu_A)**2) / sigma2_A - 1)
    return kl

#@jit
def pi(rng, y, x, model, M):
    """
    Compute pi(y | x, gamma).
    """
    probs = model.sample_predictive_distribution(rng, x, num_samples=M)
    return probs[:, y]

#@jit
def grad_pi(rng, y, x, model, M):
    """
    Compute the gradient of pi(y | x, gamma) with respect to x.
    """
    def pi_loss(x):
        return pi(rng, y, x, model, M).mean() 
    
    return grad(pi_loss)(x)

#@jit
def g_x_M(rng, y, x, model, M):
    """
    Compute g_{x, M}(y).
    """
    rng, pi_rng = jax.random.split(rng)
    numerator = grad_pi(pi_rng, y, x, model, M)
    pi_vals = pi(rng, y, x, model, M)
    denominator = pi_vals.mean()
    return numerator / (denominator + 1e-8)


def delta_g_x_l(rng, y, x, l, model, M_sequence):
    """
    Compute Δg_{x, l}(y).
    """
    M_l = M_sequence[l]
    M_l_minus_1 = M_sequence[l-1] if l > 0 else 0
    
    rng, g_rng = jax.random.split(rng)
    g_l = g_x_M(g_rng, y, x, model, M_l)
    g_rng1, g_rng2 = jax.random.split(rng)
    g_l_minus_1_a = g_x_M(g_rng1, y, x, model, M_l_minus_1) if l > 0 else 0
    g_l_minus_1_b = g_x_M(g_rng2, y, x, model, M_l_minus_1) if l > 0 else 0
    return g_l - (g_l_minus_1_a + g_l_minus_1_b) / 2


def mlmc_gradient_estimator(y, x, R, model, M0=4, tau=2):
    """
    Estimate the gradient using MLMC.
    """
    M_sequence = [M0 * 2**l for l in range(17)]
    omega = jnp.array([2**(-tau * l) for l in range(len(M_sequence))])
    omega /= omega.sum()
    l_indices = jax.random.choice(jax.random.PRNGKey(0), len(M_sequence), shape=(R,), p=omega)
    estimates = jnp.array([delta_g_x_l(jax.random.PRNGKey(l), y, x, l, model, M_sequence) / omega[l] for l in l_indices])
    return estimates.mean(axis=0)


def mlmc_attack(model, x, appd=None, lr=0.01, n_iter=1000, epsilon=0.1, R=20, early_stopping_patience=10, verbose=True, optimizer="Adam"):
    """
    Perform the attack using the MLMC gradient estimator.
    :param appd: Attacker predictive posterior distribution to approximate. If None -> Maximum disruption attack.
    """
    # Add noise to the initial input
    rng = jax.random.PRNGKey(0)
    rng, noise_rng = jax.random.split(rng)
    x_adv = x + jax.random.normal(noise_rng, shape=x.shape) * 0.0001

    # Initialize optimizer
    if optimizer == "SGD":
        opt = optax.sgd(learning_rate=lr, momentum=0.5)
    elif optimizer == "Adam":
        opt = optax.adam(learning_rate=lr)
    else:
        raise ValueError("Optimizer not recognized")

    opt_state = opt.init(x_adv)

    # Store adversarial examples and set patience counter

    x_adv_values = []
    patience = 0

    def update_step(x_adv, opt_state, grad):
        """Update adversarial example using the optimizer."""
        updates, opt_state = opt.update(grad, opt_state, x_adv)
        x_adv = optax.apply_updates(x_adv, updates)
        return x_adv, opt_state

    for it in range(n_iter):
        # Compute gradient using MLMC
        rng, sample_rng = jax.random.split(rng)
        if appd is None:
            y = model.sample_predictive_distribution(sample_rng, x, num_samples=1).argmax()
            grad = mlmc_gradient_estimator(y, x_adv, R, model)
        else:
            y = appd.sample(sample_rng).argmax()
            grad = -mlmc_gradient_estimator(y, x_adv, R, model)
            
        # Perform optimization step
        x_adv, opt_state = update_step(x_adv, opt_state, grad)

        # Project to epsilon-ball to ensure perturbation constraints
        diff = x_adv - x
        norm_diff = jnp.linalg.norm(diff, ord=2)
        if norm_diff > epsilon:
            x_adv = x + epsilon * (diff / norm_diff)

        # Store adversarial example
        x_adv_values.append(np.array(x_adv))

        # Early stopping condition based on changes in adversarial examples
        if it > 2 and np.linalg.norm(x_adv_values[-1] - x_adv_values[-2]) < 1e-4:
            patience += 1
            if patience >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at iteration {it}")
                break
        else:
            patience = 0

    return x_adv, x_adv_values



def fgsm_attack(model, x, appd=None, lr=0.01, n_iter=1000, epsilon=0.1, R=100, early_stopping_patience=10):
    """
    FGSM attack using the MLMC gradient estimator.
    """
    x_adv = x + jax.random.normal(jax.random.PRNGKey(0), shape=x.shape) * 0.00001

    for it in range(n_iter):
        if appd is None:
            y = model.sample_predictive_distribution(x, num_samples=1)
        else:
            rng, sample_rng = jax.random.split(jax.random.PRNGKey(0))
            y = appd.sample(sample_rng)

        grad = mlmc_gradient_estimator(y, x_adv, R, model)
        if appd is None:
            grad_sign = jnp.sign(grad)
        else:
            grad_sign = -jnp.sign(grad)

        x_adv += lr * grad_sign
        x_adv = x + epsilon * (x_adv - x) / jnp.linalg.norm(x_adv - x, ord=2)

    return x_adv
