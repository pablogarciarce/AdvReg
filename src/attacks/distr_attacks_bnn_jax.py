import jax
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax import jit
import optax
from jax.scipy.stats import norm
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

@jit
def pi(y, x, gamma):
    """
    Compute pi(y | x, gamma).
    """
    if 'sigma2' not in gamma:  # Classification
        w1 = gamma['w1']
        b1 = gamma['b1']
        w2 = gamma['w2']
        b2 = gamma['b2']
        hidden = jnp.dot(x, w1) + b1
        hidden = jax.nn.relu(hidden)
        logits = jnp.dot(hidden, w2) + b2
        logits = logits.diagonal(axis1=1, axis2=2)
        probs = jax.nn.softmax(logits, axis=1)
        probs = jnp.clip(probs, 1e-6, 1)  # Avoid zero probabilities
        probs = probs / probs.sum(axis=1, keepdims=True) # Normalize
        return probs[:, y, :]

    w1 = gamma['w1']
    b1 = gamma['b1']
    w2 = gamma['w2']
    b2 = gamma['b2']
    sigma2 = gamma['sigma2']
    
    hidden = jnp.dot(x, w1) + b1
    hidden = jax.nn.relu(hidden)
    mean = jnp.dot(hidden, w2.transpose(1, 0)) + b2
    mean = mean.diagonal(axis1=1, axis2=2)
    std = jnp.sqrt(sigma2)
    fy = norm.pdf(y, loc=mean, scale=std)
    return fy

@jit
def grad_pi(y, x, gamma):
    """
    Compute the gradient of pi(y | x, gamma) with respect to x.
    """
    def pi_loss(x):
        return pi(y, x, gamma).mean()
    
    return grad(pi_loss)(x)

@jit
def g_x_M(y, x, gamma_samples):
    """
    Compute g_{x, M}(y).
    """
    gamma_samples = FrozenDict(tree_map(jnp.array, gamma_samples))
    numerator = grad_pi(y, x, gamma_samples)
    pi_vals = pi(y, x, gamma_samples)
    denominator = pi_vals.mean()
    return numerator / (denominator + 1e-8)


def delta_g_x_l(rng, y, x, l, model, M_sequence):
    """
    Compute Δg_{x, l}(y).
    """
    M_l = M_sequence[l]
    M_l_minus_1 = M_sequence[l-1] if l > 0 else 0
    gamma_samples_l = model.sample_posterior_distribution(rng, M_l)
    gamma_samples_l_minus_1_a = {k: v[:M_l_minus_1] for k, v in gamma_samples_l.items()}
    gamma_samples_l_minus_1_b = {k: v[M_l_minus_1:] for k, v in gamma_samples_l.items()}
    
    g_l = g_x_M(y, x, gamma_samples_l)
    g_l_minus_1_a = g_x_M(y, x, gamma_samples_l_minus_1_a) if l > 0 else 0
    g_l_minus_1_b = g_x_M(y, x, gamma_samples_l_minus_1_b) if l > 0 else 0
    return g_l - (g_l_minus_1_a + g_l_minus_1_b) / 2


def mlmc_gradient_estimator(rng, y, x, R, model, M0=10, tau=1.1):
    """
    Estimate the gradient using MLMC.
    """
    M_sequence = [M0 * 2**l for l in range(17)]
    omega = jnp.array([2**(-tau * l) for l in range(len(M_sequence))])
    omega /= omega.sum()
    l_indices = jax.random.choice(jax.random.PRNGKey(0), len(M_sequence), shape=(R,), p=omega)
    estimates = jnp.array([delta_g_x_l(rng, y, x, l, model, M_sequence) / omega[l] for l in l_indices])
    return estimates.mean(axis=0)


def mlmc_attack(model, x, appd=None, lr=0.01, n_iter=1000, epsilon=0.1, R=100, early_stopping_patience=10, verbose=True, optimizer="Adam"):
    """
    Perform the attack using the MLMC gradient estimator.
    :param appd: Attacker predictive posterior distribution to approximate. If None -> Maximum disruption attack.
    """
    # Add noise to the initial input
    rng = jax.random.PRNGKey(int(time() * 1e9) % 2**32)
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
        if appd is None:
            y = model.sample_predictive_distribution(x, num_samples=1)
            rng, mlmc_rng = jax.random.split(rng)
            grad = mlmc_gradient_estimator(mlmc_rng, y, x_adv, R, model)
        else:
            rng, sample_rng = jax.random.split(rng)
            y = appd.sample(sample_rng)
            rng, mlmc_rng = jax.random.split(rng)
            grad = -mlmc_gradient_estimator(mlmc_rng, y, x_adv, R, model)
            
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
    lr = epsilon
    n_iter = 1
    rng = jax.random.PRNGKey(int(time() * 1e9) % 2**32)
    rng, noise_rng = jax.random.split(rng)
    x_adv = x + jax.random.normal(noise_rng, shape=x.shape) * 0.00001

    for it in range(n_iter):
        rng, sample_rng = jax.random.split(rng)
        if appd is None:
            y = model.sample_predictive_distribution(sample_rng, x, num_samples=1)
        else:
            y = appd.sample(sample_rng)
        rng, mlmc_rng = jax.random.split(rng)
        grad = mlmc_gradient_estimator(mlmc_rng, y, x_adv, R, model)
        if appd is None:
            grad_sign = jnp.sign(grad)
        else:
            grad_sign = -jnp.sign(grad)

        x_adv += lr * grad_sign
        x_adv = x + epsilon * (x_adv - x) / jnp.linalg.norm(x_adv - x, ord=2)

    return x_adv, None
