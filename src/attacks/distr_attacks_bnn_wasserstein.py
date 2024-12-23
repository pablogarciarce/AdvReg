from functools import partial
import jax
import optax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from optax import adam
import numpy as np

class InnerNN(nn.Module):
    """Neural network for Wasserstein distance approximation."""
    @nn.compact  
    def __call__(self, x):
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

def wasserstein_distance(inner_nn_params, inner_nn_apply, y, y_adv):
    """Compute the Wasserstein distance."""
    inner_nn_y = inner_nn_apply(inner_nn_params, y)
    inner_nn_y_adv = inner_nn_apply(inner_nn_params, y_adv)
    return jnp.mean(inner_nn_y - inner_nn_y_adv)

#@partial(jax.jit, static_argnames=['inner_optimizer', 'inner_nn_apply'])
def inner_step(inner_nn_params, opt_state, y, y_adv, inner_optimizer, inner_nn_apply):
    """Perform one optimization step for the inner neural network."""
    def inner_loss(params):
        return wasserstein_distance(params, inner_nn_apply, y, y_adv)

    loss, grads = jax.value_and_grad(inner_loss)(inner_nn_params)
    updates, opt_state = inner_optimizer.update(grads, opt_state)
    inner_nn_params = optax.apply_updates(inner_nn_params, updates)
    inner_nn_params = jax.tree_map(lambda x: jnp.clip(x, -0.1, 0.1), inner_nn_params)
    return inner_nn_params, opt_state, loss

#@partial(jax.jit, static_argnames=['model', 'num_samples', 'outer_optimizer', 'inner_nn_apply'])
def outer_step(x_adv, opt_state, inner_nn_params, y, model, num_samples, outer_optimizer, inner_nn_apply):
    """Perform one optimization step for x_adv."""
    def outer_loss(x_adv):
        y_adv = model.sample_predictive_distribution(x_adv, num_samples=num_samples)
        return wasserstein_distance(inner_nn_params, inner_nn_apply, y, y_adv)

    loss, grads = jax.value_and_grad(outer_loss)(x_adv)
    updates, opt_state = outer_optimizer.update(grads, opt_state)
    x_adv = optax.apply_updates(x_adv, updates)
    return x_adv, opt_state, loss

def wasserstein_attack(model, x, appd, rng, epsilon=0.2, num_samples=100, inner_lr=0.01, inner_iter=100,
                       outer_lr=0.01, patience_eps=1e-3, max_iter=1000, verbose=False):
    """Perform the Wasserstein attack using JAX."""
    x_adv = x + random.normal(rng, x.shape) * 0.0001
    x_adv_values = []

    # Initialize the inner neural network
    rng, init_rng = random.split(rng)
    inner_nn = InnerNN()
    inner_nn_params = inner_nn.init(init_rng, appd.sample(init_rng, (1,)))
    inner_nn_apply = inner_nn.apply

    # Initialize optimizers
    inner_optimizer = adam(inner_lr)
    outer_optimizer = adam(outer_lr)
    inner_opt_state = inner_optimizer.init(inner_nn_params)
    outer_opt_state = outer_optimizer.init(x_adv)

    dif, ite = jnp.inf, 0

    while dif > patience_eps and ite < max_iter:
        # Inner optimization
        loss, inner_ite = jnp.inf, 0
        while loss > patience_eps and inner_ite < inner_iter:
            rng, sample_rng = random.split(rng)
            y = model.sample_predictive_distribution(x_adv, num_samples=num_samples)
            y_adv = appd.sample(sample_rng, (num_samples, 1))
            inner_nn_params, inner_opt_state, loss = inner_step(
                inner_nn_params, inner_opt_state, y, y_adv, inner_optimizer, inner_nn_apply
            )
            inner_ite += 1

        # Outer optimization
        rng, sample_rng = random.split(rng)
        y = model.sample_predictive_distribution(x_adv, num_samples=num_samples)
        y_adv = appd.sample(sample_rng, (num_samples, 1))
        x_adv_old = x_adv
        x_adv, outer_opt_state, _ = outer_step(
            x_adv, outer_opt_state, inner_nn_params, y, model, num_samples, outer_optimizer, inner_nn_apply
        )

        # Project x_adv to L2 epsilon-ball of x
        diff = x_adv - x
        norm = jnp.linalg.norm(diff, ord=2)
        x_adv = x + jnp.where(norm > epsilon, diff * (epsilon / norm), diff)

        x_adv_values.append(np.array(x_adv))
        dif = jnp.linalg.norm(x_adv - x_adv_old, ord=2)
        ite += 1

    if verbose:
        print(f"Converged in {ite} iterations.")
    return x_adv, x_adv_values
