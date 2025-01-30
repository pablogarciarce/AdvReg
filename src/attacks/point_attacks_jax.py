import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.random import normal
from src.utils import id
from src.utils2 import l1_projection, l2_projection
import optax


def reparametrization_trick(x_adv, model, G, samples_per_iteration, func, x_0=None):
    def loss_fn(x):
        y_samples = model.sample_predictive_distribution(x, samples_per_iteration)
        f_values = func(x, y_samples)

        if x_0 is not None:
            f_0 = func(x_0, model.sample_predictive_distribution(x_0, samples_per_iteration))
            return -jnp.mean((f_values - f_0) ** 2)
        else:
            return jnp.mean((f_values - G) ** 2)

    grad_loss = grad(loss_fn)(x_adv)
    y_samples = model.sample_predictive_distribution(x_adv, samples_per_iteration)
    f_values = func(x_adv, y_samples)
    loss = loss_fn(x_adv)
    return grad_loss, jnp.mean(f_values), loss



def attack(x_clean, model, G, samples_per_iteration=100, learning_rate=1e-3,
           num_iterations=1000, epsilon=0.1, func=id, early_stopping_patience=10, projection=l2_projection, verbose=False):
    x_0 = x_clean.copy()
    key = jax.random.PRNGKey(0)
    x_adv = x_clean + 0.002 * normal(key, shape=x_clean.shape)
    x_adv_values, loss_values, func_values = [], [], []
    early_stopping_it = 0

    # Initialize the Adam optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(x_adv)

    for _ in range(num_iterations):
        x_old = x_adv.copy()
        grad_loss, f_mean, loss = reparametrization_trick(x_adv, model, G, samples_per_iteration, func)
        
        # Update the adversarial example using the Adam optimizer
        updates, opt_state = optimizer.update(grad_loss, opt_state)
        x_adv = optax.apply_updates(x_adv, updates)

        if jnp.linalg.norm(x_adv - x_0, ord=2) > epsilon:
            x_adv = projection(x_adv, x_0, epsilon)

        if jnp.linalg.norm(x_adv - x_old) < 1e-5:
            early_stopping_it += 1
            if early_stopping_it > early_stopping_patience:
                if verbose:
                    print("Early stopping")
                break
        else:
            early_stopping_it = 0

        x_adv_values.append(x_adv.copy())
        loss_values.append(float(loss))
        func_values.append(float(f_mean))

    return x_adv_values, loss_values, func_values


@jit
def true_gradient_mean(x_adv, model, G):
    beta_dot_x = jnp.dot(model.mu, x_adv)
    return 2 * (beta_dot_x - G) * model.mu


def attack_true_grad(x_clean, model, G, learning_rate=0.1, num_iterations=1000, epsilon=0.1):
    x_0 = x_clean.copy()
    x_adv = x_clean.copy()
    x_adv_values = []

    for _ in range(num_iterations):
        grad_loss = true_gradient_mean(x_adv, model, G)
        x_adv -= learning_rate * grad_loss

        if jnp.linalg.norm(x_adv - x_0, ord=2) > epsilon:
            x_adv = x_0 + epsilon * (x_adv - x_0) / jnp.linalg.norm(x_adv - x_0, ord=2)

        x_adv_values.append(x_adv.copy())

    return x_adv_values
