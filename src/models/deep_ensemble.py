import jax
import jax.numpy as jnp
from jax import random, vmap
import flax
from flax import nnx
import optax
from orbax import checkpoint
from functools import partial
import numpyro


class CNN(nnx.Module):
  """A simple CNN model."""

  def __init__(self, rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x


class MLP(nnx.Module):
    """A simple MLP model."""

    def __init__(self, input_dim, hidden_dim, output_dim, rng):
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rng)
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rng)
        self.linear3 = nnx.Linear(hidden_dim, 1, rngs=rng)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        var = self.linear3(x)
        var = nnx.softplus(var) + 1e-6
        x = self.linear2(x)
        return x, var


class MnistMLP(nnx.Module):
    """A simple MLP model."""

    def __init__(self, input_dim, hidden_dim, output_dim, rng):
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rng)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rng)
        self.linear3 = nnx.Linear(hidden_dim, output_dim, rngs=rng)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    

class DeepEnsemble(nnx.Module):
    def __init__(self, input_dim, hidden_units=10, output_dim=1, num_models=5, model_type="mlp"):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.num_models = num_models
        self.model_type = model_type
        self.models = [
            self.create_model(input_dim, hidden_units, output_dim, nnx.Rngs(i))  
            for i in range(num_models)
        ]

    def create_model(self, din, hidden, dout, rng):
        if self.model_type == "mlp":
            model = MLP(din, hidden, dout, rng)
        elif self.model_type == "cnn":
            model = CNN(rng=rng)
        elif self.model_type == "mnist_mlp":
            model = MnistMLP(din, hidden, dout, rng)
        else:
            raise ValueError("Invalid model type. Must be 'mlp', 'mnist_mlp' or 'cnn'.")
        return model

    def fit(self, X, Y, num_epochs=1000, lr=1e-3, batch_size=32):
        """
        Fits the ensemble of models using mini-batch Adam with random batches.
        """
        for i, model in enumerate(self.models):
            print("Training model...", i + 1)
            optimizer = nnx.Optimizer(model, optax.adam(lr))
            for _ in range(num_epochs):
                idx = jax.random.permutation(jax.random.PRNGKey(0), len(X))
                for batch_idx in range(0, len(X), batch_size):
                    batch = idx[batch_idx:batch_idx + batch_size]
                    X_train, y_train = X[batch], Y[batch]
                    optimizer, loss = self.train_step(model, optimizer, X_train, y_train)
                if _ % 100 == 0:
                    print(f"Loss: {loss:.4f}")
                    

    def train_step(self, model, optimizer, X_train, y_train):
        """
        Performs a single training step for a model.
        """
        if self.model_type == "cnn" or self.model_type == "mnist_mlp":
            def loss_fn(model, X, y):
                y_pred = model(X)
                # Softmax cross-entropy loss    
                loss = optax.softmax_cross_entropy(logits=y_pred, labels=y)
                return jnp.mean(loss)
        else:   
            def loss_fn(model, X, y):
                y_pred, var = model(X)
                # deep ensemble loss: MLE with Gaussian likelihood and heteroscedastic noise
                loss = jnp.log(var) + (y - y_pred) ** 2 / var
                return jnp.mean(loss)

        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grad = grad_fn(model, X_train, y_train)
        optimizer.update(grad)
        return optimizer, loss

    def sample_predictive_distribution_logits(self, X, num_samples=10):
        """
        Computes the ensemble prediction for given inputs.
        """
        predictions = [model(X)[0] for model in self.models]
        indexes = jax.random.randint(jax.random.PRNGKey(0), (num_samples,), 0, len(predictions))
        logits = jnp.array([predictions[i] for i in indexes])
        return logits

    def sample_predictive_distribution(self, rng, X, num_samples=10):
        """
        Computes the ensemble prediction for given inputs.
        """
        predictions = [model(X)[0] for model in self.models]
        indexes = jax.random.randint(rng, (num_samples,), 0, len(predictions))
        logits = jnp.array([predictions[i] for i in indexes])
        probs = jax.nn.softmax(logits, axis=1)
        return probs
    
    def save(self, path):
        """
        Saves the model to disk.
        """
        for i, model in enumerate(self.models):
            _, state = nnx.split(model)

            checkpointer = checkpoint.StandardCheckpointer()
            checkpointer.save(path + '/' + str(i), state)

    def load(self, path):
        """
        Loads the model from disk.
        """
        for i, model in enumerate(self.models):
            model_graph, state = nnx.split(model)

            checkpointer = checkpoint.StandardCheckpointer()
            state = checkpointer.restore(path + '/' + str(i), state)
            model = nnx.merge(model_graph, state)
            self.models[i] = model