import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import torch

class BayesianNN:
    def __init__(self, input_dim, hidden_units=10):
        """
        Initialize the BNN class with the input dimension and number of hidden units.
        :param input_dim: Dimensionality of the input features.
        :param hidden_units: Number of hidden units in the hidden layer.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.mcmc = None
        self.posterior_samples = None

    def model(self, X, Y=None):
        """
        Defines the probabilistic model for the shallow Bayesian Neural Network.
        :param X: Input data
        :param Y: Target data (optional for sampling predictions)
        """
        # Priors for the weights and biases of the first layer
        w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((self.input_dim, self.hidden_units)),
                                              jnp.ones((self.input_dim, self.hidden_units))))
        b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(self.hidden_units), jnp.ones(self.hidden_units)))

        # Priors for the weights and biases of the second (output) layer
        w2 = numpyro.sample("w2", dist.Normal(jnp.zeros(self.hidden_units), jnp.ones(self.hidden_units)))
        b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))

        # Prior for sigma2 (inverse of precision)
        sigma2 = numpyro.sample("sigma2", dist.Gamma(2.0, 2.0))

        # Forward pass
        hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
        output = jnp.dot(hidden, w2) + b2

        # Likelihood for regression (Gaussian likelihood)
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Normal(output, jnp.sqrt(sigma2)), obs=Y)

    def fit(self, X, Y, num_warmup=500, num_samples=1000, num_chains=2):
        """
        Fits the BNN model to the data using MCMC sampling.
        :param X: Input training data.
        :param Y: Target training data.
        :param num_warmup: Number of warmup steps for MCMC.
        :param num_samples: Number of samples to collect during MCMC.
        """
        # Initialize NUTS sampler
        nuts_kernel = NUTS(self.model)
        self.mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        
        # Run MCMC to sample from the posterior
        self.mcmc.run(jax.random.PRNGKey(0), X, Y)
        self.posterior_samples = self.mcmc.get_samples()

    def sample_posterior_distribution(self, num_samples):
        """
        Returns the posterior samples collected during MCMC.
        :param num_samples: Number of samples to draw from the posterior.
        """
        if self.posterior_samples is None:
            raise RuntimeError("You must call 'fit' before sampling from the posterior.")
        return {k: v[:num_samples] for k, v in self.posterior_samples.items()}
    
    def get_predictive_distribution(self, X_test, num_samples=None):
        """
        Computes the predictive distribution for given test inputs.
        :param X_test: Input test data.
        :param num_samples: Number of samples to draw from the predictive distribution.
        :return: Predictive distribution.
        """
        if self.posterior_samples is None:
            raise RuntimeError("You must call 'fit' before computing the predictive distribution.")
        w1 = torch.tensor(jax.device_get(self.posterior_samples["w1"]), dtype=torch.float32)
        b1 = torch.tensor(jax.device_get(self.posterior_samples["b1"]), dtype=torch.float32)
        w2 = torch.tensor(jax.device_get(self.posterior_samples["w2"]), dtype=torch.float32)
        b2 = torch.tensor(jax.device_get(self.posterior_samples["b2"]), dtype=torch.float32)
        sigma2 = torch.tensor(jax.device_get(self.posterior_samples["sigma2"]), dtype=torch.float32)[:num_samples]
        
        hidden = torch.matmul(w1.transpose(1, 2), X_test).squeeze(2) + b1
        hidden = torch.relu(hidden)
        mean = torch.matmul(w2, hidden.T).diagonal() + b2
        mean = mean[:num_samples] if num_samples is not None else mean.mean()
        return torch.distributions.Normal(mean, torch.sqrt(sigma2))
    
    def get_predictive_distribution_jax(self, X_test, num_samples=None):
        """
        Computes the predictive distribution for given test inputs.
        :param X_test: Input test data.
        :param num_samples: Number of samples to draw from the predictive distribution.
        :return: Predictive distribution.
        """
        # If X_test is a tensor: 
        X_test = X_test.reshape(self.input_dim)
        if not isinstance(X_test, jnp.ndarray):
            if isinstance(X_test, torch.Tensor):
                X_test = X_test.cpu()
            X_test = jnp.array(X_test)
        if self.posterior_samples is None:
            raise RuntimeError("You must call 'fit' before sampling from the predictive distribution.")
        
        # Create predictive function
        predictive = Predictive(self.model, self.posterior_samples)
        if num_samples is None: 
            mean = predictive(jax.random.PRNGKey(1), X_test)['obs']
            mean = torch.tensor(jax.device_get(mean), dtype=torch.float32)
            mean = mean.mean(axis=0)
        else:
            mean = predictive(jax.random.PRNGKey(1), X_test)['obs'][:num_samples, :]
            mean = torch.tensor(jax.device_get(mean), dtype=torch.float32)
        return torch.distributions.Normal(mean, self.sigma2)

    def sample_predictive_distribution(self, X_test, num_samples):
        """
        Samples from the predictive distribution for given test inputs.
        :param X_test: Input test data.
        :param num_samples: Number of samples to draw from the predictive distribution.
        :return: Predictive distribution samples.
        """
        predictive_dist = self.get_predictive_distribution(X_test, num_samples)
        return predictive_dist.sample()
    
    def save(self, path):
        """
        Saves the model to a file.
        :param path: Path to save the model.
        """
        torch.save(self.posterior_samples, path)

    def load(self, path):
        """
        Loads the model from a file.
        :param path: Path to load the model from.
        """
        self.posterior_samples = torch.load(path)
    