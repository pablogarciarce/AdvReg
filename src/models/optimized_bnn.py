import jax
import jax.numpy as jnp
from jax import random, vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, autoguide
import pickle

class BayesianNN:
    def __init__(self, input_dim, hidden_units=10):
        """
        Initialize the BNN class with the input dimension and number of hidden units.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.mcmc = None
        self.posterior_samples = None

    def model(self, X, Y=None):
        """
        Defines the probabilistic model for the Bayesian Neural Network.
        """
        # Priors for weights and biases
        w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([self.input_dim, self.hidden_units]))
        b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([self.hidden_units]))
        w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([self.hidden_units]))
        b2 = numpyro.sample("b2", dist.Normal(0, 1))
        sigma2 = numpyro.sample("sigma2", dist.Gamma(2.0, 2.0))

        # Forward pass
        hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
        output = jnp.dot(hidden, w2) + b2

        # Likelihood for regression
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Normal(output, jnp.sqrt(sigma2)), obs=Y)

    def fit(self, X, Y, num_warmup=500, num_samples=500, num_chains=4):
        """
        Fits the BNN model using MCMC.
        """
        nuts_kernel = NUTS(self.model)
        self.mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        self.mcmc.run(random.PRNGKey(0), X, Y)
        self.posterior_samples = self.mcmc.get_samples()

    def sample_predictive_distribution(self, rng, X_test, num_samples=None):
        """
        Computes the predictive distribution for given test inputs.
        """
        if self.posterior_samples is None:
            raise RuntimeError("You must call 'fit' before making predictions.")

        rng, rng_pred = random.split(rng)
        # Create predictive function
        predictive = Predictive(self.model, self.posterior_samples)
        predictions = predictive(rng_pred, X_test)["obs"]

        # Aggregate predictions
        if num_samples is not None:
            indices = random.choice(rng, len(predictions), (num_samples,))
            predictions = predictions[indices]
        else:
            predictions = predictions.mean(axis=0)
        return predictions
    
    def sample_posterior_distribution(self, rng, num_samples):
        """
        Samples from the posterior distribution.
        """
        if self.posterior_samples is None:
            raise RuntimeError("You must call 'fit' before sampling from the posterior.")

        indices = random.choice(rng, len(self.posterior_samples["w1"]), (num_samples,))
        sampled_posterior = {k: v[indices] for k, v in self.posterior_samples.items()}
        return sampled_posterior


    def save(self, path):
        """
        Saves the posterior samples to a file.
        """
        with open(path, "wb") as f:
            pickle.dump(self.posterior_samples, f)

    def load(self, path):
        """
        Loads posterior samples from a file.
        """
        with open(path, "rb") as f:
            self.posterior_samples = pickle.load(f)

class DBNN(BayesianNN):
    def __init__(self, input_dim, hidden_units=10):
        """
        Initialize the BNN class with the input dimension and number of hidden units.
        """
        super().__init__(input_dim, hidden_units)

    def model(self, X, Y=None):
        """
        Defines the probabilistic model for the Bayesian Neural Network with 3 hidden layers.
        Each hidden layer has 30 neurons.
        """
        # Priors for weights and biases for each layer
        w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([self.input_dim, self.hidden_units]))  # First hidden layer weights
        b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([self.hidden_units]))  # First hidden layer biases

        w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([self.hidden_units, self.hidden_units]))  # Second hidden layer weights
        b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([self.hidden_units]))  # Second hidden layer biases

        w3 = numpyro.sample("w3", dist.Normal(0, 1).expand([self.hidden_units, self.hidden_units]))  # Third hidden layer weights
        b3 = numpyro.sample("b3", dist.Normal(0, 1).expand([self.hidden_units]))  # Third hidden layer biases

        w4 = numpyro.sample("w4", dist.Normal(0, 1).expand([self.hidden_units, 1]))  # Output layer weights
        b4 = numpyro.sample("b4", dist.Normal(0, 1))  # Output layer bias

        sigma2 = numpyro.sample("sigma2", dist.Gamma(2.0, 2.0))  # Prior for variance

        # Forward pass through the hidden layers
        hidden1 = jax.nn.relu(jnp.dot(X, w1) + b1)  # First hidden layer
        hidden2 = jax.nn.relu(jnp.dot(hidden1, w2) + b2)  # Second hidden layer
        hidden3 = jax.nn.relu(jnp.dot(hidden2, w3) + b3)  # Third hidden layer
        output = jnp.dot(hidden3, w4) + b4  # Output layer

        # Likelihood for regression
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Normal(output, jnp.sqrt(sigma2)), obs=Y)

class RegBayesianNNVI:
    def __init__(self, input_dim, hidden_units=10):
        """
        Initialize the BNN class with the input dimension and number of hidden units.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.guide = None
        self.svi_result = None
        self.posterior_samples = None

    def model(self, X, Y=None):
        """
        Defines the probabilistic model for the Bayesian Neural Network.
        """
        # Priors for weights and biases
        w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([self.input_dim, self.hidden_units]))
        b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([self.hidden_units]))
        w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([self.hidden_units]))
        b2 = numpyro.sample("b2", dist.Normal(0, 1))
        sigma2 = numpyro.sample("sigma2", dist.Gamma(2.0, 2.0))

        # Forward pass
        hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
        output = jnp.dot(hidden, w2) + b2

        # Likelihood for regression
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Normal(output, jnp.sqrt(sigma2)), obs=Y)

    def fit(self, X, Y, num_steps=10000, lr=0.01):
        """
        Fits the BNN model using Variational Inference (VI).
        """
        # Define guide for VI
        self.guide = autoguide.AutoMultivariateNormal(self.model) #  AutoBNAFNormal and AutoIAFNormal 

        # Set up the optimizer and loss function
        optimizer = numpyro.optim.Adam(step_size=lr)
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Initialize SVI state
        svi_state = svi.init(random.PRNGKey(0), X, Y)

        # Run optimization
        def update(state, step):
            state, loss = svi.update(state, X, Y)
            return state, loss

        for step in range(num_steps):
            svi_state, loss = update(svi_state, step)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")

        # Store final parameters
        self.svi_result = svi.get_params(svi_state)

        # Sample from the posterior to store the samples
        self.posterior_samples = self._sample_posterior_distribution(random.PRNGKey(0), num_samples=100000)

    def sample_predictive_distribution(self, rng, X_test, num_samples=100):
        """
        Computes the predictive distribution for given test inputs.
        """
        if self.svi_result is None:
            raise RuntimeError("You must call 'fit' before making predictions.")

        # Create predictive function
        predictive = Predictive(self.model, guide=self.guide, params=self.svi_result, num_samples=num_samples)
        predictions = predictive(rng, X_test)["obs"]

        # Aggregate predictions
        return predictions
    
    def _sample_posterior_distribution(self, rng, num_samples=100):
        """
        Samples from the posterior distribution.
        """
        if self.svi_result is None:
            raise RuntimeError("You must call 'fit' before sampling from the posterior.")

        # Use the guide to sample posterior distributions
        predictive = Predictive(self.guide, params=self.svi_result, num_samples=num_samples)
        posterior_samples = predictive(rng, data=None)
        return posterior_samples
    
    def sample_posterior_distribution(self, rng, num_samples):
        """
        Samples from the posterior distribution.
        """
        if self.posterior_samples is None:
            raise RuntimeError("You must call 'fit' before sampling from the posterior.")

        indices = random.choice(rng, len(self.posterior_samples["w1"]), (num_samples,))
        sampled_posterior = {k: v[indices] for k, v in self.posterior_samples.items()}
        return sampled_posterior
    
    def save(self, path):
        """
        Saves the variational parameters to a file.
        """
        with open(path+'svi', "wb") as f:
            pickle.dump(self.svi_result, f)
        with open(path+'guide', "wb") as f:
            pickle.dump(self.guide, f)
        with open(path+'posterior', "wb") as f:
            pickle.dump(self.posterior_samples, f)

    def load(self, path):
        """
        Loads variational parameters from a file.
        """
        with open(path+'svi', "rb") as f:
            self.svi_result = pickle.load(f)
        with open(path+'guide', "rb") as f:
            self.guide = pickle.load(f)
        with open(path+'posterior', "rb") as f:
            self.posterior_samples = pickle.load(f)


class ClasBayesianNNVI(RegBayesianNNVI):
    def __init__(self, input_dim, hidden_units=10, num_classes=10):
        """
        Initialize the BNN class with the input dimension and number of hidden units.
        """
        super().__init__(input_dim, hidden_units)
        self.num_classes = num_classes
        self.train = False

    def model(self, X, Y=None):
        """
        Defines the probabilistic model for the Bayesian Neural Network.
        """
        if Y is not None and Y.ndim == 2:
            Y = jnp.argmax(Y, axis=1) 
        # Priors for weights and biases
        w1 = numpyro.sample("w1", dist.Normal(0, 1).expand([self.input_dim, self.hidden_units]))
        b1 = numpyro.sample("b1", dist.Normal(0, 1).expand([self.hidden_units]))
        w2 = numpyro.sample("w2", dist.Normal(0, 1).expand([self.hidden_units, self.num_classes]))
        b2 = numpyro.sample("b2", dist.Normal(0, 1).expand([self.num_classes]))

        # Forward pass
        hidden = jax.nn.relu(jnp.dot(X, w1) + b1)
        logits = jnp.dot(hidden, w2) + b2
        probs = jax.nn.softmax(logits)
        if self.train:
            probs = jnp.clip(probs, 1e-6, 1)  # Avoid zero probabilities
            probs = probs / probs.sum(axis=-1, keepdims=True) # Normalize

        numpyro.deterministic("probs", probs)

        # Likelihood for classification
        with numpyro.plate("data", X.shape[0]):
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=Y)

    def fit(self, X, Y, num_steps=10000, lr=0.001):
        """
        Fits the BNN model using Variational Inference (VI).
        """
        self.train = True
        # Define guide for VI
        self.guide = autoguide.AutoDiagonalNormal(self.model) #  AutoBNAFNormal and AutoIAFNormal 

        # Set up the optimizer and loss function
        optimizer = numpyro.optim.Adam(step_size=lr)
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Initialize SVI state
        svi_state = svi.init(random.PRNGKey(0), X, Y)

        # Run optimization
        def update(state, step):
            state, loss = svi.update(state, X, Y)
            return state, loss

        for step in range(num_steps):
            svi_state, loss = update(svi_state, step)
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")

        # Store final parameters
        self.svi_result = svi.get_params(svi_state)

        # Sample from the posterior to store the samples
        self.posterior_samples = self._sample_posterior_distribution(random.PRNGKey(0), num_samples=100000)
        self.train = False

    def sample_predictive_distribution_probs(self, rng, X_test, num_samples=100):
        """
        Computes the predictive distribution for given test inputs.
        """
        if self.svi_result is None:
            raise RuntimeError("You must call 'fit' before making predictions.")

        # Create predictive function
        predictive = Predictive(self.model, guide=self.guide, params=self.svi_result, num_samples=num_samples)
        samples = predictive(rng, X_test)

        # Extract probabilities
        probs = samples["probs"]  # Shape: (num_samples, batch_size, num_classes)

        return probs
    