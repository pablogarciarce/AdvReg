import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from src.models.conjugate_probabilistic_model import ConjugateProbabilisticModel


class FlatPriorLinearRegression(ConjugateProbabilisticModel):
    """
    The flat non-informative Jeffreys prior for a linear regression model with known
    variance (assumed to be 1) and unknown coefficients. 
    The model is given by y = Xβ + ε, with ε ~ N(0, 1).
    """

    def __init__(self, prior_params: dict = None) -> None:
        """
        Initialize the model.

        Parameters:
        - prior_params: dictionary of prior parameters which keys 
        are string (name of the parameters) and values are torch tensors
        """
        if prior_params is None:
            prior_params = {}
        self.mu = prior_params.get('mu', None)

    def fit(self, data: dict) -> None:
        """
        Compute posterior parameters.

        Parameters:
        - data: dictionary containing 'X' (features) and 'y' (targets)
        """
        X = data['X']
        y = data['y']

        v_ast = torch.tensor(np.linalg.inv(X.T @ X), dtype=torch.float32)
        mu_ast = torch.tensor(v_ast @ X.T @ y, dtype=torch.float32)

        self.mu = mu_ast
        self.v = v_ast

    def get_predictive_distribution(self, X_test: torch.Tensor):
        """
        Predict using the fitted model.

        Parameters:
        - X_test: torch.Tensor
            Test data features.

        Returns:
        torch.distributions.MultivariateNormal
            Multivariate normal distribution representing the predictive distribution.
        """
        mean = X_test @ self.mu
        cov = torch.diag(torch.ones(X_test.shape[0])) + X_test @ self.v @ X_test.T
        return MultivariateNormal(mean, cov)

    def sample_predictive_distribution(self, X_test: torch.Tensor, num_samples: int):
        """
        Sample from the predictive distribution.

        Parameters:
        - X_test: torch.Tensor
            Test data features.
        - num_samples: int
            Number of samples to draw.

        Returns:
        torch.Tensor
            Samples from the predictive distribution.
        """
        predictive_dist = self.get_predictive_distribution(X_test)
        return predictive_dist.sample((num_samples,))


if __name__ == "__main__":
    # Generate some synthetic data
    n = 100
    p = 2
    X = np.random.randn(n, p)
    beta = np.array([1.0, 2.0])
    sigma = 1.0
    y = X @ beta + np.random.randn(n) * sigma

    # Fit the model
    model = FlatPriorLinearRegression()
    data = {'X': torch.tensor(X, dtype=torch.float32), 'y': torch.tensor(y, dtype=torch.float32)}
    model.fit(data)

    # Predict
    X_test = np.random.randn(10, p)
    y_test = X_test @ beta
    predictive_dist = model.get_predictive_distribution(torch.tensor(X_test, dtype=torch.float32))
    samples = predictive_dist.sample((100,))

    # Get the mean and variances of the predictive distribution
    mean = predictive_dist.mean
    variance = predictive_dist.variance 

    # Plot test data, true regression line, predictive mean and 95% probability interval
    plt.scatter(X_test[:, 0], y_test, label='True regression line')
    plt.scatter(X_test[:, 0], mean, label='Predictive mean')
    plt.fill_between(X_test[:, 0], mean - 1.96 * variance, 
        mean + 1.96 * variance, alpha=0.2, label='95% probability interval')
    plt.legend()
    plt.show()


 