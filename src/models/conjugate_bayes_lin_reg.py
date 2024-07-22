import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.studentT import StudentT
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
        var = 1 + torch.sum((X_test @ self.v) * X_test, dim=1)
        cov = torch.diag(var)
        if len(mean.shape) != 1:
            mean = mean.squeeze()
        
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


class NormalInverseGammaPriorLinearRegression(ConjugateProbabilisticModel):
    """
    The model is given by y = Xβ + ε, with ε ~ N(0, σ^2) with priors 
    β ~ N(μ, σ^2 Lambda^(-1)) and σ^2 ~ IG(a, b).
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
        self.lam = prior_params.get('lam', None)
        self.a = prior_params.get('a', None)
        self.b = prior_params.get('b', None)

    def fit(self, data: dict) -> None:
        """
        Compute posterior parameters.

        Parameters:
        - data: dictionary containing 'X' (features) and 'y' (targets)
        """
        X = data['X']
        y = data['y']

        n = X.shape[0]

        lam_n = self.lam + X.T @ X
        mu_n = (self.lam @ self.mu + X.T @ y)
        mu_n = torch.linalg.inv(lam_n) @ mu_n
        a_n = self.a + n / 2
        b_n = self.b + 0.5 * (y.T @ y + self.mu.T @ self.lam @ self.mu - mu_n.T @ lam_n @ mu_n)

        self.mu = mu_n
        self.lam = lam_n
        self.a = a_n
        self.b = b_n

    def get_predictive_distribution(self, X_test: torch.Tensor):
        """
        Predict using the fitted model.

        Parameters:
        - X_test: torch.Tensor
            Test data features.

        Returns:
        torch.distributions.StudentT
            Student-t distribution representing the predictive distribution.
        """
        mean = X_test.T @ self.mu
        var = self.b / self.a * (1 + (X_test.T @ torch.linalg.inv(self.lam)) @ X_test)
        if len(mean.shape) != 1:
            mean = mean.squeeze()
        
        return StudentT(2 * self.a, mean, var)

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


class NormalKnownVariancePriorLinearRegression(ConjugateProbabilisticModel):
    """
    The model is given by y = Xβ + ε, with ε ~ N(0, σ^2) with prior
    β ~ N(μ, Lambda^(-1)) and σ^2 known.
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
        self.lam = prior_params.get('lam', None)
        self.sigma2 = prior_params.get('sigma2', None)

    def fit(self, data: dict) -> None:
        """
        Compute posterior parameters.

        Parameters:
        - data: dictionary containing 'X' (features) and 'y' (targets)
        """
        X = data['X']
        y = data['y']

        lam_n = self.lam + X.T @ X / self.sigma2
        mu_n = (self.lam @ self.mu + X.T @ y / self.sigma2)
        mu_n = torch.linalg.inv(lam_n) @ mu_n

        self.mu = mu_n
        self.lam = lam_n

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
        mean = X_test.T @ self.mu
        var = self.sigma2 + ((X_test.T @ torch.linalg.inv(self.lam)) @ X_test)
        if len(mean.shape) != 1:
            mean = mean.squeeze()
        
        return MultivariateNormal(mean, var)

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
    pass