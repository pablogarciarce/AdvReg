from abc import ABC, abstractmethod
import torch

class ConjugateProbabilisticModel(ABC):
    """
    Abstract class for probabilistic models
    """
    
    @abstractmethod
    def __init__(self, prior_params: dict) -> None:
        """
        Initialize the probabilistic model with prior parameters.

        Parameter:
        - prior_params: dictionary of prior parameters which keys 
        are string (name of the parameters) and values are torch tensors
        """
        pass

    @abstractmethod
    def fit(self, data: dict) -> None:
        """
        Compute posterior parameters.

        Parameters:
        - data: dictionary containing 'X' (features) and 'y' (targets)
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def sample_posterior_distribution(self, num_samples: int):
        """
        Sample from the posterior distribution.

        Parameters:
        - num_samples: int
            Number of samples to draw.

        Returns:
        torch.Tensor
            Samples from the posterior distribution.
        """
        pass