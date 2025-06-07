"""Contains implementation of truncated normal initializer class."""

from __future__ import annotations

import torch

from .base_initializer import BaseInitializer
from .initializer_config import (
    InitializerConfigDict,
    TruncatedNormalInitializerConfig,
)
from .initializer_interface import IInitializer


class TruncatedNormalInitializer(BaseInitializer):
    """Truncated normal initializer implementation.

    Initializes input tensor with values drawn from a normal
    distribution, but with values clipped within two standard
    deviations from the mean.

    Attributes
    ----------
    mean : float
        Mean value for the normal distribution used for initialization.
    std : float
        Standard deviation of the normal distribution.
    a : float
        Lower truncation bound.
    b : float
        Upper truncation bound.
    """

    mean: float
    std: float
    a: float
    b: float

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
    ) -> None:
        """Initialize truncated normal initializer.

        Parameters
        ----------
        mean : float, default=0.0
            Mean value for the normal distribution.
        std : float, default=1.0
            Standard deviation of the normal distribution.
        a : float, default=-2.0
            Lower truncation bound.
        b : float, default=2.0
            Upper truncation bound.
        """
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize tensor with truncated normal distribution.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be initialized.

        Returns
        -------
        torch.Tensor
            Initialized tensor.
        """
        return torch.nn.init.trunc_normal_(x, mean=self.mean, std=self.std, a=self.a, b=self.b)

    @classmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Create truncated normal initializer from config dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            Configuration dictionary containing initializer parameters.

        Returns
        -------
        IInitializer
            Instance of the truncated normal initializer.
        """
        config = TruncatedNormalInitializerConfig.model_validate(config_dict)
        return cls(
            mean=config.mean,
            std=config.std,
            a=config.a,
            b=config.b,
        )
