"""Contains implementation of normal initializer class."""

from __future__ import annotations

import torch

from .base_initializer import BaseInitializer
from .initializer_config import InitializerConfigDict, NormalInitializerConfig
from .initializer_factory import InitializerFactory
from .initializer_interface import IInitializer


@InitializerFactory.register_initializer("normal", config_cls=NormalInitializerConfig)
class NormalInitializer(BaseInitializer):
    """Normal initializer implementation.

    Normal initializer initializes input tensor
    with normal distribution values.

    Attributes
    ----------
    mean : float
        mean value for normal distribution that will
        be used for tensor initialization.
    std : float
        standard deviation for normal distribution
        that will be used for tensor initialization.
    """

    mean: float
    std: float

    def __init__(self, mean: float, std: float) -> None:
        """Initialize constant initializer with given value.

        Parameters
        ----------
        mean : float
            mean value for normal distribution that will
            be used for tensor initialization.
        std : float
            standard deviation for normal distribution
            that will be used for tensor initialization.
        """
        self.mean = mean
        self.std = std

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize tensor with given initializer.

        Parameters
        ----------
        x : torch.Tensor
            input torch tensor that will
            be initialize with given initializer.

        Returns
        -------
        torch.Tensor
            torch tensor after initialization.
        """
        return torch.nn.init.normal_(x, mean=self.mean, std=self.std)

    @classmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Create normal initializer from configuration dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            initializer configuration dictionary.

        Returns
        -------
        IInitializer
            constructed initializer instance.
        """
        config = NormalInitializerConfig.model_validate(config_dict)
        return NormalInitializer(mean=config.mean, std=config.std)
