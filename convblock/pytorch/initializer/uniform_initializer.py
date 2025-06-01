"""Contains implementation of uniform initializer class."""

from __future__ import annotations

import torch

from .base_initializer import BaseInitializer
from .initializer_config import InitializerConfigDict, UniformInitializerConfig
from .initializer_factory import InitializerFactory
from .initializer_interface import IInitializer


@InitializerFactory.register_initializer("uniform", config_cls=UniformInitializerConfig)
class UniformInitializer(BaseInitializer):
    """Uniforms initializer implementation.

    Uniform initializer initializes input tensor
    with values from uniform distribution.

    Attributes
    ----------
    a : float
        lower boundary of uniform distribution
        that will be used for initialization.
    b : float
        upper boundary of uniform distribution
        that will be used for initialization.
    """

    a: float
    b: float

    def __init__(self, a: float, b: float) -> None:
        """Initialize uniform initializer with given lower/upper bounds params.

        Parameters
        ----------
        a : float
            lower boundary of uniform distribution
            that will be used for initialization.
        b : float
            upper boundary of uniform distribution
            that will be used for initialization.
        """
        self.a = a
        self.b = b

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
        return torch.nn.init.uniform_(x, a=self.a, b=self.b)

    @classmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Create uniform initializer from configuration dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            initializer configuration dictionary.

        Returns
        -------
        IInitializer
            constructed initializer instance.
        """
        config = UniformInitializerConfig.model_validate(config_dict)
        return UniformInitializer(a=config.a, b=config.b)
