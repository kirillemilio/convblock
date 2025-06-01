"""Contains implementation of Kaiming uniform initializer class."""

from __future__ import annotations

import torch

from .base_initializer import BaseInitializer
from .initializer_config import (
    InitializerConfigDict,
    KaimingUniformInitializerConfig,
)
from .initializer_factory import InitializerFactory
from .initializer_interface import IInitializer


@InitializerFactory.register_initializer("ukaiming", config_cls=KaimingUniformInitializerConfig)
class KaimingUniformInitializer(BaseInitializer):
    """Kaiming uniform initializer implementation.

    Initializes weights using a uniform distribution scaled by fan-in
    or fan-out and nonlinearity scaling factor.

    Attributes
    ----------
    a : float
        Negative slope of the rectifier used (e.g., LeakyReLU).
    mode : str
        Either 'fan_in' or 'fan_out'.
    nonlinearity : str
        Name of the non-linear function ('relu', 'leaky_relu', etc.).
    """

    a: float
    mode: str
    nonlinearity: str

    def __init__(
        self,
        a: float = 0.0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
    ) -> None:
        """Initialize Kaiming uniform initializer.

        Parameters
        ----------
        a : float, default=0.0
            Negative slope for LeakyReLU.
        mode : str, default='fan_in'
            'fan_in' preserves input variance; 'fan_out' preserves output.
        nonlinearity : str, default='leaky_relu'
            Non-linear function name.
        """
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize tensor with Kaiming uniform distribution.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be initialized.

        Returns
        -------
        torch.Tensor
            Initialized tensor.
        """
        return torch.nn.init.kaiming_uniform_(
            x, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
        )

    @classmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Create Kaiming uniform initializer from configuration dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            Dictionary with parameters: a, mode, nonlinearity.

        Returns
        -------
        IInitializer
            Kaiming uniform initializer instance.
        """
        config = KaimingUniformInitializerConfig.model_validate(config_dict)
        return cls(
            a=config.a,
            mode=config.mode,
            nonlinearity=config.nonlinearity,
        )
