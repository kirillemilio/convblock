"""Contains implementation of Xavier uniform initializer class."""

from __future__ import annotations

import torch

from .base_initializer import BaseInitializer
from .initializer_config import (
    InitializerConfigDict,
    XavierUniformInitializerConfig,
)
from .initializer_factory import InitializerFactory
from .initializer_interface import IInitializer


@InitializerFactory.register_initializer("uxavier", config_cls="uxavier")
class XavierUniformInitializer(BaseInitializer):
    """Xavier uniform initializer implementation.

    Initializes weights from a gain-scaled uniform distribution based on
    fan-in and fan-out values of the tensor.

    Attributes
    ----------
    gain : float
        Gain factor applied during Xavier initialization.
    """

    gain: float

    def __init__(self, gain: float = 1.0) -> None:
        """Initialize Xavier uniform initializer.

        Parameters
        ----------
        gain : float, default=1.0
            Scaling factor applied to the distribution.
        """
        self.gain = gain

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize tensor with Xavier uniform distribution.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to be initialized.

        Returns
        -------
        torch.Tensor
            Initialized tensor.
        """
        return torch.nn.init.xavier_uniform_(x, gain=self.gain)

    @classmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Create Xavier uniform initializer from config dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            Configuration dictionary with 'gain' parameter.

        Returns
        -------
        IInitializer
            Initialized instance of XavierUniformInitializer.
        """
        config = XavierUniformInitializerConfig.model_validate(config_dict)
        return cls(gain=config.gain)
