"""Contains implementation of zeros initializer class."""

from __future__ import annotations

import torch

from .base_initializer import BaseInitializer
from .initializer_config import InitializerConfigDict, ZerosInitializerConfig
from .initializer_factory import InitializerFactory
from .initializer_interface import IInitializer


@InitializerFactory.register_initializer("zeros", ZerosInitializerConfig)
class ZerosInitializer(BaseInitializer):
    """Zeros initializer implementation.

    Zeros initializer initialize input tensor
    with zeros.
    """

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
        return torch.nn.init.zeros_(x)

    @classmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Create zeros initializer from configuration dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            initializer configuration dictionary.

        Returns
        -------
        IInitializer
            constructed initializer instance.
        """
        _ = ZerosInitializerConfig.model_validate(config_dict)
        return ZerosInitializer()
