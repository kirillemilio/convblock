"""Contains implementation of constant initializer class."""

from __future__ import annotations

import torch

from .base_initializer import BaseInitializer
from .initializer_config import ConstantInitializerConfig, InitializerConfigDict
from .initializer_interface import IInitializer


class ConstantInitializer(BaseInitializer):
    """Constant initializer implementation.

    Constant initializer initializes input tensor
    with given constant value.

    Attributes
    ----------
    val : float
        value that will be used for initialization of tensors.
    """

    val: float

    def __init__(self, val: float) -> None:
        """Initialize constant initializer with given value.

        Parameters
        ----------
        val : float
            value that will be used for tensor initialization.
        """
        self.val = val

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
        return torch.nn.init.constant_(x, val=self.val)

    @classmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Create constant initializer from configuration dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            initializer configuration dictionary.

        Returns
        -------
        IInitializer
            constructed initializer instance.
        """
        config = ConstantInitializerConfig.model_validate(config_dict)
        return ConstantInitializer(val=config.val)
