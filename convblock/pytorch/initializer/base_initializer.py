"""Contains implementation of base initializer."""

from __future__ import annotations

from abc import abstractmethod

from .initializer_config import InitializerConfigDict
from .initializer_interface import IInitializer


class BaseInitializer(IInitializer):
    """Base initializer class implementation."""

    @classmethod
    @abstractmethod
    def from_config(cls, config_dict: InitializerConfigDict) -> IInitializer:
        """Construct initializer given configuration dictionary.

        Parameters
        ----------
        config_dict : InitializerConfigDict
            initializer configuration dictionary.
        """
        raise NotImplementedError()
