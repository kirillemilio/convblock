"""Contains implementation of initializer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class IInitializer(ABC):
    """Initializer interface implementation."""

    @abstractmethod
    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize values in input tensor.

        Parameters
        ----------
        x : torch.Tensor
            input tensor to be initialized.

        Returns
        -------
        torch.Tensor
            tensor after initialization.
        """
        raise NotImplementedError()
