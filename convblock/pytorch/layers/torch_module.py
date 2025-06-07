"""Contains implementation of simple pytorch wrapped module."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch
from numpy.typing import NDArray

from .base_module import BaseModule


class TorchModule(BaseModule, torch.nn.Module):
    """Provides implementation of pytorch module wrapper."""

    def __init__(self, input_shape: NDArray[np.int64], output_shape: NDArray[np.int64]) -> None:
        """Initialize simple module with specified input and output shapes.

        Parameters
        ----------
        input_shape : NDArray[np.int64]
            1D or 2D numpy array defining expected shapes of input tensors.
        output_shape : NDArray[np.int64]
            1D or 2D numpy array defining expected shapes of output tensors.

        Raises
        ------
        ValueError
            If eighter input_shape or output_shape has more than 2 dimensions.
        """
        BaseModule.__init__(self, input_shape=input_shape, output_shape=output_shape)
        torch.nn.Module.__init__(self)

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        """Forward torch tensors through module.

        Parameters
        ----------
        *inputs : torch.Tensor
            input tensors.

        Returns
        -------
        list[torch.Tensor]
            list of output tensors.
        """
        raise NotImplementedError()

    def count_parameters(self, include_static: bool = False) -> int:
        """Count number of parameters.

        Parameters
        ----------
        include_static : bool
            whether to include static(non trainable parameters)
            Default is False meaning that only parameters
            requiring grad will be taken into consideration.

        Returns
        -------
        int
            number of parameters
            requiring gradient computation.
        """
        if include_static:
            return sum(p.numel() for p in self.parameters())
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
