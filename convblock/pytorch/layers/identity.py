"""Contains implementation of identity module."""

from __future__ import annotations

import numpy as np
import torch

from ..utils import ArrayLike
from .torch_module import TorchModule


class Identity(TorchModule):
    """Identity layer implementation."""

    def __init__(self, input_shape: ArrayLike[int]) -> None:
        input_shape = np.array(input_shape, dtype=np.int64)
        super().__init__(
            input_shape=input_shape[np.newaxis, ...], output_shape=input_shape[np.newaxis, ...]
        )

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method for identity layer.

        Parameters
        ----------
        inputs : torch.Tensor
            input tensor.

        Returns
        -------
        torch.Tensor
            same as input.
        """
        return inputs
