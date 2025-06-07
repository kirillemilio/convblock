"""Contains implementation of max pooling layer."""

from __future__ import annotations

from typing import ClassVar

import torch

from .base_pooling_layer import BasePoolLayer


class MaxPool(BasePoolLayer):
    """Max pooling layer for arbitrary dimension.

    Attributes
    ----------
    _pool_mode: ClassVar[str]
        pooling mode. "max" for max pooling.
    """

    _pool_mode: ClassVar[str] = "max"

    def forward_pool(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method.

        Parameters
        ----------
        inputs : torch.Tensor

        Returns
        -------
        torch.Tensor
            result of pooling operation.
        """
        if self.ndims == 2:
            return torch.nn.functional.max_pool1d(
                input=inputs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )
        elif self.ndims == 3:
            return torch.nn.functional.max_pool2d(
                input=inputs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )
        elif self.ndims == 4:
            return torch.nn.functional.max_pool3d(
                input=inputs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )

        return inputs
