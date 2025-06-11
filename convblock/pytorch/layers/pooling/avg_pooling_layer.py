"""Contains implementation of average pooling layer."""

from __future__ import annotations

from typing import ClassVar

import torch

from .base_pooling_layer import BasePoolLayer


class AvgPool(BasePoolLayer):
    """Average pooling layer implementation."""

    _pool_mode: ClassVar[str] = "avg"

    def forward_pool(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method.

        Parameters
        ----------
        inputs : Tensor

        Returns
        -------
        Tensor
            result of pooling operation.
        """
        ndims = len(self.get_input_shape(0)) - 1
        if ndims == 1:
            return torch.nn.functional.avg_pool1d(
                input=inputs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )
        elif ndims == 2:
            return torch.nn.functional.avg_pool2d(
                input=inputs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )
        elif ndims == 3:
            return torch.nn.functional.avg_pool3d(
                input=inputs,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )
        return inputs
