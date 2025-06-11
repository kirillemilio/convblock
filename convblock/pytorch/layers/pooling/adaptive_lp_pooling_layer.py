"""Contains implementation of adaptive lp pooling layer."""

from __future__ import annotations

import torch

from ...utils import ArrayLike, transform_to_int_tuple
from ..torch_module import TorchModule


class AdaptiveLPPool(TorchModule):
    """Adaptive lp pooling module.

    Attributes
    ----------
    output_size : Tuple[int, ...]
        output_siz from adaptive lp pooling layer.
    norm_type : float
        normalization type value for lp-pooling.
    """

    output_size: tuple[int, ...]
    norm_type: float

    def __init__(
        self, input_shape: ArrayLike[int], output_size: ArrayLike[int], norm_type: float = 1.0
    ):
        """Adaptive lp pooling layer generalized for different dimensions."""
        ndims = len(input_shape) - 1
        output_size = transform_to_int_tuple(output_size, "output_size", ndims - 1)
        super().__init__(input_shape=input_shape, output_shape=[input_shape[0], *output_size])
        self.output_size = output_size
        self.norm_type = norm_type

    def __repr__(self) -> str:
        """Return detailed string representation of the module."""
        return (
            f"{self.__class__.__name__}("
            f"input_shape={self.input_shape}, "
            f"output_shape={self.output_shape}, "
            f"output_size={self.output_size}, "
            f"norm_type={self.norm_type}"
            f")"
        )

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method.

        Parameters
        ----------
        inputs : torch.Tensor
            input tensor for adaptive average pooling layer.

        Returns
        -------
        torch.Tensor
            result of pooling operation.
        """
        ndims = len(self.get_input_shape(0)) - 1
        if ndims == 1:
            return torch.nn.functional.lp_pool1d(
                input=inputs,
                norm_type=self.norm_type,
                kernel_size=self.output_size,
                stride=self.output_size,
            )
        elif ndims == 2:
            return torch.nn.functional.lp_pool2d(
                input=inputs,
                norm_type=self.norm_type,
                kernel_size=self.output_size,
                stride=self.output_size,
            )
        elif ndims == 3:
            return torch.nn.functional.lp_pool3d(
                input=inputs,
                norm_type=self.norm_type,
                kernel_size=self.output_size,
                stride=self.output_size,
            )
        return inputs
