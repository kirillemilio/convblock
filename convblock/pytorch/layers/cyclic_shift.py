"""Contains implementation of cycle shift layer."""

from __future__ import annotations

import torch

from ..utils import ArrayLike, transform_to_int_tuple
from .torch_module import TorchModule


class CyclicShift(TorchModule):
    """Cyclic shift layer used by swin transformer.

    Attributes
    ----------
    displacement : ArrayLike[int]
        displacement along spatial axes.
    """

    displacement: ArrayLike[int]

    def __init__(self, input_shape: ArrayLike[int], displacement: ArrayLike[int] | int):
        """Initialize cyclic shift layer with input shape and displacement.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            input shape of cyclic shift layer.
        displacement : ArrayLike[int] | int
            displacement along spatial axes.
        """
        super().__init__(input_shape=input_shape, output_shape=input_shape)
        self.displacement = transform_to_int_tuple(
            displacement, "displacement", length=len(input_shape) - 1
        )

    def forward(self, inputs: torch.Tensor, *other: torch.Tensor) -> torch.Tensor:
        """Forward input tensor throught cyclic shift layer.

        Parameters
        ----------
        inputs : torch.Tensor
            tensor to forward throught cyclic shift layer.
        """
        ndims = len(self.get_input_shape(0)) - 1
        if ndims == 1:
            return torch.roll(inputs, shifts=self.displacement, dims=(2,))
        elif ndims == 2:
            return torch.roll(inputs, shifts=self.displacement, dims=(2, 3))
        elif ndims == 3:
            return torch.roll(inputs, shifts=self.displacement, dims=(2, 3, 4))
        return inputs
