"""Contains implementation of adaptive max pooling layer."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ...bases import Module
from ...utils import ArrayLike, transform_to_int_tuple


class AdaptiveMaxPool(Module):
    """Adaptive max pooling module implementation.

    Attributes
    ----------
    output_shape : Tuple[int, ...]
        output shape.
    """

    def __init__(self, input_shape: ArrayLike[int], output_size: ArrayLike[int]):
        """Adaptive MaxPooling module generalized for different dimensions.

        All pooling layers from this module slightly
        extends functionality of original torch.nn.*Pool modules into four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.
        2) Shape of the output tensor can be accessed by 'output_shape'
        property of module.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        output_size : ArrayLike[int]
            output spatial size for adaptive pool layer.
        """
        super().__init__(input_shape)
        self.output_size = transform_to_int_tuple(output_size, "output_size", self.ndims - 1)

    @property
    def output_shape(self) -> NDArray[np.int64]:
        """Get output shape of Adaptive MaxPooling module."""
        return np.array([self.input_shape[0], *self.output_size], dtype=np.int64)

    @property
    def stride(self) -> tuple[int, ...]:
        """Get stride for Adaptive MaxPooling module."""
        return tuple(float(s) for s in (self.input_shape[1:] / self.output_shape[1:]))

    def __repr__(self) -> str:
        """Get string representation of the module."""
        s = "{name}(input_shape={input_shape}, output_shape={output_shape})"
        values_dict = {"input_shape": self.input_shape, "output_shape": self.output_shape}
        return s.format(name=self.__class__.__name__, **values_dict)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
            return torch.nn.functional.adaptive_max_pool1d(
                input=inputs, output_size=self.output_size
            )
        elif self.ndims == 3:
            return torch.nn.functional.adaptive_max_pool2d(
                input=inputs, output_size=self.output_size
            )
        elif self.ndims == 4:
            return torch.nn.functional.adaptive_max_pool3d(
                input=inputs, output_size=self.output_size
            )
        return inputs
