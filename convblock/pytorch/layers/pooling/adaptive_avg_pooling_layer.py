"""Contains implementation of adaptive average pooling layer."""

from __future__ import annotations

import torch

from ...utils import ArrayLike, transform_to_int_tuple
from ..torch_module import TorchModule


class AdaptiveAvgPool(TorchModule):
    """Adaptive average pooling module.

    Attributes
    ----------
    output_size : Tuple[int, ...]
        output size from adaptive average pooling layer.
    """

    output_size: tuple[int, ...]

    def __init__(self, input_shape: ArrayLike[int], output_size: ArrayLike[int]):
        """Adaptive AvgPooling module generalized for different dimensions.

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
        ndims = len(input_shape) - 1
        output_size = transform_to_int_tuple(output_size, "output_size", ndims - 1)
        super().__init__(input_shape=input_shape, output_shape=[input_shape[0], *output_size])
        self.output_size = output_size

    def __repr__(self) -> str:
        """Get string representation of the module."""
        s = "{name}(input_shape={input_shape}, output_shape={output_shape})"
        values_dict = {"input_shape": self.input_shape, "output_shape": self.output_shape}
        return s.format(name=self.__class__.__name__, **values_dict)

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
            return torch.nn.functional.adaptive_avg_pool1d(
                input=inputs, output_size=self.output_size
            )
        elif ndims == 2:
            return torch.nn.functional.adaptive_avg_pool2d(
                input=inputs, output_size=self.output_size
            )
        elif ndims == 3:
            return torch.nn.functional.adaptive_avg_pool3d(
                input=inputs, output_size=self.output_size
            )
        return inputs
