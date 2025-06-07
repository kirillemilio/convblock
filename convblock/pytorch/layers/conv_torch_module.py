"""Contains implementation of conv module."""

from __future__ import annotations

from ..utils import ArrayLike, transform_to_int_tuple
from .torch_module import TorchModule


class ConvTorchModule(TorchModule):
    """Convolution like torch module implementation."""

    @property
    def stride(self) -> tuple[int, ...]:
        """Get stride associated with layer.

        Returns
        -------
        Tuple[int, ...]
            tuple of size ndims - 1
        """
        return self._stride[:]

    @property
    def kernel_size(self) -> tuple[int, ...]:
        """Get kernel size associated with layer.

        Returns
        -------
        Tuple[int]
            tuple of size ndims - 1 representing
            kernel size along spatial axes.
        """
        return self._kernel_size[:]

    @property
    def dilation(self) -> tuple[int, ...]:
        """Get dilation rate associated with layer.

        Returns
        -------
        Tuple[int]
            tuple of size ndims - 1 representing
            dilation rate along spatial axes.
        """
        return self._dilation[:]

    @property
    def ndims(self) -> int:
        """Get number of spatial dimensions of convolutional like layer.

        Returns
        -------
        int
            number of spatial dimensions of convolutional like layer.
        """
        return len(self.get_input_shape(input_id=0)) - 1

    @property
    def in_channels(self) -> int:
        """Get number of input channels for convolutional-like layer.

        Returns
        -------
        int
            number of input channels for convolutional like layer.
        """
        input_shape = self.get_input_shape(input_id=0)
        return input_shape[0]

    @property
    def out_channels(self) -> int:
        """Get number of output channels for convolutional-like layer.

        Returns
        -------
        int
            number of output channels for convolutional like layer.
        """
        output_shape = self.get_output_shape(output_id=0)
        return output_shape[0]

    def __init__(
        self,
        input_shape: ArrayLike[int],
        output_shape: ArrayLike[int],
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] | int = 1,
        dilation: ArrayLike[int] | int = 1,
    ):
        ndims = len(input_shape) - 1
        if ndims not in (2, 3, 4):
            raise ValueError(
                "Input tensor must be 2, 3 or 4 dimensional "
                + " with zero axis meaning number of channels."
            )
        self._kernel_size = transform_to_int_tuple(kernel_size, "kernel_size", ndims - 1)
        self._stride = transform_to_int_tuple(stride, "stride", ndims - 1)
        self._dilation = transform_to_int_tuple(dilation, "dilation", ndims - 1)
        super().__init__(input_shape=input_shape, output_shape=output_shape)
