"""Contains implementation of transposed convolutional layer."""

from __future__ import annotations

import numpy as np
import torch

from ...initializer import IInitializer, InitializerConfigDict
from ...utils import (
    ArrayLike,
    compute_transposed_padding_and_output_padding,
    transform_to_int_tuple,
)
from .base_conv_layer import BaseConvLayer


class ConvTransposed(BaseConvLayer):
    """
    N-dimensional transposed convolutional layer with auto-computed padding and shape logic.

    This layer generalizes transposed convolutions (`ConvTranspose*D`) to support
    flexible dimensions, dynamic padding/output padding computation,
    initializer support, and shape introspection.
    It builds on top of `BaseConvLayer` and provides unified behavior
    for 1D, 2D, and 3D cases.

    Features
    --------
    - Automatically infers padding and output padding for spatial alignment.
    - Computes output shape from input shape and kernel parameters.
    - Supports custom initializers for weights and biases via factory.
    - Fully compatible with modular ConvBlock registry.

    Notes
    -----
    Registered as a `ConvBlock` under name `"t"`.
    Can be constructed using vectorized params such as `kernel_size`, `stride`, `dilation`.

    Example
    -------
    >>> layer = ConvTransposed((16, 32, 32), filters=64, kernel_size=3, stride=2)
    >>> out = layer(torch.randn(1, 16, 32, 32))
    >>> out.shape  # torch.Size([1, 64, 64, 64])
    """

    _padding: list[int]
    _output_padding: list[int]

    def __init__(
        self,
        input_shape: ArrayLike[int],
        filters: int,
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] | int = 1,
        dilation: ArrayLike[int] = 1,
        groups: int = 1,
        bias: bool = False,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ):
        """
        Initialize a transposed convolutional layer with automatic padding and output shape logic.

        This layer implements transposed convolution ("deconvolution")
        with shape-consistent behavior. Padding and output padding
        are computed dynamically to preserve spatial symmetry in upsampling.
        The layer supports full configurability and optional
        weight/bias initialization via factory configs.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor excluding the batch dimension, e.g., (C, H, W).
        filters : int
            Number of output channels after transposed convolution.
        kernel_size : int or ArrayLike[int], default=3
            Size of the transposed convolution kernel.
        stride : int or ArrayLike[int], default=1
            Upsampling stride along each spatial axis. Determines output size.
        dilation : int or ArrayLike[int], default=1
            Dilation factor for transposed kernel. Scales receptive field.
        groups : int, default=1
            Number of groups for grouped convolution. Must divide both input and output channels.
        bias : bool, default=False
            Whether to include a trainable bias term in the layer.
        init_weight : InitializerConfigDict or None, optional
            Configuration dictionary for custom weight initialization.
            Uses initializer factory to construct the appropriate initializer.
        init_bias : InitializerConfigDict or None, optional
            Configuration dictionary for custom bias initialization.
            Uses initializer factory to construct the appropriate initializer.

        Notes
        -----
        - Padding and output padding are automatically calculated to produce symmetric upscaling.
        - Unlike direct padding in regular conv layers, transposed conv requires
            careful control over
        both spatial alignment and kernel overlap, especially when `stride > 1`.

        Raises
        ------
        RuntimeError
            If output shape cannot be matched with the inferred stride,
            dilation, and kernel parameters.
        """
        ndims = len(input_shape) - 1
        kernel_size_norm = transform_to_int_tuple(kernel_size, "kernel_size", ndims - 1)
        stride_norm = transform_to_int_tuple(stride, "stride", ndims - 1)
        dilation_norm = transform_to_int_tuple(dilation, "dilation", ndims - 1)
        self._padding = []
        self._output_padding = []

        spatial_in = self.input_shape[1:]
        spatial_out = self.compute_output_shape(input_shape=input_shape, stride=stride_norm)

        for i, (inp, out, k, s, d) in enumerate(
            zip(spatial_in, spatial_out, kernel_size_norm, stride_norm, dilation_norm)
        ):
            p, op = compute_transposed_padding_and_output_padding(inp, out, k, s, d)
            self._padding.append(p)
            self._output_padding.append(op)

        output_shape = np.array([self.filters, *spatial_out], dtype=np.int64)
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            filters=filters,
            kernel_size=kernel_size_norm,
            stride=stride_norm,
            dilation=dilation_norm,
            groups=groups,
            bias=bias,
            init_weight=init_weight,
            init_bias=init_bias,
        )

    @classmethod
    def compute_output_shape(cls, input_shape: ArrayLike[int], stride: ArrayLike[int]) -> list[int]:
        """
        Compute the expected output spatial shape (excluding batch and channels).

        Assumes 'same-mode' transposed convolution — upsampling input by stride.
        Does not take into account padding or kernel effects for simplicity.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            input shape array.
        stride : ArrayLike[int]
            stride array.

        Returns
        -------
        list[int]
            List of spatial output sizes per dimension, e.g., [H_out, W_out].
        """
        return [s * i for s, i in zip(stride, input_shape)]  # naive upscaling for same-mode

    @property
    def stride(self) -> tuple[float, ...]:
        """
        Get normalized inverse stride for introspection.

        Returns
        -------
        tuple[float, ...]
            Stride values per dimension, as fractional upsampling ratios (1 / s).
        """
        return tuple(float(1.0 / s) for s in self._stride)

    def _create_weight(self, initializer: IInitializer | None = None) -> torch.nn.Parameter:
        """Create and initialize weight tensor as torch.nn.Parameter.

        Parameters
        ----------
        initializer : IInitializer or None, optional
            Initializer instance used to initialize the weight values.

        Returns
        -------
        torch.nn.Parameter
            Weight parameter tensor of shape
            (filters, in_channels // groups, *kernel_size).
        """
        weight = torch.nn.Parameter(
            torch.Tensor(self.in_channels, self.filters // self.groups, *self.kernel_size)
        )
        if initializer is not None:
            weight = initializer.initialize(weight)
        return torch.nn.Parameter(weight)

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transposed convolution operation.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, ...), where `...` is spatial dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor after applying transposed convolution.
        """
        ndims = len(self.get_input_shape(0)) - 1

        if ndims == 1:
            return torch.nn.functional.conv_transpose1d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self._padding,
                output_padding=self._output_padding,
                groups=self.groups,
                dilation=self.dilation,
            )
        elif ndims == 2:
            return torch.nn.functional.conv_transpose2d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self._padding,
                output_padding=self._output_padding,
                groups=self.groups,
                dilation=self.dilation,
            )
        elif ndims == 3:
            return torch.nn.functional.conv_transpose3d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self._padding,
                output_padding=self._output_padding,
                groups=self.groups,
                dilation=self.dilation,
            )
        return inputs

    def __repr__(self) -> str:
        """
        Return a string representation of the layer configuration.

        Includes kernel size, stride, dilation, padding, groups, bias presence, etc.

        Returns
        -------
        str
            Readable layer configuration string.
        """
        s = "{name} ({in_channels}, {out_channels}, kernel_size={}" ", stride={stride}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        values_dict = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "groups": self.groups,
            "bias": self.bias,
            "dilation": self.dilation,
            "stride": self._stride,
            "padding": tuple(self._padding),
            "output_padding": tuple(self._output_padding),
        }
        return s.format(name=self.__class__.__name__, **values_dict)
