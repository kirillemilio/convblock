"""Contains implementation of convolutional layer."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from ...initializer import InitializerConfigDict
from ...utils import (
    ArrayLike,
    compute_direct_output_shape,
    compute_direct_same_padding,
    pad,
    transform_to_int_tuple,
)
from ..conv_block import ConvBlock
from .base_conv_layer import BaseConvLayer


@ConvBlock.register_option(name="c", vectorized_params=("kernel_size", "stride", "dilation"))
class Conv(BaseConvLayer):
    """
    Standard N-dimensional convolutional layer with configurable padding logic.

    This class implements a convolutional layer supporting 1D, 2D, or 3D convolutions
    using PyTorch's functional API. It supports flexible padding strategies (including
    'same' behavior via explicit padding calculation) and optional weight/bias initializers.

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Shape of the input tensor as (channels, *spatial_dims).
    filters : int
        Number of output channels (filters) after convolution.
    kernel_size : int or ArrayLike[int], default=3
        Size of the convolutional kernel.
    stride : int or ArrayLike[int], default=1
        Stride value(s) for each spatial dimension.
    dilation : int or ArrayLike[int], default=1
        Dilation rate(s) for the convolution kernel.
    groups : int, default=1
        Number of blocked connections from input channels to output channels.
    bias : bool, default=False
        Whether to include a learnable bias parameter.
    pad_mode : {"constant", "reflect", "replicate", "valid"}, default="constant"
        Padding strategy used prior to convolution.
    pad_value : float, default=0.0
        Constant value used for 'constant' padding.
    init_weight : InitializerConfigDict or None, optional
        Configuration dictionary for weight initializer.
    init_bias : InitializerConfigDict or None, optional
        Configuration dictionary for bias initializer.

    Attributes
    ----------
    pad_sizes : list[int]
        Computed padding values for each spatial dimension (before and after).
    pad_mode : str
        Padding strategy as string ("constant", "reflect", "replicate", "valid").
    pad_value : float
        Value to use if constant padding is selected.
    output_shape : NDArray[np.int32]
        Shape of output tensor after applying padding and convolution.

    Examples
    --------
    >>> layer = Conv(input_shape=(3, 64, 64), filters=32, kernel_size=3, stride=1)
    >>> x = torch.randn(1, 3, 64, 64)
    >>> y = layer.forward(x)
    >>> y.shape
    torch.Size([1, 32, 64, 64])
    """

    _pad_mode: Literal["constant", "reflect", "replicate", "valid"]
    _pad_value: float
    _pad_sizes: list[int]
    _output_shape: NDArray[np.int32]

    def __init__(
        self,
        input_shape: ArrayLike[int],
        filters: int,
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] | int = 1,
        dilation: ArrayLike[int] | int = 1,
        groups: int = 1,
        bias: bool = False,
        pad_mode: Literal["reflect", "replicate", "valid", "constant"] = "constant",
        pad_value: float = 0.0,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ):
        """
        Initialize a convolutional layer with explicit padding behavior and optional weight initialization.

        This constructor sets up a standard convolutional layer with configurable kernel, stride,
        dilation, grouping, and padding parameters. Padding is handled manually based on the selected
        mode, mimicking 'same' or 'valid' behaviors. The layer supports optional weight and bias
        initialization via pluggable initializer configs.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor in (C, H, W) or (C, D, H, W), etc., format.
        filters : int
            Number of output channels after convolution.
        kernel_size : int or ArrayLike[int], default=3
            Size of the convolution kernel. Can be scalar or list-like for N-D input.
        stride : int or ArrayLike[int], default=1
            Step size of the convolution. Supports per-dimension specification.
        dilation : int or ArrayLike[int], default=1
            Dilation factor for kernel elements, effectively spacing out kernel points.
        groups : int, default=1
            Number of groups for grouped convolution. Must divide both in_channels and out_channels.
        bias : bool, default=False
            Whether to include a learnable bias term.
        pad_mode : {"reflect", "replicate", "valid", "constant"}, default="constant"
            Padding strategy to apply before convolution.
        pad_value : float, default=0.0
            Value used for padding when pad_mode is "constant".
        init_weight : InitializerConfigDict or None, optional
            Configuration for initializing the weight tensor. If None, default initialization is used.
        init_bias : InitializerConfigDict or None, optional
            Configuration for initializing the bias tensor. If None and `bias=True`, default is used.

        Notes
        -----
        If `pad_mode` is not "valid", the layer will compute explicit symmetric padding to emulate
        "same" padding for the given kernel, stride, and dilation. This allows deterministic
        output shapes without relying on built-in PyTorch padding.

        Raises
        ------
        ValueError
            If padding or shape computations fail due to invalid parameter combinations.
        """
        ndims = len(input_shape) - 1
        kernel_size_norm = transform_to_int_tuple(kernel_size, "kernel_size", ndims - 1)
        stride_norm = transform_to_int_tuple(stride, "stride", ndims - 1)
        dilation_norm = transform_to_int_tuple(dilation, "dilation", ndims - 1)

        self._pad_mode = pad_mode
        self._pad_value = pad_value
        if self._pad_mode == "valid":
            self._pad_sizes = [0] * (ndims - 1) * 2
        else:
            self._pad_sizes = compute_direct_same_padding(
                kernel_size=kernel_size_norm, stride=stride_norm, dilation=dilation_norm
            )

        _shape = compute_direct_output_shape(
            input_shape=input_shape[1:],
            kernel_size=kernel_size_norm,
            stride=stride_norm,
            dilation=dilation_norm,
            padding=self._pad_sizes,
        )

        output_shape = np.array([filters, *_shape], dtype=np.int64)
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

    @property
    def pad_sizes(self) -> list[int]:
        """Get padding sizes.

        Returns
        -------
        list[int]
            list padding sizes.
            [pad_x_l, pad_x_r, ...].
        """
        return self._pad_sizes

    @property
    def pad_mode(self) -> Literal["constant", "reflect", "replicate", "valid"]:
        """Get padding model.

        Returns
        -------
        Literal["constant", "reflect", "replicate", "valid"]
            padding mode.
        """
        return self._pad_mode

    @property
    def pad_value(self) -> float:
        """Get padding value.

        Returns
        -------
        float
            padding value.
        """
        return self._pad_value

    def __repr__(self) -> str:
        """Get string representation of convolutional layer."""
        s = "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if tuple(self.pad_sizes) != (0,) * len(self.pad_sizes):
            s += ", padding={padding}"
            s += ", padding_mode={padding_mode}"
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
            "padding": tuple(self.pad_sizes),
            "padding_mode": self.pad_mode,
            "groups": self.groups,
            "bias": self.bias,
            "dilation": self.dilation,
            "stride": self._stride,
        }
        return s.format(name=self.__class__.__name__, **values_dict)

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method for tranposed convolution layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor for transposed convolution layer.

        Returns
        -------
        Tensor
            result of convolutional operation applied to the input tensor.
        """
        inputs = pad(inputs, self._pad_sizes, mode=self._pad_mode, value=self._pad_value)

        if self.ndims == 2:
            return torch.nn.functional.conv1d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.ndims == 3:
            return torch.nn.functional.conv2d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.ndims == 4:
            return torch.nn.functional.conv3d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        return inputs
