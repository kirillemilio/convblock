"""Contains implementation of deformable convolutional layer."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from tvdcn import deform_conv1d, deform_conv2d, deform_conv3d

from ...initializer import InitializerConfigDict
from ...utils import (
    ArrayLike,
    compute_direct_output_shape,
    compute_direct_same_padding,
    pad,
    transform_to_int_tuple,
)
from .base_conv_layer import BaseConvLayer
from .conv_layer import Conv


class DeformableConv(BaseConvLayer):
    """
    Implement deformable convolution layer with optional modulation.

    Supports 1D, 2D, and 3D deformable convolutions. Automatically
    constructs offset and modulation branches using regular Conv layers
    with the same kernel size, stride, and padding.

    The class is compatible with tvdcn-based deformable operators and
    defaults to standard convolution if offset and modulation are disabled.
    """

    _pad_mode: Literal["constant", "reflect", "replicate", "valid"]
    _pad_value: float
    _pad_sizes: list[int]
    _output_shape: NDArray[np.int32]

    offset_conv: Conv | None
    modulation_conv: Conv | None

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
        offset_groups: int | None = 1,
        use_offsets: bool = True,
        use_modulation: bool = True,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> None:
        """
        Initialize a deformable convolution layer with optional modulation.

        Constructs a deformable convolution with optional learned spatial
        offsets and/or modulation masks. Both offset and modulation branches
        are implemented as standard convolutional layers with identical
        spatial hyperparameters.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor, including channels.
        filters : int
            Number of output filters.
        kernel_size : ArrayLike[int] or int, optional
            Size of the convolution kernel. Default is 3.
        stride : ArrayLike[int] or int, optional
            Stride of the convolution. Default is 1.
        dilation : ArrayLike[int] or int, optional
            Dilation factor for kernel elements. Default is 1.
        groups : int, optional
            Number of groups in the main convolution. Default is 1.
        bias : bool, optional
            Whether to include bias in all internal convolutions. Default is False.
        pad_mode : {'reflect', 'replicate', 'valid', 'constant'}, optional
            Padding mode applied before convolution. Default is 'constant'.
        pad_value : float, optional
            Value used for 'constant' padding. Default is 0.0.
        offset_groups : int or None, optional
            Number of groups used to split offsets. Default is 1.
        use_offsets : bool, optional
            Whether to learn and apply spatial offsets. Default is True.
        use_modulation : bool, optional
            Whether to apply modulation mask via sigmoid. Default is True.
        init_weight : InitializerConfigDict or None, optional
            Weight initializer configuration. Default is None.
        init_bias : InitializerConfigDict or None, optional
            Bias initializer configuration. Default is None.
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

        self.offset_groups = offset_groups if offset_groups is not None else input_shape[0]

        offset_channels = 2 * np.prod(self.kernel_size) * self.offset_groups
        modulation_channels = np.prod(self.kernel_size) * self.offset_groups

        self.offset_conv = None
        self.modulation_conv = None

        if use_offsets:
            self.offset_conv = Conv(
                input_shape=input_shape,
                filters=offset_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                groups=1,
                bias=bias,
                pad_mode=pad_mode,
                pad_value=pad_value,
                init_weight=init_weight,
                init_bias=init_bias,
            )

        if use_modulation:
            self.modulation_conv = Conv(
                input_shape=input_shape,
                filters=modulation_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                groups=1,
                bias=bias,
                pad_mode=pad_mode,
                pad_value=pad_value,
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
        """Return string representation of the deformable convolution layer."""
        parts = [
            f"{self.__class__.__name__}(",
            f"{self.in_channels}, {self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self._stride}",
        ]
        if any(self.pad_sizes):
            parts.append(f"padding={tuple(self.pad_sizes)}")
            parts.append(f"padding_mode='{self.pad_mode}'")
        if self.dilation != (1,) * len(self.dilation):
            parts.append(f"dilation={self.dilation}")
        if self.groups != 1:
            parts.append(f"groups={self.groups}")
        if self.bias is None:
            parts.append("bias=False")
        if hasattr(self, "offset_conv"):
            parts.append(f"offset_groups={self.offset_groups}")
        if hasattr(self, "modulation_conv"):
            parts.append("modulated=True")
        parts.append(")")
        return ", ".join(parts)

    def get_offset(self, inputs: torch.Tensor) -> torch.Tensor | None:
        """Get offset tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            inputs tensor over which offsets will be computed.

        Returns
        -------
        torch.Tensor | None
            torch.Tensor representing offsets or None
            if not offset convolution is present.
        """
        return None if self.offset_conv is None else self.offset_conv.forward(inputs)

    def get_mask(self, inputs: torch.Tensor) -> torch.Tensor | None:
        """Get mask tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            inputs tensor over which mask will be computed.

        Returns
        -------
        torch.Tensor | None
            torch.Tensor representing mask or None
            if no mask convolution is present.
        """
        return (
            None if self.modulation_conv is None else self.modulation_conv.forward(inputs).sigmoid()
        )

    def forward(self, inputs: torch.Tensor, *others: torch.Tensor) -> torch.Tensor:
        """Forward pass method for deformable convolution layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor for deformable convolution layer.

        Returns
        -------
        Tensor
            result of deformable convolution operation applied to the input tensor.
        """
        inputs = pad(inputs, self._pad_sizes, mode=self._pad_mode, value=self._pad_value)
        ndims = len(self.get_input_shape(0)) - 1

        if ndims == 1:
            return deform_conv1d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                offset=self.get_offset(inputs),
                mask=self.get_mask(inputs),
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif ndims == 2:
            return deform_conv2d(
                inputs=inputs,
                weight=self.weight,
                bias=self.bias,
                offset=self.get_offset(inputs),
                mask=self.get_mask(inputs),
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif ndims == 3:
            return deform_conv3d(
                inputs=inputs,
                weight=self.weight,
                bias=self.bias,
                offset=self.get_offset(inputs),
                mask=self.get_mask(inputs),
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        return inputs
