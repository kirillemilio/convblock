"""Contains implementation of deformable transposed convolutional layer."""

from __future__ import annotations

import numpy as np
import torch
from tvdcn import deform_conv_transpose1d, deform_conv_transpose2d, deform_conv_transpose3d

from ...initializer import InitializerConfigDict
from ...utils import (
    ArrayLike,
    compute_transposed_padding_and_output_padding,
    transform_to_int_tuple,
)
from .base_conv_layer import BaseConvLayer
from .conv_transposed_layer import ConvTransposed


class DeformableConvTransposed(BaseConvLayer):
    """Implements deformable transposed convolution with optional offsets and modulation mask."""

    offset_conv: ConvTransposed | None
    modulation_conv: ConvTransposed | None

    def __init__(
        self,
        input_shape: ArrayLike[int],
        filters: int,
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] | int = 1,
        dilation: ArrayLike[int] = 1,
        groups: int = 1,
        bias: bool = False,
        offset_groups: int | None = 1,
        use_offsets: bool = True,
        use_modulation: bool = True,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> None:
        ndims = len(input_shape) - 1
        kernel_size_norm = transform_to_int_tuple(kernel_size, "kernel_size", ndims - 1)
        stride_norm = transform_to_int_tuple(stride, "stride", ndims - 1)
        dilation_norm = transform_to_int_tuple(dilation, "dilation", ndims - 1)

        spatial_in = input_shape[1:]
        spatial_out = [s * i for s, i in zip(stride_norm, spatial_in)]

        padding = []
        output_padding = []
        for i, (inp, out, k, s, d) in enumerate(
            zip(spatial_in, spatial_out, kernel_size_norm, stride_norm, dilation_norm)
        ):
            p, op = compute_transposed_padding_and_output_padding(inp, out, k, s, d)
            padding.append(p)
            output_padding.append(op)

        output_shape = np.array([filters, *spatial_out], dtype=np.int64)
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

        self._padding = padding
        self._output_padding = output_padding
        self.offset_groups = offset_groups or input_shape[0]

        offset_channels = 2 * np.prod(self.kernel_size) * self.offset_groups
        modulation_channels = np.prod(self.kernel_size) * self.offset_groups

        self.offset_conv = None
        self.modulation_conv = None

        if use_offsets:
            self.offset_conv = ConvTransposed(
                input_shape=input_shape,
                filters=offset_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                groups=1,
                bias=bias,
                init_weight=init_weight,
                init_bias=init_bias,
            )

        if use_modulation:
            self.modulation_conv = ConvTransposed(
                input_shape=input_shape,
                filters=modulation_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                groups=1,
                bias=bias,
                init_weight=init_weight,
                init_bias=init_bias,
            )

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
            return deform_conv_transpose1d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                offset=self.get_offset(inputs),
                mask=self.get_mask(inputs),
                stride=self.stride,
                padding=self._padding,
                output_padding=self._output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif ndims == 2:
            return deform_conv_transpose2d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                offset=self.get_offset(inputs),
                mask=self.get_mask(inputs),
                stride=self.stride,
                padding=self._padding,
                output_padding=self._output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif ndims == 3:
            return deform_conv_transpose3d(
                input=inputs,
                weight=self.weight,
                bias=self.bias,
                offset=self.get_offset(inputs),
                mask=self.get_mask(inputs),
                stride=self.stride,
                padding=self._padding,
                output_padding=self._output_padding,
                dilation=self.dilation,
                groups=self.groups,
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
        parts = [
            f"{self.__class__.__name__}(",
            f"{self.in_channels}, {self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
        ]
        if self.dilation != (1,) * len(self.dilation):
            parts.append(f"dilation={self.dilation}")
        if self.groups != 1:
            parts.append(f"groups={self.groups}")
        if self.bias is None:
            parts.append("bias=False")
        if self.offset_conv is not None:
            parts.append(f"offset_groups={self.offset_groups}")
        if self.modulation_conv is not None:
            parts.append("modulated=True")
        parts.append(
            f"padding={tuple(self._padding)}, output_padding={tuple(self._output_padding)}"
        )
        parts.append(")")
        return ", ".join(parts)
