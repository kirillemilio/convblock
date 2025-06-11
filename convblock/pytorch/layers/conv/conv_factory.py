"""Contains implementation of convolutional layer factory."""

from __future__ import annotations

from typing import Literal

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from ..conv_block import ConvBlock
from .base_conv_layer import BaseConvLayer
from .conv_layer import Conv
from .conv_transposed_layer import ConvTransposed
from .deformable_conv_layer import DeformableConv
from .deformable_conv_transposed_layer import DeformableConvTransposed


class ConvFactory:
    """
    Factory class for creating convolutional layers.

    Provides a unified interface for instantiating either standard or deformable
    convolutional layers, including both regular and transposed variants. The layer
    type is automatically determined based on `use_offsets` and `use_modulation` flags.

    This class is useful when building dynamic model architectures that may optionally
    include deformable kernels or modulation masks.
    """

    @ConvBlock.register_option(name="c", vectorized_params=("kernel_size", "stride", "dilation"))
    @classmethod
    def create_conv(
        cls,
        input_shape: ArrayLike[int],
        filters: int,
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] = 1,
        dilation: ArrayLike[int] = 1,
        groups: int = 1,
        bias: bool = False,
        pad_mode: Literal["reflect", "replicate", "valid", "constant"] = "constant",
        pad_value: float = 0.0,
        offset_groups: int | None = 1,
        use_offsets: bool = False,
        use_modulation: bool = False,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> BaseConvLayer:
        """
        Create a convolutional layer (standard or deformable).

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor as (C, D1, D2, ...).
        filters : int
            Number of output channels (filters).
        kernel_size : ArrayLike[int] | int, default=3
            Size of the convolution kernel.
        stride : ArrayLike[int], default=1
            Stride of the convolution.
        dilation : ArrayLike[int], default=1
            Dilation rate of the convolution kernel.
        groups : int, default=1
            Number of groups for grouped convolution.
        bias : bool, default=False
            Whether to include a learnable bias term.
        pad_mode : Literal["reflect", "replicate", "valid", "constant"], default="constant"
            Padding mode applied before convolution.
        pad_value : float, default=0.0
            Value used for constant padding if `pad_mode="constant"`.
        offset_groups : int | None, default=1
            Number of offset groups used in deformable convolution.
        use_offsets : bool, default=False
            Whether to enable offset-based deformation of kernel sampling.
        use_modulation : bool, default=False
            Whether to enable modulation mask for adaptive kernel reweighting.
        init_weight : InitializerConfigDict | None, default=None
            Initializer configuration for weight tensor.
        init_bias : InitializerConfigDict | None, default=None
            Initializer configuration for bias tensor.

        Returns
        -------
        BaseConvLayer
            An instance of either `Conv` or `DeformableConv`, depending on `use_offsets`
            and `use_modulation`.
        """
        if use_offsets or use_modulation:
            return DeformableConv(
                input_shape=input_shape,
                filters=filters,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                pad_mode=pad_mode,
                pad_value=pad_value,
                offset_groups=offset_groups,
                use_offsets=use_offsets,
                use_modulation=use_modulation,
                init_weight=init_weight,
                init_bias=init_bias,
            )
        return Conv(
            input_shape=input_shape,
            filters=filters,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            pad_mode=pad_mode,
            pad_value=pad_value,
            init_weight=init_weight,
            init_bias=init_bias,
        )

    @ConvBlock.register_option(name="t", vectorized_params=("kernel_size", "stride", "dilation"))
    @classmethod
    def create_transposed_conv(
        cls,
        input_shape: ArrayLike[int],
        filters: int,
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] = 1,
        dilation: ArrayLike[int] = 1,
        groups: int = 1,
        bias: bool = False,
        offset_groups: int | None = 1,
        use_offsets: bool = False,
        use_modulation: bool = False,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> BaseConvLayer:
        """
        Create a transposed convolutional layer (standard or deformable).

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor as (C, D1, D2, ...).
        filters : int
            Number of output channels (filters).
        kernel_size : ArrayLike[int] | int, default=3
            Size of the convolution kernel.
        stride : ArrayLike[int], default=1
            Stride of the convolution.
        dilation : ArrayLike[int], default=1
            Dilation rate of the convolution kernel.
        groups : int, default=1
            Number of groups for grouped convolution.
        bias : bool, default=False
            Whether to include a learnable bias term.
        offset_groups : int | None, default=1
            Number of offset groups used in deformable convolution.
        use_offsets : bool, default=False
            Whether to enable offset-based deformation of kernel sampling.
        use_modulation : bool, default=False
            Whether to enable modulation mask for adaptive kernel reweighting.
        init_weight : InitializerConfigDict | None, default=None
            Initializer configuration for weight tensor.
        init_bias : InitializerConfigDict | None, default=None
            Initializer configuration for bias tensor.

        Returns
        -------
        BaseConvLayer
            An instance of either `ConvTransposed` or `DeformableConvTransposed`,
            depending on flags.
        """
        if use_offsets or use_modulation:
            return DeformableConvTransposed(
                input_shape=input_shape,
                filters=filters,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                offset_groups=offset_groups,
                use_offsets=use_offsets,
                use_modulation=use_modulation,
                init_weight=init_weight,
                init_bias=init_bias,
            )
        return ConvTransposed(
            input_shape=input_shape,
            filters=filters,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            init_weight=init_weight,
            init_bias=init_bias,
        )
