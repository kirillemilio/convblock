"""Contains implementation of pooling layers factory."""

from __future__ import annotations

from typing import Literal

from ...utils import ArrayLike
from ..conv_block import ConvBlock
from .adaptive_avg_pooling_layer import AdaptiveAvgPool
from .adaptive_lp_pooling_layer import AdaptiveLPPool
from .adaptive_max_pooling_layer import AdaptiveMaxPool
from .avg_pooling_layer import AvgPool
from .base_pooling_layer import BasePoolLayer
from .lp_pooling_layer import LPPool
from .max_pooling_layer import MaxPool


class PoolingFactory:
    """Factory class for creating pooling layers from configuration.

    Provides a unified interface for building various types of pooling layers
    (standard and adaptive) based on input shape and pooling mode.

    The class supports registration into `ConvBlock` under short aliases like `"p"`, `"g"`, or `">"`,
    making it suitable for integration into model configuration pipelines.

    Supported pooling modes
    -----------------------
    - "max": Max pooling
    - "avg": Average pooling
    - "lp": Lp pooling
    - "adaptive_max": Adaptive max pooling
    - "adaptive_avg": Adaptive average pooling
    - "adaptive_lp": (not implemented)

    Examples
    --------
    >>> layer = PoolingFactory.create_pooling([3, 32, 32], mode="avg", kernel_size=3)
    >>> layer.output_shape
    [3, 16, 16]

    Notes
    -----
    This factory abstracts both dimensionality handling and mode switching
    and simplifies instantiation of pooling layers from config.
    """

    @ConvBlock.register_option(name="p", vectorized_params=("kernel_size", "stride", "dilation"))
    @classmethod
    def create_pooling(
        cls,
        input_shape: ArrayLike[int],
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] | int = 2,
        dilation: ArrayLike[int] | int = 1,
        mode: Literal["max", "avg", "lp", "adaptive_max", "adaptive_avg", "adaptive_lp"] = "max",
        pad_mode: Literal["reflect", "replicate", "valid", "constant"] = "constant",
        pad_value: float = 1.0,
        norm_type: float = 1.0,
        output_size: ArrayLike[int] | None = None,
    ) -> BasePoolLayer:
        """Create a pooling layer based on the provided mode and configuration.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor (C, H, W) or higher dimensional.
        kernel_size : int or tuple of int
            Size of the pooling window.
        stride : int or tuple of int
            Stride of the pooling window.
        dilation : int or tuple of int
            Dilation factor for pooling.
        mode : str
            Type of pooling. One of ["max", "avg", "lp", "adaptive_max", "adaptive_avg"].
        pad_mode : str
            Padding mode to apply before pooling. One of ["constant", "reflect", "replicate", "valid"].
        pad_value : float
            Value to use for constant padding.
        norm_type : float
            Norm value used for Lp pooling. Ignored if mode is not "lp".
        output_size : ArrayLike[int] or None
            Target output size for adaptive pooling. Required for adaptive modes.

        Returns
        -------
        BasePoolLayer
            Instance of a pooling layer corresponding to the given configuration.

        Raises
        ------
        NotImplementedError
            If the provided pooling mode is not supported or not implemented.
        """
        match mode:
            case "max":
                return MaxPool(
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding_mode=pad_mode,
                    padding_value=pad_value,
                )
            case "avg":
                return AvgPool(
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding_mode=pad_mode,
                    padding_value=pad_value,
                )
            case "lp":
                return LPPool(
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding_mode=pad_mode,
                    padding_value=pad_value,
                    norm_type=norm_type,
                )
            case "adaptive_max":
                return AdaptiveMaxPool(input_shape=input_shape, output_size=output_size)
            case "adaptive_avg":
                return AdaptiveAvgPool(input_shape=input_shape, output_size=output_size)
            case "adaptive_lp":
                return AdaptiveLPPool(
                    input_shape=input_shape, output_size=output_size, norm_type=norm_type
                )
            case _:
                raise NotImplementedError(f"Unknown pooling mode: `{mode}`")

    @ConvBlock.register_option(name="g")
    @ConvBlock.register_option(name=">")
    @classmethod
    def create_global_pooling(
        cls,
        input_shape: ArrayLike[int],
        mode: Literal["max", "avg", "lp"] = "avg",
        norm_type: float = 1.0,
    ) -> BasePoolLayer:
        """Create a global pooling layer that collapses spatial dimensions.

        Global pooling is equivalent to pooling over the entire spatial extent
        of the input tensor (e.g., HxW for 2D inputs), effectively reducing
        each feature map to a single value.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor (C, H, W) or similar.
        mode : str, optional
            Pooling type: one of "max", "avg", or "lp". Default is "avg".
        norm_type : float
            normalization power for lp pooling layer.
            Default is 1.0.

        Returns
        -------
        BasePoolLayer
            Instantiated global pooling layer.

        Raises
        ------
        ValueError
            If mode is not one of the supported types.
        """
        ndims = len(input_shape) - 1
        output_size = [1] * ndims
        match mode:
            case "max":
                return AdaptiveMaxPool(input_shape=input_shape, output_size=output_size)
            case "avg":
                return AdaptiveAvgPool(input_shape=input_shape, output_size=output_size)
            case "lp":
                return AdaptiveLPPool(
                    input_shape=input_shape, output_size=output_size, norm_type=norm_type
                )
            case _:
                raise ValueError(f"Argument 'mode' must be 'max', 'avg' or 'lp'. Got '{mode}'.")
