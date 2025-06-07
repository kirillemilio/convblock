"""Contains pytorch modules compatible with ConvBlock interface."""

import math
import operator
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.autograd.function import Function

from ..bases import ConvModule, Layer, Module
from ..utils import INT_TYPES, ArrayLike, transform_to_int_tuple
from .conv import map_initializer
from .conv_block import ConvBlock
from .custom import ChannelsShuffleFunction


class FlattenFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.reshape(input.shape[0], -1)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        return grad_output.reshape(*[int(v) for v in input.shape])

    @staticmethod
    def symbolic(g, input):
        r = g.op("Flatten", input)
        return r


@ConvBlock.register_option(name="<")
class Flatten(Module):
    """Flatten input tensor."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass method for flatten layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            flattened input tensor.
        """
        return FlattenFunction.apply(inputs)

    @property
    def output_shape(self) -> NDArray[np.int64]:
        """Get shape of the output tensor."""
        return np.array([np.prod(self.input_shape)], dtype=np.int64)

    def __repr__(self) -> str:
        """Get string representtion of flatten layer."""
        s = "{name}(input_shape={input_shape}, output_shape={output_shape})"
        return s.format(
            name=self.__class__.__name__,
            input_shape=tuple(self.input_shape),
            output_shape=self.output_shape[0],
        )


@ConvBlock.register_option(name="u")
class Upsample(Module):

    def __init__(
        self,
        input_shape: ArrayLike[int],
        scale: int = 2,
        kernel_size: ArrayLike[int] = 3,
        size: ArrayLike[int] = None,
        mode: str = "linear",
    ):
        """Generalized upsampling layer.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken into account.
        scale : int
            scale factor.
        kernel_size : int, Tuple[int], List[int] or NDArrray[int]
            kernel_size required by unpooling operation.
        size : ArrayLike[int] or None
            if not None then output tensor will have specified
            spatial shape. Note that if this parameter is provided
            then parameter scale will be ignored.
        mode : upsampling mode
            can be 'linear', 'nearest' ('bilinear' and 'trilinear'
            values are also supported).
        """
        super().__init__(input_shape)
        if size is not None:
            self.size = transform_to_int_tuple(size, "size", self.ndims - 1)
            self.scale = None
        else:
            self.size = None
            self.scale = int(scale)
        self.kernel_size = transform_to_int_tuple(kernel_size, "kernel_size", self.ndims - 1)
        if mode == "linear":
            if self.ndims == 2:
                self.mode = "linear"
            elif self.ndims == 3:
                self.mode = "bilinear"
            elif self.ndims == 4:
                self.mode = "trilinear"
        else:
            self.mode = mode

    @property
    def output_shape(self) -> NDArray[np.int64]:
        """Get shape of the output tensor."""
        if self.scale is None:
            return np.array([self.in_channels, *self.size], dtype=np.int)
        return np.array([self.input_shape[0], *(self.scale * self.input_shape[1:])])

    @property
    def stride(self) -> "Tuple[float]":
        """Get stride associated with layer.

        Returns
        -------
        Tuple[float] or None
            tuple of size ndims - 1 or None if stride
            does not exist for current layer.
        """
        return tuple(
            self.input_shape[i + 1] / self.output_shape[i + 1] for i in range(self.ndims - 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass method for upsampling layer."""
        if self.mode == "unpool":
            raise NotImplementedError("Mode 'unpool' is not implemented yet.")
        else:
            return F.interpolate(inputs, size=self.size, scale_factor=self.scale, mode=self.mode)

    def __repr__(self) -> str:
        """Get string representation of upsampling layer."""
        s = "{name}(input_shape={input_shape}, "
        s += "output_shape={output_shape}, "
        s += "mode='{mode}')"
        return s.format(
            name=self.__class__.__name__,
            mode=self.mode,
            input_shape=tuple(self.input_shape),
            output_shape=tuple(self.output_shape),
        )


@ConvBlock.register_option("n")
class BatchNorm(Layer):

    def __init__(
        self,
        input_shape: ArrayLike[int],
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """Generalized batch normalization layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        eps : float
            epsilon value, required by batch normalization operation.
            Default is 1e-05.
        momentum : float
            value of momentum for exponential moving average computation.
            Default is 0.1.
        affine : bool
            this parameter is required by native pytorch batch normalization
            layer. Default is True.

        Raises
        ------
        ValueError
            if input_shape argument has length greater than 4.
        """
        if self.ndims <= 2:
            layer = torch.nn.BatchNorm1d(self.in_channels, eps, momentum, affine)
        elif self.ndims == 3:
            layer = torch.nn.BatchNorm2d(self.in_channels, eps, momentum, affine)
        elif self.ndims == 4:
            layer = torch.nn.BatchNorm3d(self.in_channels, eps, momentum, affine)
        else:
            raise ValueError("Incorrect input shape.")
        super().__init__(input_shape=input_shape, layer=layer)


@ConvBlock.register_option(name="i")
class InstanceNorm(Layer):

    def __init__(
        self,
        input_shape: ArrayLike[int],
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """Generalized instance normalization layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        eps : float
            epsilon value, required by batch normalization operation.
            Default is 1e-05.
        momentum : float
            value of momentum for exponential moving average computation.
            Default is 0.1.
        affine : bool
            this parameter is required by native pytorch batch normalization
            layer. Default is True.

        Raises
        ------
        ValueError
            if input_shape argument has length greater than 4.
        """
        if self.ndims <= 2:
            layer = torch.nn.InstanceNorm1d(self.in_channels, eps, momentum, affine)
        elif self.ndims == 3:
            layer = torch.nn.InstanceNorm2d(self.in_channels, eps, momentum, affine)
        elif self.ndims == 4:
            layer = torch.nn.InstanceNorm3d(self.in_channels, eps, momentum, affine)
        else:
            raise ValueError("Incorrect input shape.")

        super().__init__(input_shape=input_shape, layer=layer)


@ConvBlock.register_option(name="l")
class Lambda(Module):

    def __init__(
        self,
        input_shape: "ArrayLike[int]",
        op: "callable",
        output_shape: "ArrayLike[int]" = None,
        annotation=None,
    ):
        super().__init__(input_shape)
        if not callable(op):
            raise TypeError("Argument 'op' must be callable.")
        if output_shape is None:
            self._output_shape = self.input_shape
        else:
            self._output_shape = np.array(output_shape, dtype=np.int)
        self.op = op
        self.annotation = annotation

    @property
    def output_shape(self) -> "NDArray[int]":
        """Get shape of the output tensor."""
        return self._output_shape

    def forward(self, inputs):
        """Forward pass method for Lambda layer."""
        return self.op(inputs)

    def __repr__(self) -> str:
        """String representation of Lambda layer."""
        if self.annotation is None:
            return super().__repr__()
        else:
            return self.__class__.__name__ + "(" + self.annotation + ")"


@ConvBlock.register_option(name="s")
class ChannelsShuffle(Module):

    def __init__(self, input_shape: "ArrayLike[int]"):
        """Choose random permutation for channels shuffle."""
        super().__init__(input_shape)
        permutation = np.random.permutation(input_shape[0])
        permutation = torch.LongTensor(permutation)
        self.register_buffer("permutation", torch.LongTensor(permutation))

    def forward(self, inputs):
        """Forward pass method for ChannelsShuffle layer."""
        return ChannelsShuffleFunction.apply(inputs, self.permutation)
