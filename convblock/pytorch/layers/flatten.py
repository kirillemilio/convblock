"""Contains implementation of flatten operation as a separate module."""

from __future__ import annotations

import numpy as np
import torch

from ..utils import ArrayLike
from .base_module import BaseModule
from .conv_block import ConvBlock


class FlattenFunction(torch.autograd.Function):
    """
    Implement flattening operation for autograd and ONNX export.

    This function reshapes the input tensor to flatten all dimensions
    except the batch dimension.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        Flatten input tensor to shape (B, -1).

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (B, C, ...).

        Returns
        -------
        torch.Tensor
            Flattened tensor of shape (B, N), where N = product of remaining dims.
        """
        ctx.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient for flattened tensor during backprop.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient tensor of shape (B, N).

        Returns
        -------
        torch.Tensor
            Reshaped gradient matching original input shape.
        """
        return grad_output.reshape(*ctx.input_shape)

    @staticmethod
    def symbolic(g, input):
        """
        Define symbolic representation for ONNX export.

        Parameters
        ----------
        g : ONNX graph
            ONNX graph builder.
        input : torch.Value
            Input value.

        Returns
        -------
        torch.Value
            Flattened value.
        """
        return g.op("Flatten", input)


@ConvBlock.register_option("<")
class FlattenLayer(BaseModule):
    """
    Flatten input tensor while preserving batch dimension.

    This module reshapes the input tensor from shape (B, C, H, W, ...)
    to (B, C * H * W * ...), commonly used before feeding data to
    linear layers.
    """

    def __init__(self, input_shape: ArrayLike[int]) -> None:
        input_shape = np.array(input_shape, dtype=np.int64)
        input_shape = input_shape[np.newaxis, ...]
        output_shape = np.array([np.prod(input_shape.shape)], dtype=np.int64)
        super().__init__(input_shape=input_shape, output_shape=output_shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply flattening to input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, H, W, ...).

        Returns
        -------
        torch.Tensor
            Flattened tensor of shape (B, N), where N is total number of features.
        """
        return FlattenFunction.apply(inputs)

    def __repr__(self) -> str:
        """
        Return string representation of Flatten module.

        Returns
        -------
        str
            Summary of input/output shape information.
        """
        return (
            f"{self.__class__.__name__}("
            f"input_shape={tuple(self.input_shape[0])}, "
            f"output_shape={self.output_shape[0]})"
        )
