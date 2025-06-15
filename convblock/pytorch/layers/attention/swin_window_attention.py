"""Contains implementation of swin transformer block."""

from __future__ import annotations

import math

import numpy as np
import torch
from einops import einsum, rearrange

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike, transform_to_int_tuple
from ..conv import Conv
from ..cyclic_shift import CyclicShift
from ..torch_module import TorchModule
from .mask import generate_shift_mask


class SwinWindowAttentionModule(TorchModule):
    """
    Window-based Multi-Head Self-Attention module for image or video features.

    This module applies multi-head self-attention within non-overlapping windows.
    Optionally, it supports shifted windows and masking to enable cross-window
    connections while preserving computational efficiency, as in Swin Transformer.

    Supports input of 1D, 2D, or 3D spatial data, such as sequences, images, or volumes.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    scale : float
        Scaling factor for dot-product attention.
    window_size : tuple of int
        Size of attention windows along each spatial dimension.
    in_proj : Conv
        Linear projection for computing queries, keys, and values.
    out_proj : Conv
        Linear projection applied to attention output.
    forward_shift : CyclicShift or None
        Optional module that applies cyclic spatial shift before attention.
    backward_shift : CyclicShift or None
        Optional module that reverses the shift after attention.
    shift_mask : torch.nn.Parameter or None
        Precomputed attention mask used in shifted attention to prevent
        information leakage between different windows.
    """

    num_heads: int
    scale: float
    window_size: ArrayLike[int]
    in_proj: Conv
    out_proj: Conv

    forward_shift: CyclicShift | None
    backward_shift: CyclicShift | None
    shift_mask: torch.nn.Parameter | None

    def __init__(
        self,
        input_shape: ArrayLike[int],
        window_size: ArrayLike[int] | int,
        num_heads: int,
        in_filters: int | None = None,
        out_filters: int | None = None,
        dropout_prob: float = 0.0,
        bias: bool = False,
        shift: bool = False,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> None:
        """
        Initialize the WindowAttention module.

        Parameters
        ----------
        input_shape : array-like of int
            Shape of the input tensor as [C, ...], where C is the number of channels
            and ... are spatial dimensions (1D, 2D, or 3D).
        window_size : int or array-like of int
            Size of the attention window in each spatial dimension.
        num_heads : int
            Number of attention heads. Must evenly divide the number of input channels.
        in_filters : int, optional
            Number of filters for the input projection. If None, uses input_shape[0].
        out_filters : int, optional
            Number of filters for the output projection. If None, uses input_shape[0].
        dropout_prob : float, optional
            Dropout probability applied after attention weights. Default is 0.0.
        bias : bool, optional
            Whether to include bias in projection layers. Default is False.
        shift : bool, optional
            If True, applies cyclic shift and masking for shifted attention. Default is False.
        init_weight : dict, optional
            Configuration dictionary for initializing projection weights.
        init_bias : dict, optional
            Configuration dictionary for initializing projection biases.

        Raises
        ------
        ValueError
            If the number of input filters is not divisible by the number of heads.
        """
        super().__init__(input_shape=input_shape, output_shape=input_shape)
        self.window_size = transform_to_int_tuple(
            window_size, "window_size", len(input_shape) - 1
        )
        for dim, win in zip(input_shape[1:], self.window_size):
            if dim % win != 0:
                raise ValueError(
                    f"Dimension {dim} must be divisible by window size {win}."
                )

        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        proj_dim = input_shape[0] if in_filters is None else in_filters
        if proj_dim % num_heads:
            raise ValueError(
                f"Invalid number of filters in input projection: {proj_dim} "
                + f"must be divisible by num heads {num_heads}"
            )
        self.scale = 1.0 / math.sqrt(proj_dim / self.num_heads)
        self.in_proj = Conv(
            input_shape=input_shape,
            filters=3 * proj_dim,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            init_weight=init_weight,
            init_bias=init_bias,
        )
        self.out_proj = Conv(
            input_shape=[proj_dim, *input_shape[1:]],
            filters=out_filters if out_filters is not None else input_shape[0],
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            init_weight=init_weight,
            init_bias=init_bias,
        )

        self.forward_shift = None
        self.backward_shift = None
        self.shift_mask = None

        if shift:
            self.forward_shift = CyclicShift(
                displacement=(-s // 2 for s in self.window_size)
            )
            self.backward_shift = CyclicShift(
                displacement=(s // 2 for s in self.window_size)
            )
            mask = generate_shift_mask(
                input_shape=input_shape, window_size=self.window_size
            )
            self.shift_mask = torch.nn.Parameter(
                mask.view(
                    1, 1, -1, np.prod(self.window_size) * np.prod(self.window_size)
                ),
                requires_grad=False,
            )

    def forward(self, inputs: torch.Tensor, *other: torch.Tensor) -> torch.Tensor:
        """Apply window-based multi-head self-attention to the input.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, ...) where ... is 1D, 2D, or 3D spatial shape.

        Returns
        -------
        torch.Tensor
            Output tensor of same shape as input.
        """
        shape = self.get_input_shape(0)
        ndims = len(shape) - 1

        inputs = self.in_proj(inputs)

        if self.forward_shift is not None:
            inputs = self.forward_shift(inputs)

        if ndims == 1:
            w = shape[1]
            nw_w = w // self.window_size[0]
            inputs = rearrange(
                "b (u h d) (nw_w w_w) -> b u h nw_w w_w d",
                inputs,
                u=3,
                h=self.num_heads,
                w_w=self.window_size[0],
            )
            query, key, value = inputs.chunk(3, dim=1)
            dots = einsum("b h t i d, b h t j d -> b h t i j", query, key).mul(
                self.scale
            )
            if self.shift_mask is not None:
                dots = dots.add(self.shift_mask)
            dots = dots.softmax(dim=-1)
            dots = torch.nn.functional.dropout(
                dots, p=self.dropout_prob, training=self.training
            )
            res = einsum("b h t i j, b h t j d", dots, value)
            res = rearrange(
                "b h nw_w w_w d -> b (h d) (nw_w w_w)",
                nw_w=nw_w,
                w_w=self.window_size[0],
            )

        elif ndims == 2:
            h, w = shape[1:]
            nw_h, nw_w = h // self.window_size[0], w // self.window_size[1]
            inputs = rearrange(
                "b (u h d) (nw_h w_h) (nw_w w_w) -> b u h (nw_h nw_w) (w_h w_w) d",
                inputs,
                u=3,
                h=self.num_heads,
                w_h=self.window_size[0],
                w_w=self.window_size[1],
            )
            query, key, value = inputs.chunk(3, dim=1)
            dots = einsum("b h t i d, b h t j d -> b h t i j", query, key).mul(
                self.scale
            )
            if self.shift_mask is not None:
                dots = dots.add(self.shift_mask)
            dots = dots.softmax(dim=-1)
            dots = torch.nn.functional.dropout(
                dots, p=self.dropout_prob, training=self.training
            )
            res = einsum("b h t i j, b h t j d", dots, value)
            res = rearrange(
                "b h (nw_h nw_w) (w_h w_w) d -> b (h d) (nw_h w_h) (nw_w w_w)",
                nw_h=nw_h,
                nw_w=nw_w,
                w_h=self.window_size[0],
                w_w=self.window_size[1],
            )

        elif ndims == 3:
            q, h, w = shape[1:]
            nw_q, nw_h, nw_w = (
                q // self.window_size[0],
                h // self.window_size[1],
                w // self.window_size[2],
            )
            inputs = rearrange(
                "b (u h d) (nw_q w_q) (nw_h w_h) (nw_w w_w) -> "
                + " b u h (nw_q nw_h nw_w) (w_q w_h w_w) d",
                inputs=inputs,
                u=3,
                h=self.num_heads,
                w_q=self.window_size[0],
                w_h=self.window_size[1],
                w_w=self.window_size[2],
            )
            query, key, value = inputs.chunk(3, dim=1)
            dots = einsum("b h t i d, b h t j d -> b h t i j", query, key).mul(
                self.scale
            )
            if self.shift_mask is not None:
                dots = dots.add(self.shift_mask)
            dots = dots.softmax(dim=-1)
            dots = torch.nn.functional.dropout(
                dots, p=self.dropout_prob, training=self.training
            )
            res = einsum("b h t i j, b h t j d", dots, value)
            res = rearrange(
                "b h (nw_q nw_h nw_w) (w_q w_h w_w) d -> "
                + "b (h d) (nw_q w_q) (nw_h, w_h) (nw_w, w_w)",
                nw_q=nw_q,
                nw_h=nw_h,
                nw_w=nw_w,
                w_q=self.window_size[0],
                w_h=self.window_size[1],
                w_w=self.window_size[2],
            )
        else:
            raise ValueError("Invalid shape of input")

        res = self.out_proj(res)
        if self.backward_shift is not None:
            res = self.backward_shift(res)
        return res
