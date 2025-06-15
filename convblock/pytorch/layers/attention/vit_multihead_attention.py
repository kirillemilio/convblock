"""Contains implementation of visual transformer block."""

from __future__ import annotations

import torch

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from ..conv import Conv
from ..torch_module import TorchModule


class VitMultiHeadAttentionModule(TorchModule):
    """
    Multi-head attention module for visual transformers.

    Applies a projection to obtain query, key, and value tensors, computes
    scaled dot-product attention across multiple heads, and projects the
    result back to the output space.

    Supports generic tensor inputs with 1D, 2D, or 3D spatial structure.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    in_proj : Conv or None
        Convolutional projection used to compute Q, K, and V.
    out_proj : Conv or None
        Output projection applied after attention aggregation.
    dropout_prob : float
        Dropout probability applied to attention weights during training.
    """

    num_heads: int
    in_proj: Conv | None
    out_proj: Conv | None

    def __init__(
        self,
        input_shape: ArrayLike[int],
        num_heads: int,
        in_filters: int | None = None,
        out_filters: int | None = None,
        dropout_prob: float = 0.1,
        bias: bool = False,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> None:
        """Initialize multi-head attention module.

        Parameters
        ----------
        input_shape : array-like of int
            Shape of input tensor as [C, ...], where C is channel count.
        num_heads : int
            Number of attention heads.
        in_filters : int, optional
            Number of filters in the input projection. Defaults to C.
        out_filters : int, optional
            Number of filters in the output projection. Defaults to C.
        dropout_prob : float, optional
            Dropout probability on attention weights. Default is 0.1.
        bias : bool, optional
            Whether to add bias in projection layers. Default is False.
        init_weight : dict, optional
            Configuration for weight initializer.
        init_bias : dict, optional
            Configuration for bias initializer.

        Raises
        ------
        ValueError
            If in_filters is not divisible by num_heads.
        """
        super().__init__(input_shape=input_shape, output_shape=input_shape)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        proj_dim = input_shape[0] if in_filters is None else in_filters
        if proj_dim % num_heads:
            raise ValueError(
                f"Invalid number of filters in input projection: {proj_dim} "
                + f"must be divisible by num heads {num_heads}"
            )
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
            input_shape=[in_filters, *input_shape[1:]],
            filters=out_filters if out_filters is not None else input_shape[0],
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            init_weight=init_weight,
            init_bias=init_bias,
        )

    def forward(self, inputs: torch.Tensor, *other: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention to the input.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, ...) where ... are spatial dims.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C, ...) after attention and projection.
        """
        inputs = self.in_proj.forward(inputs)

        batch_size = inputs.size(0)
        num_channels = inputs.size(1)
        shape = self.get_input_shape(0)

        query, key, value = (
            inputs.view(
                batch_size, 3, self.num_heads, num_channels // self.num_heads, -1
            )
            .permute(0, 1, 2, 4, 3)
            .chunk(3, dim=1)
        )

        outputs = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
            enable_gqa=True,
        )

        res = outputs.transpose(3, 2).view(batch_size, num_channels, *shape[1:])
        return self.out_proj.forward(res)
