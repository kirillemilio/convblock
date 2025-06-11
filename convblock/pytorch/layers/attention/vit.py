"""Contains implementation of visual transformer block."""

from __future__ import annotations

import math

import torch

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from ..conv import Conv
from ..torch_module import TorchModule


class MultiHeadAttentionModule(TorchModule):
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
        """Forward through multihead attention layer."""
        inputs = self.in_proj.forward(inputs)

        batch_size = inputs.size(0)
        num_channels = inputs.size(1)
        shape = self.get_input_shape(0)

        query, key, value = (
            inputs.view(batch_size, 3, self.num_heads, num_channels // self.num_heads, -1)
            .permute(0, 1, 2, 4, 3)
            .chunk(3, dim=1)
        )

        outputs = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

        res = outputs.transpose(3, 2).view(batch_size, num_channels, *shape[1:])
        return self.out_proj.forward(res)
