"""Contains implementation of mobile visual transformer block."""

from __future__ import annotations

import torch
from einops import rearrange

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike, transform_to_int_tuple
from ..conv import Conv
from ..torch_module import TorchModule


class MVitPatchMultiHeadAttentionModule(TorchModule):
    """
    Patch-based Multi-Head Self-Attention block.

    Splits the input tensor into non-overlapping patches and applies
    multi-head self-attention within each patch using PyTorch's
    scaled dot-product attention.

    Supports 1D, 2D, and 3D input tensors.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    patch_size : tuple of int
        Size of the non-overlapping patch in each spatial dimension.
    in_proj : Conv
        Input projection producing queries, keys, and values.
    out_proj : Conv
        Output projection applied after attention.
    """

    num_heads: int
    in_proj: Conv | None
    out_proj: Conv | None

    def __init__(
        self,
        input_shape: ArrayLike[int],
        num_heads: int,
        patch_size: ArrayLike[int] | int,
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
        self.patch_size = transform_to_int_tuple(
            patch_size, name="patch_size", length=len(input_shape[1:])
        )
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
        Apply patch-based multi-head self-attention to the input tensor.

        Split the input into non-overlapping patches and perform scaled
        dot-product attention within each patch region. Reassemble the
        output and apply a final projection layer.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, L), (B, C, H, W), or (B, C, D, H, W),
            where B is batch size and C is the number of channels.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as the input, after attention
            and output projection.
        """
        inputs = self.in_proj.forward(inputs)
        batch_size, num_channels, *size = inputs.size()

        if len(size) == 1:
            query, key, value = rearrange(
                inputs,
                "b (u n c) (w pw) -> b u (pw n) w c",
                u=3,
                n=self.num_heads,
                pw=self.patch_size[0],
            ).chunk(3, dim=1)

            outputs = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=False,
                enable_gqa=True,
            )

            outputs = rearrange(
                outputs, "b (pw n) w c -> b (n c) (w pw)", n=self.num_heads
            )
            return self.out_proj.forward(outputs)

        elif len(size) == 2:
            query, key, value = rearrange(
                inputs,
                "b (u n c) (h ph) (w pw) -> b u (ph pw n) (h w) c",
                u=3,
                n=self.num_heads,
                ph=self.patch_size[0],
                pw=self.patch_size[1],
            ).chunk(3, dim=1)
            outputs = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=False,
                enable_gqa=True,
            )
            outputs = rearrange(
                outputs,
                "b (ph pw n) (h w) c -> b (n c) (h ph) (w pw)",
                n=self.num_heads,
                h=size[0] // self.patch_size[0],
                w=size[1] // self.patch_size[1],
                ph=self.patch_size[0],
                pw=self.patch_size[1],
            )
            return self.out_proj.forward(outputs)

        elif len(size) == 3:
            query, key, value = rearrange(
                inputs,
                "b (u n c) (q pq) (h ph) (w pw) -> b u (pq ph pw n) (q h w) c",
                u=3,
                n=self.num_heads,
                pq=self.patch_size[0],
                ph=self.patch_size[1],
                pw=self.patch_size[2],
            ).chunk(3, dim=1)
            outputs = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=False,
                enable_gqa=True,
            )
            outputs = rearrange(
                outputs,
                "b (pq ph pw n) (q h w) c -> b (n c) (q pq) (h ph) (w pw)",
                n=self.num_heads,
                q=size[0] // self.patch_size[0],
                h=size[1] // self.patch_size[1],
                w=size[2] // self.patch_size[2],
                pq=self.patch_size[0],
                ph=self.patch_size[1],
                pw=self.patch_size[2],
            )
            return self.out_proj.forward(outputs)

        else:
            raise ValueError()
