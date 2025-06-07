"""Contains implementation of SPADE block using functional conv2d."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...bases import Module
from ...initializer import IInitializer, InitializerConfigDict, InitializerFactory
from ...utils import ArrayLike


class Spade(Module):
    """
    Spatially-Adaptive Denormalization (SPADE) block with functional 1x1 convolutions.

    Normalizes the input tensor (instance or group norm), then modulates it spatially
    using a condition tensor projected via functional 1×1 convolutions (without nn.Conv2d).

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Shape of input tensor (C, H, W, ...).
    cond_channels : int
        Number of channels in the condition tensor.
    groups : int, optional
        If set, uses group normalization with given group count.
        Otherwise, uses instance normalization.
    init_weight : InitializerConfigDict or None, optional
        Initializer config for conv weights.
    init_bias : InitializerConfigDict or None, optional
        Initializer config for conv biases.
    bias : bool, default=True
        Whether to include bias in modulation projections.
    """

    gamma_weight: torch.nn.Parameter
    gamma_bias: torch.nn.Parameter | None
    beta_weight: torch.nn.Parameter | None
    beta_bias: torch.nn.Parameter | None
    cond_channels: int
    out_channels: int
    groups: int | None
    norm_type: str

    def __init__(
        self,
        input_shape: ArrayLike[int],
        cond_channels: int,
        groups: int | None = None,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__(input_shape=input_shape)
        self.out_channels = input_shape[0]
        self.cond_channels = cond_channels
        self.groups = groups
        self.norm_type = "group" if groups else "instance"

        if groups and self.out_channels % groups != 0:
            raise ValueError(f"Channels ({self.out_channels}) must be divisible by groups={groups}")

        weight_init = InitializerFactory.create_initializer(init_weight or {"init_type": "nxavier"})
        bias_init = InitializerFactory.create_initializer(init_bias or {"init_type": "nxavier"})

        # gamma projection: (out_channels, cond_channels, 1, 1)
        self.gamma_weight = torch.nn.Parameter(
            weight_init.initialize(torch.empty(self.out_channels, cond_channels, 1, 1))
        )
        self.gamma_bias = (
            torch.nn.Parameter(bias_init.initialize(torch.empty(self.out_channels)))
            if bias
            else None
        )

        # beta projection: (out_channels, cond_channels, 1, 1)
        self.beta_weight = (
            torch.nn.Parameter(
                weight_init.initialize(torch.empty(self.out_channels, cond_channels, 1, 1))
            )
            if bias
            else None
        )
        self.beta_bias = (
            torch.nn.Parameter(bias_init.initialize(torch.empty(self.out_channels)))
            if bias
            else None
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply SPADE modulation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        cond : torch.Tensor
            Condition tensor of shape (B, cond_channels, H, W).

        Returns
        -------
        torch.Tensor
            Modulated tensor of shape (B, C, H, W).
        """
        if self.norm_type == "group":
            normed = F.group_norm(x, num_groups=self.groups)
        else:
            normed = F.instance_norm(x)

        gamma = F.conv2d(cond, self.gamma_weight, bias=self.gamma_bias)
        beta = F.conv2d(cond, self.beta_weight, bias=self.beta_bias) if self.beta_weight else 0.0

        return normed * (1 + gamma) + beta

    def __repr__(self) -> str:
        return (
            f"Spade(norm_type='{self.norm_type}', cond_channels={self.cond_channels}, "
            f"out_channels={self.out_channels}, groups={self.groups}, "
            f"uses_bias={self.beta_bias is not None})"
        )
