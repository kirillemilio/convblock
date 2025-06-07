"""Contains implementation of normalization layer."""

from __future__ import annotations

from typing import Literal

import torch

from ...bases import Layer
from ...utils import ArrayLike
from ..conv_block import ConvBlock


@ConvBlock.register_option("n")
class NormLayer(Layer):
    """
    Generalized normalization layer with support for spectral norm and multiple modes.

    Supported modes:
        - "batch": BatchNorm1d/2d/3d
        - "sync_batch": SyncBatchNorm
        - "instance": InstanceNorm1d/2d/3d
        - "layer": LayerNorm
        - "group": GroupNorm

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Input shape excluding batch dimension (e.g., [C, H, W]).
    eps : float, default=1e-5
        Epsilon value added to avoid division by zero.
    momentum : float, default=0.1
        Momentum for batch statistics.
    mode : Literal["batch", "sync_batch", "instance", "layer", "group"]
        Normalization strategy.
    affine : bool, default=True
        Whether to include learnable scale and shift.
    num_groups : int or None, optional
        Number of groups for GroupNorm. Required if mode == "group".
    """

    def __init__(
        self,
        input_shape: ArrayLike[int],
        eps: float = 1e-5,
        momentum: float = 0.1,
        mode: Literal["batch", "sync_batch", "instance", "layer", "group"] = "batch",
        affine: bool = True,
        num_groups: int | None = None,
    ):
        self.mode = mode.lower()

        if len(input_shape[1:]) > 3:
            raise ValueError(f"Unsupported input dims: {self.ndims}. Expected 1D-3D.")

        if self.mode == "group" and num_groups is None:
            raise ValueError("GroupNorm requires num_groups argument.")

        norm_layer = self._select_norm(
            eps=eps,
            momentum=momentum,
            affine=affine,
            num_groups=num_groups,
        )

        super().__init__(input_shape=input_shape, layer=norm_layer)

    def _select_norm(
        self,
        ndims: int,
        eps: float,
        momentum: float,
        affine: bool,
        num_groups: int | None,
    ) -> torch.nn.Module:
        if self.mode == "batch":
            return {
                1: torch.nn.BatchNorm1d,
                2: torch.nn.BatchNorm2d,
                3: torch.nn.BatchNorm3d,
            }[
                self.ndims
            ](self.in_channels, eps=eps, momentum=momentum, affine=affine)

        elif self.mode == "sync_batch":
            return {
                1: torch.nn.SyncBatchNorm,
                2: torch.nn.SyncBatchNorm,
                3: torch.nn.SyncBatchNorm,
            }[ndims](self.in_channels, eps=eps, momentum=momentum, affine=affine)

        elif self.mode == "instance":
            return {
                1: torch.nn.InstanceNorm1d,
                2: torch.nn.InstanceNorm2d,
                3: torch.nn.InstanceNorm3d,
            }[ndims](self.in_channels, eps=eps, momentum=momentum, affine=affine)

        elif self.mode == "layer":
            return torch.nn.LayerNorm(self.input_shape, eps=eps, elementwise_affine=affine)

        elif self.mode == "group":
            return torch.nn.GroupNorm(
                num_groups=num_groups, num_channels=self.in_channels, eps=eps, affine=affine
            )

        else:
            raise ValueError(f"Unsupported normalization mode: {self.mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def __repr__(self) -> str:
        spectral = (
            " + spectral" if isinstance(self.layer, torch.nn.utils.spectral_norm.__class__) else ""
        )
        return f"NormLayer(mode={self.mode}, in_channels={self.in_channels}, ndim={self.ndims}){spectral}"
