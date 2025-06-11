"""Contains implementation of normalization layer."""

from __future__ import annotations

from typing import Literal

import torch

from ...utils import ArrayLike
from ..conv_block import ConvBlock
from ..torch_module import TorchModule


@ConvBlock.register_option("n")
class NormLayer(TorchModule):
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
        ndims = len(input_shape) - 1
        self.mode = mode.lower()

        if len(input_shape[1:]) > 3:
            raise ValueError(f"Unsupported input dims: {ndims}. Expected 1D-3D.")

        if self.mode == "group" and num_groups is None:
            raise ValueError("GroupNorm requires num_groups argument.")

        super().__init__(input_shape=input_shape, output_shape=input_shape)
        self.layer = self._select_norm(
            ndims=ndims,
            input_shape=input_shape,
            mode=mode,
            eps=eps,
            momentum=momentum,
            affine=affine,
            num_groups=num_groups,
        )

    @classmethod
    def _select_norm(
        cls,
        ndims: int,
        input_shape: ArrayLike[int],
        mode: Literal["batch", "sync_batch", "layer", "group"],
        eps: float,
        momentum: float,
        affine: bool,
        num_groups: int | None,
    ) -> torch.nn.Module:
        """
        Select appropriate PyTorch normalization layer based on mode.

        Parameters
        ----------
        ndims : int
            Number of spatial dimensions (1D, 2D, or 3D).
        input_shape : ArrayLike[int]
            Input shape in format (C, ...), excluding batch dimension.
        mode : {"batch", "sync_batch", "instance", "layer", "group"}
            Type of normalization to apply.
        eps : float
            Small constant added to avoid divide-by-zero.
        momentum : float
            Momentum value for running statistics.
        affine : bool
            Whether to include learnable affine parameters.
        num_groups : int or None
            Number of groups for GroupNorm. Required if mode is "group".

        Returns
        -------
        torch.nn.Module
            Instantiated PyTorch normalization module.

        Raises
        ------
        ValueError
            If the provided normalization mode is not supported.
        """
        num_channels = input_shape[0]
        if mode == "batch":
            return {
                1: torch.nn.BatchNorm1d,
                2: torch.nn.BatchNorm2d,
                3: torch.nn.BatchNorm3d,
            }[
                ndims
            ](num_channels, eps=eps, momentum=momentum, affine=affine)

        elif mode == "sync_batch":
            return {
                1: torch.nn.SyncBatchNorm,
                2: torch.nn.SyncBatchNorm,
                3: torch.nn.SyncBatchNorm,
            }[ndims](num_channels, eps=eps, momentum=momentum, affine=affine)

        elif mode == "instance":
            return {
                1: torch.nn.InstanceNorm1d,
                2: torch.nn.InstanceNorm2d,
                3: torch.nn.InstanceNorm3d,
            }[ndims](num_channels, eps=eps, momentum=momentum, affine=affine)

        elif mode == "layer":
            return torch.nn.LayerNorm(input_shape, eps=eps, elementwise_affine=affine)

        elif mode == "group":
            return torch.nn.GroupNorm(
                num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine
            )

        else:
            raise ValueError(f"Unsupported normalization mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, *), where * represents
            spatial dimensions matching the input shape.

        Returns
        -------
        torch.Tensor
            Normalized tensor with same shape as input.
        """
        return self.layer(x)

    def __repr__(self) -> str:
        """
        Return a string representation of the normalization layer.

        The representation includes mode, input channel count,
        number of spatial dimensions, and whether spectral norm
        is applied.

        Returns
        -------
        str
            String representation of the layer.
        """
        spectral = (
            " + spectral" if isinstance(self.layer, torch.nn.utils.spectral_norm.__class__) else ""
        )
        channels = self.get_input_shape(input_id=0)
        ndims = self.get_ndims()
        return f"NormLayer(mode={self.mode}, in_channels={channels}, ndim={ndims}){spectral}"
