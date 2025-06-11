"""Contains implementation of AdaIN block."""

from __future__ import annotations

from typing import Literal

import torch

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from ..conv import Conv
from ..modulated_torch_module import ModulatedTorchModule


class AdaIn(ModulatedTorchModule):
    """
    Adaptive Instance Normalization (AdaIN) block with optional grouping.

    This block normalizes input feature maps using instance or group normalization,
    and then applies spatially-invariant affine modulation using a style vector.
    The modulation consists of learned per-group scale (γ) and shift (β), projected
    from the style input using functional 1x1 convolutions (no nn.Linear).

    Can emulate:
    - Standard AdaIN (use groups=None or groups=C and mod_dim=D)
    - FiLM (use norm=False)
    - AdaGroupNorm (use groups > 1)
    - Multiplicative attention masks (use_affine_shift=False)

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Shape of the input tensor, typically (C, H, W).
    mod_dim : int
        Dimensionality of the style vector.
    groups : int, optional
        Number of groups for normalization and modulation. Defaults to input channels.
    init_weight : InitializerConfigDict, optional
        Initializer configuration for projection weights.
    init_bias : InitializerConfigDict, optional
        Initializer configuration for projection biases.
    use_bias : bool, default=False
        Whether to use bias in projection convolutions.
    use_affine_scale : bool, default=True
        Whether to apply learned scaling (γ).
    use_affine_shift : bool, default=True
        Whether to apply learned bias (β).
    norm : bool, default=True
        Whether to normalize input before modulation.

    Examples
    --------
    >>> # Classical AdaIN: instance norm + learned affine scale and shift
    AdaIn(input_shape=[64, 32, 32], mod_dim=512)

    >>> # AdaGroupNorm: group norm with grouped scale/shift
    AdaIn(input_shape=[64, 32, 32], mod_dim=128, groups=8)

    >>> # Multiplicative attention without shift, no norm
    AdaIn(input_shape=[32, 16, 16], mod_dim=32, norm=False, use_affine_shift=False)

    >>> # FiLM block: no normalization, dense γ and β from style
    AdaIn(input_shape=[128, 64, 64], mod_dim=256, norm=False)
    """

    mod_dim: int
    groups: int
    group_size: int

    norm_type: Literal["group", "instance", "none"]

    mod_conv: Conv | None

    def __init__(
        self,
        input_shape: ArrayLike[int],
        mod_dim: int,
        groups: int | None = None,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
        use_bias: bool = False,
        use_affine_scale: bool = True,
        use_affine_shift: bool = True,
        norm: bool = True,
    ) -> None:
        """
        Initialize an AdaIN-style normalization and modulation block.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            The input shape of the tensor to be modulated (e.g. [C, H, W]).
        mod_dim : int
            The number of channels in the style input vector.
        groups : int, optional
            Number of modulation groups. If None, defaults to the number of input channels.
            If provided, must divide input channels evenly.
        init_weight : InitializerConfigDict, optional
            Initializer for weights in γ and β projection layers.
        init_bias : InitializerConfigDict, optional
            Initializer for biases in γ and β projection layers.
        use_bias : bool, default=False
            Whether to include bias in the γ/β convolutions.
        use_affine_scale : bool, default=True
            Whether to include learnable scale (γ).
        use_affine_shift : bool, default=True
            Whether to include learnable shift (β).
        norm : bool, default=True
            Whether to apply normalization before modulation. If False, acts as FiLM.
        """
        ndims = len(input_shape) - 1
        modulation_shape = [mod_dim] + [1] * ndims
        super().__init__(input_shape=input_shape, modulation_shape=modulation_shape)

        self.norm_type = "none"
        if self.groups is not None and norm:
            self.norm_type = "group"
        elif norm:
            self.norm_type = "instance"

        if groups is not None and groups <= 0:
            raise ValueError("Groups parameter must greater than zero")
        elif groups is not None and self.input_shape[0] % groups != 0:
            raise ValueError(
                f"Number of channels {self.input_shape[0]} must be divisible by groups={groups}"
            )
        self.mod_dim = mod_dim
        self.groups = groups if groups is not None else self.input_shape[0]
        self.group_size = self.input_shape[0] // self.groups

        self.beta_conv = None
        self.gamma_conv = None

        if use_affine_shift:
            self.beta_conv = Conv(
                input_shape=[mod_dim] + [1] * ndims,
                filters=groups,
                kernel_size=1,
                stride=1,
                dilation=1,
                groups=1,
                use_bias=use_bias,
                init_bias=init_bias,
                init_weight=init_weight,
            )

        if use_affine_scale:
            self.gamma_conv = Conv(
                input_shape=[mod_dim] + [1] * ndims,
                filters=self.groups,
                kernel_size=1,
                stride=1,
                dilation=1,
                groups=1,
                bias=use_bias,
                init_bias=init_bias,
                init_weight=init_weight,
            )

    def forward(
        self, inputs: torch.Tensor, style: torch.Tensor, *other: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply AdaIN modulation to the input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, H, W, ...).
        style : torch.Tensor
            Style tensor of shape (B, mod_dim) or (B, mod_dim, 1, 1, ...).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C, H, W, ...) after normalization and modulation.
        """
        batch_size = style.size(0)
        ndims = self.get_ndims()

        if len(style.shape) == 2:
            shape = [batch_size, style.size(1)] + [1] * ndims
            style = style.view(*shape)
        modulation, *_ = self.mod_conv(style)

        normed = inputs
        if self.norm_type == "group":
            repeats = [1, 1, self.group_size] + [1] * self.get_ndims()
            shape = [batch_size, -1] + [1] * ndims
            modulation = modulation.unsqueeze(2).repeat(*repeats).view(*shape)
            normed = torch.nn.functional.group_norm(input=inputs, num_groups=self.groups)
            if self.gamma_conv is not None:
                normed = normed.mul(
                    self.gamma_conv.forward(style)
                    .unsqueeze(2)
                    .repeat(*repeats)
                    .view(*shape)
                    .add(1.0)
                )
            if self.beta_conv is not None:
                normed = normed.add(
                    self.beta_conv.forward(style).unsqueeze(2).repeat(*repeats).view(*shape)
                )

        elif self.norm_type == "instance":
            normed = torch.nn.functional.instance_norm(inputs)
            if self.gamma_conv is not None:
                normed = normed.mul(self.gamma_conv.forward(style).add(1.0))
            if self.beta_conv is not None:
                normed = normed.add(self.beta_conv.forward(style))

        return normed

    def __repr__(self) -> str:
        """Get string representation of the AdaIN block.

        Returns
        -------
        str
            Human-readable description of the block’s configuration.
        """
        return (
            f"AdaIn(norm='{self.norm_type}', mod_dim={self.mod_dim}, groups={self.groups}, "
            f"γ={'✓' if self.gamma_conv is not None else '✗'}, "
            f"β={'✓' if self.beta_conv is not None else '✗'})"
        )
