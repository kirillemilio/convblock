"""Contains implementation of SPADE block using functional conv2d."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from ..conv import Conv
from ..modulated_torch_module import ModulatedTorchModule


class Spade(ModulatedTorchModule):
    """
    Spatially-Adaptive Denormalization (SPADE) block using functional 1x1 convolutions.

    SPADE performs per-location modulation of normalized feature maps using
    a condition (style) tensor. This modulation happens after applying group
    or instance normalization. The scale (γ) and shift (β)
    parameters are learned via functional 1×1 convolutions over the spatial conditioning tensor.

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Shape of the input tensor, usually (C, H, W).
    cond_channels : int
        Number of channels in the conditioning (style) tensor.
    affine_groups : int, optional
        Number of groups for γ and β projection convolutions. Default is 1.
    groups : int or None, optional
        Number of groups for group normalization. If None, defaults to input channels.
    init_weight : InitializerConfigDict or None, optional
        Initializer for convolutional weights (for γ and β).
    init_bias : InitializerConfigDict or None, optional
        Initializer for convolutional biases (if bias is used).
    use_bias : bool, default=True
        Whether to include bias in γ and β convolutions.
    use_affine_scale : bool, default=True
        Whether to learn a scale (γ) modulation.
    use_affine_shift : bool, default=True
        Whether to learn a shift (β) modulation.
    norm : bool, default=True
        Whether to apply normalization before modulation (group or instance).

    Examples
    --------
    >>> # Full SPADE block with instance norm, γ and β modulations
    Spade(input_shape=[64, 128, 128], cond_channels=32)

    >>> # SPADE block with group norm, only γ modulation, no bias
    Spade(input_shape=[64, 64, 64], cond_channels=16, groups=8,
          use_affine_shift=False, use_bias=False)

    >>> # Mask-based multiplicative attention (no norm, γ only)
    Spade(input_shape=[32, 64, 64], cond_channels=1, norm=False, use_affine_shift=False)

    >>> # Pure spatial shift modulation (β only), useful for residual conditioning
    Spade(input_shape=[128, 32, 32], cond_channels=8, use_affine_scale=False)
    """

    cond_channels: int
    out_channels: int
    affine_groups: int
    groups: int
    norm_type: str

    beta_conv: Conv | None
    gamma_conv: Conv | None

    def __init__(
        self,
        input_shape: ArrayLike[int],
        cond_channels: int,
        affine_groups: int = 1,
        groups: int | None = None,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
        use_bias: bool = True,
        use_affine_scale: bool = True,
        use_affine_shift: bool = True,
        norm: bool = True,
    ) -> None:
        """
        Initialize SPADE block.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Input tensor shape in CHW format (e.g., [C, H, W]).
        cond_channels : int
            Number of channels in the style/conditioning tensor.
        affine_groups : int, default=1
            Number of groups for modulation convolutions (γ and β).
        groups : int or None, optional
            Number of groups used in group normalization. If None and norm=True,
            instance normalization will be applied instead.
        init_weight : InitializerConfigDict or None, optional
            Weight initializer configuration for modulation convolutions.
        init_bias : InitializerConfigDict or None, optional
            Bias initializer configuration for modulation convolutions.
        use_bias : bool, default=True
            Whether γ and β convolutions include a bias term.
        use_affine_scale : bool, default=True
            Whether to apply multiplicative scale (γ).
        use_affine_shift : bool, default=True
            Whether to apply additive shift (β).
        norm : bool, default=True
            Whether to normalize input tensor before modulation.
        """
        # initializing base class
        ndims = len(input_shape) - 1
        modulation_shape = [cond_channels] + [1] * ndims
        super().__init__(input_shape=input_shape, modulation_shape=modulation_shape)

        self.out_channels = input_shape[0]
        self.cond_channels = cond_channels
        self.groups = groups if groups is not None else input_shape[0]
        self.affine_groups = affine_groups

        self.norm_type = "none"
        if self.groups is not None and norm:
            self.norm_type = "group"
        elif norm:
            self.norm_type = "instance"

        if groups is not None and groups <= 0:
            raise ValueError("Groups parameter must be greater than zero")
        elif groups is not None and self.out_channels % groups != 0:
            raise ValueError(f"Channels ({self.out_channels}) must be divisible by groups={groups}")

        self.beta_conv = None
        self.gamma_conv = None

        if use_affine_shift:
            self.beta_conv = Conv(
                input_shape=modulation_shape,
                filters=self.out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                groups=affine_groups,
                bias=use_bias,
                init_bias=init_bias,
                init_weight=init_weight,
            )
        if use_affine_scale:
            self.gamma_conv = Conv(
                input_shape=modulation_shape,
                filters=self.out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                groups=affine_groups,
                bias=use_bias,
                init_bias=init_bias,
                init_weight=init_weight,
            )

    @property
    def use_affine_shift(self) -> bool:
        """Whether block uses affine shift.

        Returns
        -------
        bool
            whether model uses affine shift.
        """
        return self.beta_conv is not None

    @property
    def use_affine_scale(self) -> bool:
        """Whether block uses affine scale.

        Returns
        -------
        bool
            whether model uses affine scale.
        """
        return self.gamma_conv is not None

    def forward(self, x: torch.Tensor, style: torch.Tensor, *other: torch.Tensor) -> torch.Tensor:
        """
        Apply SPADE modulation to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        style : torch.Tensor
            Condition tensor of shape (B, cond_channels, H, W).

        Returns
        -------
        torch.Tensor
            Modulated tensor of shape (B, C, H, W).
        """
        if self.norm_type == "group":
            normed = F.group_norm(x, num_groups=self.groups)
        elif self.norm_type == "instance":
            normed = F.instance_norm(x)
        else:
            normed = x

        res = normed
        if self.gamma_conv is not None:
            res = res.mul(self.gamma_conv.forward(style).add(1.0))
        if self.beta_conv is not None:
            res = res.add(self.beta_conv.forward(style))
        return res

    def __repr__(self) -> str:
        """Get string representation of spade layer.

        Returns
        -------
        str
            string representation of spade layer.
        """
        return (
            f"Spade(norm='{self.norm_type}', cond={self.cond_channels}, out={self.out_channels}, "
            f"grp={self.groups}, aff_grp={self.affine_groups}, "
            f"γ={'✓' if self.use_affine_scale else '✗'}"
            f"({self.gamma_conv.bias is not None if self.use_affine_scale else '-'}) "
            f"β={'✓' if self.use_affine_shift else '✗'}"
            f"({self.beta_conv.bias is not None if self.use_affine_shift else '-'})"
            ")"
        )
