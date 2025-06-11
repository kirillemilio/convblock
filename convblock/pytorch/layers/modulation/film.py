"""Contains implementation of FiLM modulation layer."""

from __future__ import annotations

import torch

from ...initializer import InitializerConfigDict
from ...utils import ArrayLike
from .adain import AdaIn


class FiLM(AdaIn):
    """
    Feature-wise Linear Modulation (FiLM) block.

    Applies learned affine transformation (scale and/or shift)
    to the input tensor, conditioned on an external style tensor.
    Unlike AdaIN, FiLM does **not** perform normalization
    and simply modulates the features directly.

    FiLM can be seen as a special case of AdaIN where normalization is disabled,
    making it suitable for injecting contextual
    or sequential information via gating or dynamic control.

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Shape of the input tensor (C, H, W).
    mod_dim : int
        Dimensionality of the style (modulation) tensor.
    groups : int or None, optional
        Number of modulation groups. Must divide number of channels.
    init_weight : InitializerConfigDict or None, optional
        Initializer configuration for projection weights.
    init_bias : InitializerConfigDict or None, optional
        Initializer configuration for projection biases.
    use_bias : bool, default=False
        Whether to use bias in modulation projection layers.
    use_affine_scale : bool, default=True
        Whether to use affine scale (γ) in modulation.
    use_affine_shift : bool, default=True
        Whether to use affine shift (β) in modulation.

    Examples
    --------
    >>> film = FiLM(input_shape=[64, 32, 32], mod_dim=512)
    >>> out = film(x, style)
    """

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
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            mod_dim=mod_dim,
            groups=groups,
            init_bias=init_bias,
            init_weight=init_weight,
            use_bias=use_bias,
            use_affine_scale=use_affine_scale,
            use_affine_shift=use_affine_shift,
            norm=False,
        )

    def forward(self, inputs: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, H, W).
        style : torch.Tensor
            Style tensor of shape (B, mod_dim).

        Returns
        -------
        torch.Tensor
            Modulated tensor after applying FiLM transformation.
        """
        return super().forward(inputs, style)

    def __repr__(self) -> str:
        """
        Return a string representation of the FiLM layer.

        Returns
        -------
        str
            Human-readable configuration summary.
        """
        return (
            f"FiLM(mod_dim={self.mod_dim}, groups={self.groups}, "
            f"scale={'✓' if self.gamma_conv is not None else '✗'}, "
            f"shift={'✓' if self.beta_conv is not None else '✗'}, "
            f"bias={self.gamma_conv.bias is not None if self.gamma_conv else '-'})"
        )
