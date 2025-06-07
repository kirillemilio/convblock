"""Contains implementation of AdaIN block."""

from __future__ import annotations

import torch

from ...bases import Module
from ...initializer import IInitializer, InitializerConfigDict, InitializerFactory
from ...utils import ArrayLike


class AdaIn(Module):
    """
    Adaptive Instance Normalization (AdaIN) block.

    This module applies instance normalization to the input tensor and then
    modulates the normalized output using a style vector. The modulation
    parameters (scale and shift) are learned from the style input using a
    linear projection.

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Shape of the input tensor (C, H, W) or similar.
    mod_dim : int
        Dimension of the style vector.
    groups : int, default=1
        Number of channel groups for grouped modulation. Channels must be
        divisible by this value.
    init_weight : InitializerConfigDict or None, optional
        Configuration for weight initializer. If None, uses Xavier.
    init_bias : InitializerConfigDict or None, optional
        Configuration for bias initializer. If None, uses Xavier.
    bias : bool, default=False
        Whether to use bias in the modulation projection.
    """

    fc_weight: torch.nn.Parameter
    fc_bias: torch.nn.Parameter | None
    mod_dim: int
    groups: int | None
    group_size: int

    def __init__(
        self,
        input_shape: ArrayLike[int],
        mod_dim: int,
        groups: int | None = None,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__(input_shape=input_shape)
        if groups is not None and self.input_shape[0] % groups != 0:
            raise ValueError(
                f"Number of channels {self.input_shape[0]} must be divisible by groups={groups}"
            )
        self.mod_dim = mod_dim
        self.groups = groups
        self.group_size = 1 if groups is None else self.input_shape[0] // groups

        weight_initializer = InitializerFactory.create_initializer(
            init_weight or {"init_type": "nxavier"}
        )
        bias_initializer = InitializerFactory.create_initializer(
            init_bias or {"init_type": "nxavier"}
        )

        self.fc_weight = self._create_weight(initializer=weight_initializer)
        self.fc_bias = self._create_bias(initializer=bias_initializer) if bias else None

    def _create_bias(self, initializer: IInitializer | None = None) -> torch.nn.Parameter:
        """Create and initialize the bias parameter tensor.

        Parameters
        ----------
        initializer : IInitializer or None, optional
            Initializer for the bias tensor. If None, leaves tensor unchanged.

        Returns
        -------
        torch.nn.Parameter
            Initialized bias tensor.
        """
        groups = self.input_shape[0] if self.groups is None else self.groups
        bias = torch.rand(groups * 2)
        return torch.nn.Parameter(initializer.initialize(bias) if initializer else bias)

    def _create_weight(self, initializer: IInitializer | None = None) -> torch.nn.Parameter:
        """Create and initialize the weight parameter tensor.

        Parameters
        ----------
        initializer : IInitializer or None, optional
            Initializer for the weight tensor.

        Returns
        -------
        torch.nn.Parameter
            Initialized weight tensor of shape (2 * C // groups, mod_dim).
        """
        groups = self.input_shape[0] if self.groups is None else self.groups
        weight = torch.empty(groups * 2, self.mod_dim)
        return torch.nn.Parameter(initializer.initialize(weight) if initializer else weight)

    def forward(self, inputs: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply adaptive instance normalization to input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, C, H, W, ...).
        style : torch.Tensor
            Style tensor of shape (B, mod_dim).

        Returns
        -------
        torch.Tensor
            Output tensor after AdaIN modulation.
        """
        batch_size = style.size(0)
        modulation = torch.nn.functional.linear(style, self.fc_weight, self.fc_bias)

        if self.groups is not None:
            repeats = [1, 1, self.group_size] + [1] * self.ndims
            shape = [batch_size, -1] + [1] * self.ndims
            modulation = modulation.unsqueeze(2).repeat(*repeats).view(*shape)
            gamma, beta, *_ = torch.chunk(modulation, chunks=2, dim=1)
            normed = torch.nn.functional.group_norm(input=inputs, num_groups=self.groups)
        else:
            gamma, beta, *_ = torch.chunk(modulation, chunks=2, dim=1)
            normed = torch.nn.functional.instance_norm(inputs)
        return (1 + gamma) * normed + beta

    def __repr__(self):
        """Get string representation of adain module.

        Returns
        -------
        str
            string representation of adain module.
        """
        return f"AdaIn(mod_dim={self.mod_dim}, groups={self.groups})"
