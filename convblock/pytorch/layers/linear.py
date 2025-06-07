"""Contains implementation of linear layer."""

from __future__ import annotations

import numpy as np
import torch

from ..initializer import IInitializer, InitializerConfigDict, InitializerFactory
from ..utils import ArrayLike
from .base_module import BaseModule
from .conv_block import ConvBlock


@ConvBlock.register_option("f")
class LinearLayer(BaseModule):
    """
    Implement a fully connected (linear) layer with custom initialization.

    This layer performs a linear transformation of the input using a
    weight matrix and optional bias. It uses functional implementation
    via `torch.nn.functional.linear`.

    Parameters
    ----------
    input_shape : ArrayLike[int]
        Input shape of the tensor. Should be 1D, e.g., (in_features,).
    out_features : int
        Number of output features for the linear layer.
    bias : bool, default=True
        Whether to use a bias term in the linear transformation.
    init_weight : InitializerConfigDict or None, optional
        Weight initializer configuration. Uses Xavier by default.
    init_bias : InitializerConfigDict or None, optional
        Bias initializer configuration. Uses Xavier by default.
    """

    def __init__(
        self,
        input_shape: ArrayLike[int],
        out_features: int,
        bias: bool = True,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ) -> None:
        input_shape = np.array(input_shape, dtype=np.int64)
        output_shape = np.array([out_features, *input_shape[1:]], dtype=np.int64)
        super().__init__(
            input_shape=input_shape[np.newaxis, ...], output_shape=output_shape[np.newaxis, ...]
        )
        self.out_features = out_features
        self.in_features = int(self.input_shape[0, 0])

        weight_initalizer = InitializerFactory.create_initializer({"init_type": "nxavier"})
        if init_weight is not None:
            weight_initalizer = InitializerFactory.create_initializer(init_weight)

        bias_initializer = InitializerFactory.create_initializer({"init_type": "nxavier"})
        if init_bias is not None:
            bias_initializer = InitializerFactory.create_initializer(init_bias)

        self.weight = self._create_weight(initializer=weight_initalizer)
        if bias:
            self.bias = self._create_bias(initializer=bias_initializer)
        else:
            self.register_parameter("bias", None)

    def _create_bias(self, initializer: IInitializer | None = None) -> torch.nn.Parameter:
        """Create and initialize bias tensor as torch.nn.Parameter.

        Parameters
        ----------
        initializer : IInitializer or None, optional
            Initializer instance used to initialize the bias values.
            If None, the tensor will remain uninitialized.

        Returns
        -------
        torch.nn.Parameter
            Bias parameter tensor ready for optimization.
        """
        bias = torch.rand(self.out_channels)
        if initializer is not None:
            bias = initializer.initialize(bias)
        return torch.nn.Parameter(bias)

    def _create_weight(self, initializer: IInitializer | None = None) -> torch.nn.Parameter:
        """Create and initialize weight tensor as torch.nn.Parameter.

        Parameters
        ----------
        initializer : IInitializer or None, optional
            Initializer instance used to initialize the weight values.

        Returns
        -------
        torch.nn.Parameter
            Weight parameter tensor of shape
            (out_channels, in_channels)
        """
        weight = torch.Tensor(self.out_channels, self.in_channels)
        if initializer is not None:
            weight = initializer.initialize(weight)
        return torch.nn.Parameter(weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to the input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (B, in_features), where B is batch size.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_features).
        """
        return torch.nn.functional.linear(input=inputs, weight=self.weight, bias=self.bias)

    def __repr__(self) -> str:
        """
        Return string representation of the linear layer.

        Returns
        -------
        str
            Readable summary of the layer configuration.
        """
        s = (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
        )

        if self.bias is None:
            s += ", bias=False"
        else:
            s += ", bias=True"
        s += ")"
        return s
