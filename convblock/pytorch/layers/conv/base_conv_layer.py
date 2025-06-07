"""Contains implementation of base convolutional layer."""

from __future__ import annotations

import torch

from ...initializer import IInitializer, InitializerConfigDict, InitializerFactory
from ...utils import ArrayLike
from ..conv_torch_module import ConvTorchModule


class BaseConvLayer(ConvTorchModule):
    """Base class for convolutional layers with configurable initialization.

    This class generalizes standard PyTorch convolutional layers by
    providing explicit support for:

    - Input shape inference and output shape tracking
    - Multi-dimensional kernel/stride/dilation configuration
    - Optional grouped convolution
    - Custom weight and bias initialization via structured dictionaries
    - Padding logic inherited from `ConvModule` base

    Subclasses (e.g., Conv1D, Conv2D, Conv3D) can extend this class
    by implementing the appropriate forward logic or kernel behavior.

    Attributes
    ----------
    filters : int
        Number of output channels produced by the convolution.
    groups : int
        Number of groups for grouped convolution.
    weight : torch.nn.Parameter
        Learnable weight tensor initialized using `init_weight`.
    bias : torch.nn.Parameter or None
        Learnable bias tensor initialized using `init_bias`, or None if `bias=False`.
    input_shape : tuple[int]
        Shape of the input tensor excluding batch dimension.
    output_shape : tuple[int]
        Shape of the output tensor inferred during construction.
    kernel_size : tuple[int]
        Size of the convolutional kernel.
    stride : tuple[int]
        Stride applied to the convolution.
    dilation : tuple[int]
        Dilation applied to the convolution.
    """

    def __init__(
        self,
        input_shape: ArrayLike[int],
        output_shape: ArrayLike[int],
        filters: int,
        kernel_size: ArrayLike[int] | int = 3,
        stride: ArrayLike[int] | int = 1,
        dilation: ArrayLike[int] | int = 1,
        groups: int = 1,
        bias: bool = False,
        init_weight: InitializerConfigDict | None = None,
        init_bias: InitializerConfigDict | None = None,
    ):
        """Initialize base class for convolutional layers.

        This layer extends functionality of standard PyTorch Conv modules by
        supporting:
        - shape-aware configuration (`input_shape`)
        - generalized dimension handling (1D, 2D, 3D)
        - custom initialization logic via structured config dictionaries

        Parameters
        ----------
        input_shape : ArrayLike[int]
            Shape of the input tensor, excluding the batch dimension.
            Must be of format (channels, spatial_dim1, spatial_dim2, ...).
        output_shape : ArrayLike[int]
            Shape of the output tensor, excluding batch dimension.
            Must be of format (channels, spatial_dim1, spatial_dim2, ...)
        filters : int
            Number of output channels for the convolution operation.
        kernel_size : int or ArrayLike[int], default=3
            Size of the convolutional kernel.
        stride : int or ArrayLike[int], default=1
            Stride of the convolution.
        dilation : int or ArrayLike[int], default=1
            Dilation rate of the convolution.
        groups : int, default=1
            Number of blocked connections from input to output channels.
        bias : bool, default=False
            Whether to include a learnable bias term.
        init_weight : InitializerConfigDict or None, optional
            Dictionary specifying the initializer for weights.
            Must match one of the supported initializer types registered
            in the InitializerFactory. If None, defaults to Xavier normal.
        init_bias : InitializerConfigDict or None, optional
            Dictionary specifying the initializer for biases.
            Must match one of the supported initializer types registered
            in the InitializerFactory. If None, defaults to Xavier normal.

        Notes
        -----
        Weight and bias are created as torch.nn.Parameter with values initialized
        by calling the selected initializer's `.initialize()` method.
        """
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.filters = int(filters)
        self.groups = int(groups)

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
        bias = torch.rand(self.filters)
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
            (filters, in_channels // groups, *kernel_size).
        """
        weight = torch.Tensor(self.filters, self.in_channels // self.groups, *self.kernel_size)
        if initializer is not None:
            weight = initializer.initialize(weight)
        return torch.nn.Parameter(weight)
