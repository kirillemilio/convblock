import math
import numpy as np
import torch
import torch.nn.functional as F

from ..layers import ConvBlock
from ..bases import Module, Sequential
from .res_block import VanillaResBlock


class BaseEncoder(Module):
    """ Base class for all encoders. """

    def __init__(self, input_shape):
        super().__init__(input_shape)

    @property
    def output_shape(self):
        """ Get shape of the output tensor.

        Returns
        -------
        NDArray[int]
            shape of the output tensor.
        """
        return self.layers.output_shape

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for VNet encoder.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            output tensor of decoder block.
        """
        return self.layers.forward(inputs)


class VanillaUNetEncoder(BaseEncoder):

    def __init__(self, input_shape, filters, layout='cna cna', kernel_size=3,
                 downsample=True, downsampling_kernel=2, block=None):
        """ Build UNet encoder block.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            shape of input tensor. Batch dimension is not taken into account.
        filters : ArrayLike[int] or int
            filters for convolutions in ConvBlock.
        layout : str
            layout of ConvBlock or partially applied ConvBlock.
        kernel_size : ArrayLike[int] or int
            kernel_size required by convolutions of ConvBlock.
            Default is 3 meaning that all convolutions inside block along
            each spatial dimension will have kernel_size equal to 3.
        downsample : bool
            whether downsampling via max pooling is required. If set True
            then max pooling with kernel_size 'downsampling_kernel' and
            stride 2 is applied before all convolutions.
        downsampling_kernel : ArrayLike[int] or int
            kernel size of max pooling operation.
        block : ConvBlock, partially applied ConvBlock or None.
            this argument can be used for passing non-default
            global parameters to ConvBlock.
        """
        super().__init__(input_shape)
        layout = 'p' + layout if downsample else layout
        block = ConvBlock if block is None else block
        self.layers = block(
            input_shape=input_shape, layout=layout,
            c=dict(kernel_size=kernel_size, filters=filters),
            p=dict(kernel_size=downsampling_kernel, stride=2, mode='max')
        )


class VanillaVNetEncoder(BaseEncoder):

    def __init__(self, input_shape, filters, layout, kernel_size=5,
                 downsample=True, downsampling_kernel=2,
                 post_activation=False, block=None):
        """ Build VNet encoder block.

        Parameters
        ----------
        input_shape : ArrayLike[int]
            shape of input tensor. Batch dimension is not taken into account.
        filters : ArrayLike[int] or int
            filters for convolutions in ConvBlock.
        layout : str
            layout of ConvBlock or partially applied ConvBlock.
        kernel_size : ArrayLike[int] or int
            kernel_size required by convolutions of ConvBlock.
            Default is 3 meaning that all convolutions inside block along
            each spatial dimension will have kernel_size equal to 3.
        downsample : bool
            whether downsampling via max pooling is required. If set True
            then max pooling with kernel_size 'downsampling_kernel' and
            stride 2 is applied before all convolutions.
        downsampling_kernel : ArrayLike[int] or int
            kernel size of convolution operation used for downsampling.
        post_activation : bool
            whether to use post activation with normalization
            in VanillaResBlock. Default is False.
        block : ConvBlock, partially applied ConvBlock or None.
            this argument can be used for passing non-default
            global parameters to ConvBlock.
        """
        super().__init__(input_shape)
        block = ConvBlock if block is None else block

        if downsample:
            layers = Sequential()
            downsample_layer = block(
                input_shape=input_shape, layout='cna',
                c=dict(kernel_size=downsampling_kernel,
                       stride=2, filters=filters)
            )

            res_block = VanillaResBlock(
                input_shape=downsample_layer.output_shape,
                filters=filters, layout=layout,
                kernel_size=kernel_size, block=block,
                post_activation=post_activation
            )

            layers.add_module('Downsample', downsample_layer)
            layers.add_module('ResBlock', res_block)

        else:
            layers = VanillaResBlock(
                input_shape=input_shape,
                filters=filters, layout=layout,
                kernel_size=kernel_size, block=block,
                post_activation=post_activation
            )
        self.layers = layers
