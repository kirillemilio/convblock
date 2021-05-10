""" Contains implementation of fire block of SqueezeNet model. """

import numpy as np
import torch
from ..layers import ConvBlock
from ..bases import Module, MetaModule, Sequential
from .res_block import BaseResBlock


class FireBlock(Module, metaclass=MetaModule):

    def __init__(self, input_shape, s1x1, e1x1, e3x3, layout='cna',
                 pool_kernel=3, pool_mode='max',
                 downsample=False, block=None, how='.'):
        """ Build simple fire block used in SqueezeNet architecture.

        For more details see http://arxiv.org/abs/1602.07360.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not taken
            into account.
        s1x1 : int
            number of output channels of squeezing 1x1 convolution.
        e1x1 : int
            number of output channels of expanding 1x1 convolution.
        e3x3 : int
            number of output channels of expanding 3x3 convolution.
        layout : str
            layout that will be used for squeezing 1x1 convolution,
            expanding 1x1 convolution, expanding 3x3 convolution.
            Default is 'cna'.
        pool_kernel : str
            size of pooling kernel when downsample is 'True'. Default is 3.
        pool_mode : str
            mode of pooling when downsample is 'True'. Default is 'max'.
        downsample : str
            whether downsampling is required in the begining of the block.
            Downsampling is performed via pooling operation with stride 2.
            Default is 'False'.
        block : partially applied ConvBlock or None
            base block for all conv opeartions used under the hood.
        how : str
            merge method mode. Default is '+'.
        """
        super().__init__(input_shape)
        block = ConvBlock if block is None else block

        s1x1, e1x1, e3x3 = int(s1x1), int(e1x1), int(e3x3)
        self.squeeze_layer = block(
            input_shape=input_shape,
            layout='p' + layout if downsample else layout,
            c=dict(kernel_size=1, filters=s1x1),
            p=dict(kernel_size=pool_kernel, mode=pool_mode, stride=2)
        )
        self.fire_x1 = block(
            input_shape=self.squeeze_layer.output_shape, layout=layout,
            c=dict(kernel_size=1, filters=e1x1)
        )
        self.fire_x3 = block(
            input_shape=self.squeeze_layer.output_shape, layout=layout,
            c=dict(kernel_size=3, filters=e3x3)
        )

        self.downsample = downsample
        self.how = how

    @property
    def output_shape(self) -> 'Tensor':
        """ Get shape of the output tensor. """
        if self.how == '.':
            filters = (self.fire_x1.output_shape[0]
                       + self.fire_x3.output_shape[0])
            spatial_dims = tuple(self.fire_x1.output_shape[1:])
            return np.array([filters, *spatial_dims], dtype=np.int)
        else:
            return self.fire_x1.output_shape

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for FireModule of SqueezeNet model. """
        x = self.squeeze_layer(inputs)
        return self.merge(self.fire_x1(x), self.fire_x3(x), how=self.how)


class FireBlockWithBypass(Module, metaclass=MetaModule):

    def __new__(cls, input_shape, s1x1, e1x1, e3x3, layout='cna',
                pool_kernel=3, pool_mode='max', downsample=False,
                post_activation=False, block=None, how='+'):
        """ Build fire block with bypass used in SqueezeNet architecture.

        For more details see http://arxiv.org/abs/1602.07360.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not taken
            into account.
        s1x1 : int
            number of output channels of squeezing 1x1 convolution.
        e1x1 : int
            number of output channels of expanding 1x1 convolution.
        e3x3 : int
            number of output channels of expanding 3x3 convolution.
        layout : str
            layout that will be used for squeezing 1x1 convolution,
            expanding 1x1 convolution, expanding 3x3 convolution.
            Default is 'cna'.
        pool_kernel : str
            size of pooling kernel when downsample is 'True'. Default is 3.
        pool_mode : str
            mode of pooling when downsample is 'True'. Default is 'max'.
        downsample : str
            whether downsampling is required in the begining of the block.
            Downsampling is performed via pooling operation with stride 2.
            Default is 'False'.
        block : partially applied ConvBlock or None
            base block for all conv opeartions used under the hood.
        how : str
            merge method mode. Default is '+'.
        """
        s1x1, e1x1, e3x3 = int(s1x1), int(e1x1), int(e3x3)
        input_shape = np.array(input_shape, dtype=np.int)
        block = ConvBlock if block is None else block
        body = FireBlock.partial(
            s1x1=s1x1, e1x1=e1x1, e3x3=e3x3, layout=layout,
            downsample=False, block=block, how=how
        )
        if np.all(body.output_shape != input_shape):
            shortcut = block.partial(
                layout='cn',
                c=dict(kernel_size=1,
                       filters=e1x1 + e3x3 if how == '.' else e1x1,
                       stride=1)
            )
        else:
            shortcut = None

        if post_activation:
            head = block.partial(layout='an')
        else:
            head = None

        if downsample:
            pool_layer = block(
                input_shape=input_shape, layout='p',
                p=dict(kernel_size=pool_kernel, mode=pool_mode, stride=2)
            )

            resblock = BaseResBlock(pool_layer.output_shape,
                                    body, shortcut, head, '+')

            return Sequential(pool_layer, resblock)
        else:
            return BaseResBlock(input_shape, body, shortcut, head, '+')
