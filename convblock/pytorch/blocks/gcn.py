""" Contains implementation of GlobalConvolutional blocks. """

import numpy as np
import torch
from ..utils import INT_TYPES, transform_to_int_tuple
from ..layers import ConvBlock
from ..bases import MetaModule, Module


class GCNBlock(Module, metaclass=MetaModule):

    def __init__(self, input_shape, filters, kernel_size=11,
                 layout='cc', how='+', block=None):
        """ Build Global Convolutional Block.

        Implementation of global convolutional block from
        http://arxiv.org/abs/1703.02719.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        filters : int
            number of output channels for all convolutional operations.
        kernel_size: int
            size of kernel. Note that it must be int value, because
            1xk and kx1 convolutions are used under the hood.
        layout : str
            layout that is common for two branches. Must contain strictly two
            convolutional operation. Default is 'cc', but can be changed to
            'cna cna', 'cn cn' and etc.
        how : str
            the way to merge two branches. Can be '+', '.', '*'.
            Default is '+'.
        block : ConvBlock or None
            partially applied ConvBlock or None. Default is None meaning that
            default ConvBlock will be used.
        """
        if len(input_shape) != 3:
            raise ValueError("Length of 'input_shape' must 3."
                             + " Got {}.".format(len(input_shape)))

        super().__init__(input_shape)

        if how not in ('+', '*', '.'):
            raise ValueError("Argument 'how' must be one of following values: "
                             + "'+', '*', '.'. Got {}.".format(how))

        if not isinstance(kernel_size, INT_TYPES):
            raise TypeError("Argument 'kernel_size' must have "
                            + "int type. Got {}.".format(type(kernel_size)))

        if layout.lower().count('c') != 2:
            raise ValueError("Argument 'layout' must contain exactly two "
                             + "convolutional opeations. Got {}.".format(layout))

        kernel_size, filters = int(kernel_size), int(filters)
        block = ConvBlock if block is None else block

        self.how = how
        self.filters = filters
        self.branch_1xkxkx1 = block(
            input_shape=self.input_shape, layout=layout,
            c=dict(kernel_size=[(1, kernel_size),
                                (kernel_size, 1)], filters=filters)
        )

        self.branch_kx1x1xk = block(
            input_shape=self.input_shape, layout=layout,
            c=dict(kernel_size=[(kernel_size, 1),
                                (1, kernel_size)], filters=filters)
        )

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        shape = np.zeros_like(self.input_shape, dtype=np.int)
        shape[1:] = self.input_shape[1:]
        if self.how in ('+', '*'):
            shape[0] = self.filters
        else:
            shape[0] = self.filters * 2
        return shape

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for Global Convolutional block. """
        x = self.branch_1xkxkx1(inputs)
        y = self.branch_kx1x1xk(inputs)
        return self.merge(x, y, how=self.how)
