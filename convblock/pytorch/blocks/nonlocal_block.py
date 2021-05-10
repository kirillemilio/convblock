""" Contains implementation of non-local convolutional block. """

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..layers import ConvBlock
from ..bases import Module


def put_channels_last(x: 'Tensor') -> 'Tensor':
    """ Put channels dimension of input tensor as it's last dim. """
    if len(x.shape) == 1:
        return x
    elif len(x.shape) == 3:
        return x.permute(0, 2, 1).contiguous()
    elif len(x.shape) == 4:
        return x.permute(0, 2, 3, 1).contiguous()
    elif len(x.shape) == 5:
        return x.permute(0, 2, 3, 4, 1).contiguous()
    raise ValueError("Input tensor must have shape greter then 1 "
                     + "but less then 5. Got {}.".format(x.shape))


def put_channels_first(x: 'Tensor') -> 'Tensor':
    """ The reverse of 'put_channels_last' function. """
    if len(x.shape) == 2:
        return x
    elif len(x.shape) == 3:
        return x.permute(0, 2, 1).contiguous()
    elif len(x.shape) == 4:
        return x.permute(0, 3, 1, 2).contiguous()
    elif len(x.shape) == 5:
        return x.permute(0, 4, 1, 2, 3).contiguous()
    raise ValueError("Input tensor must have shape greter then 1 "
                     + "but less then 5. Got {}.".format(x.shape))


class NonLocalBlock(Module):

    def __init__(self, input_shape, filters):
        """ Build non-local convolutional block block.

        Implementation of non-local convolutional block from
        https://arxiv.org/abs/1711.07971.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        filters : Tuple[int], List[int] or NDArray[int]
            filters argument must have length equal to 2.
        """
        super().__init__(input_shape)

        self.theta = ConvBlock(
            input_shape=self.input_shape, layout='cna',
            c=dict(kernel_size=1, filters=filters[0]),
        )

        self.phi = ConvBlock(
            input_shape=self.input_shape, layout='cna',
            c=dict(kernel_size=1, filters=filters[0])
        )

        self.mu = ConvBlock(
            input_shape=self.input_shape, layout='cna',
            c=dict(kernel_size=1, filters=filters[1])
        )

        self.pre_head = ConvBlock(
            input_shape=np.array([filters[1], *self.input_shape[1:]]),
            layout='c', c=dict(kernel_size=1, filters=self.input_shape[0])
        )

        self.post_head = ConvBlock(
            input_shape=self.input_shape, layout='na'
        )

    @classmethod
    def _forward_kernel(cls, x: 'Tensor', y: 'Tensor', z: 'Tensor') -> 'Tensor':
        """ Main part of forward pass method of nonlocal block.

        This part doesn't contain head part with residuals.

        Parameters
        ----------
        x : Tensor
            first input Tensor, output of theta layer.
        y : Tensor
            second input Tensor, output of phi layer.
        z : Tensor
            third input Tensor, output of mu layer.

        Returns
        -------
        Tensor
        """
        x_filters, *x_shape = x.shape[1:]
        y_filters, *y_shape = y.shape[1:]
        z_filters, *z_shape = z.shape[1:]
        u_list = []
        for i in range(x.shape[0]):
            xi = (x[i, ...]).view(x_filters, -1).t()
            yi = (y[i, ...]).view(y_filters, -1)
            zi = (z[i, ...]).view(z_filters, -1).t()
            v = F.softmax(torch.matmul(xi, yi), 1)
            v = torch.matmul(v, zi).view(1, *z_shape, z_filters)
            v = put_channels_first(v)
            u_list.append(v)
        return torch.cat(u_list, 0)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for nonlocal block. """
        x = self._forward_kernel(self.theta(inputs),
                                 self.phi(inputs),
                                 self.mu(inputs))
        x = self.pre_head(x)
        return self.post_head(x + inputs)

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return self.input_shape
