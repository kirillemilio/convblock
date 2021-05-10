""" Contains implementation of NasNet architecture convolutional cells. """

import torch
from ..layers import ConvBlock
from ..layers.conv_block import ConvBranches
from ..bases import Module


class NASCell(Module):

    def __init__(self, input_shape: 'ArrayLike[int]',
                 stride: tuple = (1, 1)):
        super().__init__(input_shape)
        filters = self.input_shape[:, 0]
        stride = [int(v) for v in stride]
        self.conv_3x3_1 = ConvBlock(
            input_shape=self.input_shape[0],
            layout='cna',
            c=dict(kernel_size=3,
                   filters=filters[0],
                   groups=filters[0],
                   stride=stride[0])
        )
        self.conv_3x3_2 = ConvBlock(
            input_shape=self.input_shape[1],
            layout='cna',
            c=dict(kernel_size=3,
                   filters=filters[1],
                   groups=filters[1],
                   stride=stride[1])
        )
        self.pool_3x3 = ConvBlock(
            input_shape=self.input_shape[0],
            layout='p', p=dict(kernel_size=3,
                               stride=stride[0])
        )
        self.block_1 = ConvBranches(
            input_shape=self.input_shape[0],
            mode='+',
            branch_conv7={
                'layout': 'cna',
                'c': {
                    'kernel_size': 7,
                    'filters': filters[0],
                    'groups': filters[0],
                    'stride': stride[0]
                }
            },
            branch_pool={
                'layout': 'p',
                'p': {
                    'kernel_size': 3,
                    'stride': stride[0]
                }
            }
        )
        self.block_2 = ConvBranches(
            input_shape=self.input_shape[0],
            mode='+',
            branch_conv3={
                'layout': 'cna',
                'c': {
                    'kernel_size': 7,
                    'filters': filters[0],
                    'groups': filters[0],
                    'stride': stride[0]
                }
            },
            branch_pool={
                'layout': 'p',
                'p': {
                    'kernel_size': 3,
                    'stride': stride
                }
            }
        )
        self.block_3 = ConvBranches(
            input_shape=self.input_shape[1],
            mode='+',
            branch_conv3={
                'layout': 'cna',
                'c': {
                    'kernel_size': 5,
                    'filters': filters[1],
                    'groups': filters[1],
                    'stride': stride[1]
                }
            },
            branch_conv5={
                'layout': 'cna',
                'c': {
                    'kernel_size': 3,
                    'filters': filters[1],
                    'groups': filters[1],
                    'stride': stride[1]
                }
            }
        )

    def forward(self, inputs):
        x, y = inputs
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.conv_3x3_1(x2) + self.pool_3x3(x)
        x4 = self.block_3(y)
        x5 = self.conv_3x3_2(y) + x
        return torch.cat([x1, x2, x3, x4, x5], dim=1)
