import numpy as np
import torch

from ..bases import Sequential, Module
from ..layers.conv_block import ResConvBlock as ConvBlock, ConvBranches
from ..utils import LIST_TYPES


class ASPP(Module):

    def __init__(self, input_shape, filters, rates, proba=0.5):
        super().__init__(input_shape)

        if not isinstance(rates, LIST_TYPES):
            raise ValueError("Argument rates must be list-like. "
                             + "Got type {}.".format(type(rates)))

        self._input_shape = input_shape
        self.layers = Sequential(
            ConvBranches(
                input_shape=self.input_shape, mode='.',
                branch0=dict(layout='p cna u',
                            c={'filters': filters, 'kernel_size': 1,
                                'stride': 1, 'dilation': 1},
                            p={'mode': 'avg', 'output_size': 1, 'adaptive': True},
                            u={'size': tuple(self.input_shape[1:])}),
                branch1=dict(layout='cna', c={'filters': filters, 'kernel_size': 3,
                                              'stride': 1, 'dilation': rates[0]}),
                branch2=dict(layout='cna', c={'filters': filters, 'kernel_size': 3,
                                              'stride': 1, 'dilation': rates[1]}),
                branch3=dict(layout='cna', c={'filters': filters, 'kernel_size': 3,
                                              'stride': 1, 'dilation': rates[2]}),
                branch4=dict(layout='cna', c={'filters': filters, 'kernel_size': 3,
                                              'stride': 1, 'dilation': rates[3]})
            ),
            ConvBlock(input_shape=(filters * 5, *self.input_shape[1:]), layout='cna d',
                    c=dict(kernel_size=3, filters=32, stride=1, dilation=1), d=dict(p=proba))
        )

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        self.layers.output_shape

    def forward(self, inputs):
        return self.layers(inputs)
