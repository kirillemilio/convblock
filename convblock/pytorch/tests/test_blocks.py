""" Contains tests for basic blocks. """

import os
import sys
import pytest
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from ..utils import transform_to_int_tuple, transform_to_float_tuple
from ..layers import ConvBlock
from ..blocks import VanillaResBlock, DenseBlock


@pytest.fixture(scope='module', params=[2, 3, 4])
def inputs(request):
    ndims = request.param
    if ndims == 2:
        inputs = torch.rand(2, 4, 64)
    elif ndims == 3:
        inputs = torch.rand(2, 4, 61, 64)
    elif ndims == 4:
        inputs = torch.rand(2, 4, 61, 65, 64)

    return Variable(inputs)


def get_input_shape(inputs):
    input_shape = transform_to_int_tuple(inputs.shape[1:],
                                         'input_shape',
                                         len(inputs.shape)-1)
    return np.array(input_shape, np.int)


@pytest.mark.parametrize('kernel_size,filters,layout',
                         [(1, 16, 'ca'), (3, [16], 'ca'),
                          ([3], (16, ), 'ca'), ((3, ), 16, 'ca'),
                          (np.array([3]), 16, 'ca'), (3, 16, 'ca ca'),
                          ([3, 1], [16, 8], 'ca ca')])
def test_conv_block(inputs, kernel_size, filters, layout):
    block = ConvBlock(
        input_shape=get_input_shape(inputs),
        layout=layout,
        c=dict(kernel_size=kernel_size, filters=filters)
    )
    assert np.all(block.output_shape == block(inputs).shape[1:])


@pytest.mark.parametrize('how', ['+', '.'])
@pytest.mark.parametrize('downsample', [True, False])
@pytest.mark.parametrize('filters', [8, [2], (4, 2)])
@pytest.mark.parametrize('layout', ['cna', 'cna cn', 'cna cna'])
def test_vanilla_resblock(inputs, layout, filters, downsample, how):
    block = VanillaResBlock(
        input_shape=get_input_shape(inputs),
        filters=filters, layout=layout,
        downsample=downsample, how=how
    )
    assert np.all(block.output_shape == block(inputs).shape[1:])
