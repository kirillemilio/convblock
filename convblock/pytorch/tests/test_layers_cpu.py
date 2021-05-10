""" Contains tests for basic layers. """

import os
import sys
from itertools import product
import numpy as np
import pandas as pd
import pytest
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from ..utils import transform_to_int_tuple, transform_to_float_tuple
from ..layers.activation import softmax
from ..layers.conv import Conv, ConvTransposed
from ..layers.pool import MaxPool, AvgPool, LPPool, Pool, GlobalPool
from ..layers.layers import Dropout, BatchNorm, InstanceNorm, Linear, Flatten
from ..layers.activation import Activation


@pytest.fixture(scope='module')
def cpu_inputs_1d():
    return Variable(torch.rand(2, 3, 64))


@pytest.fixture(scope='module')
def cpu_inputs_2d():
    return Variable(torch.rand(2, 3, 64, 64))


@pytest.fixture(scope='module')
def cpu_inputs_3d():
    return Variable(torch.rand(2, 1, 64, 64, 64))


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


def test_flatten(inputs):
    input_shape = get_input_shape(inputs)
    output_shape = np.prod(input_shape)
    layer = Flatten(input_shape=input_shape)

    assert np.all(layer.input_shape == input_shape)
    assert np.all(layer.output_shape == output_shape)
    assert np.all(layer(inputs).shape[1:] == output_shape)


def test_batch_norm(inputs):
    input_shape = get_input_shape(inputs)
    output_shape = input_shape
    layer = BatchNorm(input_shape=input_shape)

    assert np.all(layer.input_shape == input_shape)
    assert np.all(layer.output_shape == output_shape)
    assert np.all(layer(inputs).shape[1:] == output_shape)


def test_instance_norm(inputs):
    input_shape = get_input_shape(inputs)
    output_shape = input_shape
    layer = InstanceNorm(input_shape=input_shape)

    assert np.all(layer.input_shape == input_shape)
    assert np.all(layer.output_shape == output_shape)
    assert np.all(layer(inputs).shape[1:] == output_shape)


@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('p', [0, 0.1, 0.35, 0.7, 1.0])
def test_dropout_pixelwise(inputs, p, inplace):
    input_shape = get_input_shape(inputs)
    output_shape = input_shape
    layer = Dropout(input_shape, p, inplace)

    assert np.all(layer.input_shape == input_shape)
    assert np.all(layer.output_shape == output_shape)
    assert np.all(layer(inputs).shape[1:] == output_shape)


def test_linear(out_features=10):
    x = torch.rand(2, 128)
    input_shape = get_input_shape(x)
    output_shape = np.array([out_features], dtype=np.int)

    layer = Linear(input_shape=input_shape, out_features=out_features)

    assert np.all(layer.input_shape == input_shape)
    assert np.all(layer.output_shape == output_shape)
    assert np.all(layer(x).shape[1:] == output_shape)


@pytest.mark.parametrize('dilation', [1, 2, 4])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('kernel_size', [1, 3, 4, 7])
@pytest.mark.parametrize('filters', [4, 16])
def test_conv(inputs, filters, kernel_size, stride, dilation):
    padding, bias, groups = 'constant', True, 1
    input_shape = get_input_shape(inputs)
    layer = Conv(input_shape, filters, kernel_size,
                 stride, dilation, groups, padding, bias)

    assert np.all(layer.output_shape == layer(inputs).shape[1:])


@pytest.mark.parametrize('padding,bias,groups', [('constant', True, 1),
                                                 ('reflect', False, 2),
                                                 ('replicate', -100, 2)])
def test_conv_padding_bias_groups(inputs, padding, bias, groups):
    filters, kernel_size, stride, dilation = 8, 3, 1, 1
    input_shape = get_input_shape(inputs)
    if padding == 'reflect' and len(input_shape) == 4:
        return
    layer = Conv(input_shape, filters, kernel_size, stride,
                 dilation, groups, padding, bias)

    assert np.all(layer.output_shape == layer(inputs).shape[1:])


@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('kernel_size', [3, 5])
@pytest.mark.parametrize('filters', [4, 16])
def test_conv_transposed(inputs, filters, kernel_size, stride, dilation):
    padding, bias, groups = 'constant', True, 1
    input_shape = get_input_shape(inputs)
    layer = ConvTransposed(input_shape, filters, kernel_size,
                           stride, dilation, groups, padding, bias)

    assert np.all(layer.output_shape == layer(inputs).shape[1:])


@pytest.mark.parametrize('mode', ['max'])
@pytest.mark.parametrize('dilation', [1, 2, 4])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('kernel_size', [1, 3, 7])
def test_max_pool(inputs, kernel_size, stride, dilation, mode):
    padding = 'constant'
    input_shape = get_input_shape(inputs)
    layer = Pool(input_shape, kernel_size, stride, dilation, mode, padding)
    assert np.all(layer.output_shape == layer(inputs).shape[1:])


@pytest.mark.parametrize('mode', ['avg'])
@pytest.mark.parametrize('dilation', [1])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('kernel_size', [1, 3, 7])
def test_avg_pool(inputs, kernel_size, stride, dilation, mode):
    padding = 'constant'
    input_shape = get_input_shape(inputs)
    layer = Pool(input_shape, kernel_size, stride, dilation, mode, padding)
    assert np.all(layer.output_shape == layer(inputs).shape[1:])


@pytest.mark.parametrize('mode', ['lp'])
@pytest.mark.parametrize('dilation', [1])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('kernel_size', [1, 3, 7])
def test_lp_pool(inputs, kernel_size, stride, dilation, mode):
    padding = 'constant'
    input_shape = get_input_shape(inputs)
    if len(input_shape) == 4:
        return
    layer = Pool(input_shape, kernel_size, stride, dilation, mode, padding)
    assert np.all(layer.output_shape == layer(inputs).shape[1:])


@pytest.mark.parametrize('mode', ['max', 'avg'])
def test_global_pool(inputs, mode):
    input_shape = get_input_shape(inputs)
    layer = GlobalPool(input_shape, mode)
    assert np.all(layer.output_shape == layer.forward(inputs).shape[1:])


@pytest.mark.parametrize('activation', [None, 'linear', 'relu', 'prelu', 'elu',
                                        'selu', 'sigmoid', 'leaky_relu', 'softmax'])
def test_activation(inputs, activation):
    input_shape = get_input_shape(inputs)
    layer = Activation(input_shape, activation)
    assert np.all(layer.output_shape == layer(inputs).shape[1:])
