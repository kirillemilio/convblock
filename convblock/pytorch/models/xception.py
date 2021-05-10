import os
import sys
import numpy as np
import torch
from torch.nn import functional as F

from ..layers.conv_block import ResConvBlock as ConvBlock
from ..bases import Sequential
from .base_model import BaseModel


class Xception(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['input'] = {
            'layout': 'cna cna',
            'c': {
                'kernel_size': [3, 3],
                'stride': [2, 1],
                'filters': [32, 64]
            }
        }

        config['body'] = {
            'filters': [128, 256, 728] + 8 * [728],
            'num_repeats': [2, 2, 2] + 8 * [3],
            'downsample': [True, True, True] + 8 * [False],
            'kernel_size': 3,
            'pool_size': 3,
            'pool_mode': 'max',
            'layout': 'acn'
        }

        config['head'] = {
            'layout': '+ accn accn p + ccna ccna > fa',
            'c': {
                'kernel_size': [3, 1, 3, 1,
                                3, 1, 3, 1],
                'filters': [728, 728,
                            1024, 1024,
                            1536, 1536,
                            2048, 2048],
            },
            'p': {
                'kernel_size': 3,
                'stride': 2,
                'mode': 'max'
            },
            'f': {
                'out_features': 1000,
                'bias': True
            }
        }
        return config

    @classmethod
    def block(cls,
              input_shape: 'Tuple[int]',
              filters: int = 128,
              kernel_size: int = 3,
              dilation: int = 1,
              pool_size: int = 3,
              pool_mode: str = 'max',
              num_repeats: int = 3,
              layout: str = 'acn',
              downsample: bool = True,
              init_relu: bool = True,
              **kwargs):
        kernel_size = int(kernel_size)
        filters = int(filters)
        dilation = int(dilation)
        pool_size = int(pool_size)

        layout = (
            '+' + 'a' * init_relu
            + layout.replace('a', '').replace('c', 'cc')
            + layout.replace('c', 'cc') * (num_repeats - 1)
            + 'p' * downsample + '+'
        )
        groups = [input_shape[0], 1] + (num_repeats - 1) * [filters, 1]
        filters = [input_shape[0], filters] + (num_repeats - 1) * [filters, filters]
        dilation = num_repeats * [dilation, 1]
        kernel_size = num_repeats * [kernel_size, 1]
        return ConvBlock(
            input_shape=input_shape, layout=layout,
            c=dict(kernel_size=kernel_size, groups=groups,
                   filters=filters, dilation=dilation),
            **kwargs
        )

    def build_body(self, input_shape, config=None, **kwargs):
        filters = config['filters']
        kernel_size = config['kernel_size']
        pool_size = config['pool_size']
        pool_mode = config['pool_mode']
        num_repeats = config['num_repeats']
        downsample = config['downsample']

        shape = input_shape
        layers = Sequential()

        for i, (f, d, n) in enumerate(zip(filters, downsample, num_repeats)):
            layer = self.block(input_shape=shape,
                               pool_size=pool_size,
                               pool_mode=pool_mode,
                               kernel_size=kernel_size,
                               downsample=d, filters=f,
                               num_repeats=n, init_relu=i != 0)
            shape = layer.output_shape
            layers.add_module(f'Block_{i}', layer)

        return layers
