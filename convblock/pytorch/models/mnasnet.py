import math
import numpy as np
import torch
from torch.nn import functional as F
from ..layers.conv_block import ResConvBlock as ConvBlock
from ..bases import Sequential
from .base_model import BaseModel


class MNasNet(BaseModel):

    @classmethod
    def inverted_residual(cls,
                          input_shape,
                          filters: int,
                          kernel_size: int = 3,
                          dilation: int = 1,
                          factor: float = 4,
                          downsample: bool = False):
        mid_channels = int(input_shape[0] * factor)
        groups = [1, mid_channels, 1]
        filters = [mid_channels, mid_channels, filters]
        kernel_size = [1, kernel_size, 1], [1, dilation, 1]
        if input_shape[0] == filters and not downsample:
            layout = '+ cna cna cn +'
            stride = [1, 1, 1]
        else:
            layout = 'cna cna cn'
            stride = [1, 2, 1]
        return ConvBlock(
            input_shape=input_shape,
            layout=layout,
            c=dict(kernel_size=kernel_size, stride=stride,
                   filters=filters, groups=groups,
                   dilation=dilation)
        )

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cna cna cn',
            'c': {
                'kernel_size': [3, 3, 1],
                'groups': [1, 32, 1],
                'stride': [1, 2, 1],
                'filters': [32, 32, 16]
            }
        }

        config['body'] = {
            'filters': [24, 40, 80, 96, 192, 320],
            'num_repeats': [3, 3, 3, 2, 4, 1],
            'kernel_size': [3, 5, 5, 3, 5, 3],
            'factor': [3, 3, 6, 6, 6, 6],
            'downsample': [True, True, True,
                           False, True, False]
        }

        config['head'] = {
            'layout': 'cna > df',
            'c': {
                'kernel_size': 3,
                'filters': 1280
            },
            'f': {
                'out_features': 1000
            }
        }
        return config

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        downsample = config.get('downsample')
        factor = config.get('factor')
        kernel_size = config.get('kernel_size')
        num_repeats = config.get('num_repeats')
        assert len(filters) == len(downsample)
        assert len(filters) == len(factor)
        assert len(filters) == len(kernel_size)
        assert len(filters) == len(num_repeats)

        shape = input_shape
        body = Sequential()
        for i, n in enumerate(num_repeats):
            for j in range(n):
                x = self.inverted_residual(
                    input_shape=shape,
                    filters=filters[i],
                    downsample=downsample[i] and (j == 0),
                    factor=factor[i],
                    kernel_size=kernel_size[i]
                )
                body.add_module(f"Block-{i}-{j}", x)
                shape = x.output_shape

        return body
