""" Contains implementation of InceptionV4 architecture. """

from ..layers.conv_block import ConvBlock
from ..layers.conv_block import ConvBranches
from ..bases import Sequential, Module
from .base_model import BaseModel
from ..blocks.inception import InceptionA, InceptionB, InceptionC
from ..blocks.inception import ReductionA, ReductionB


class InceptionV4(BaseModel):

    @classmethod
    def mixed_3a(cls, input_shape):
        return ConvBranches(
            input_shape=input_shape, mode='.',
            maxpool={
                'layout': 'p',
                'p': {
                    'kernel_size': 3,
                    'stride': 2,
                    'mode': 'max'
                }
            },
            conv={
                'layout': 'cna',
                'c': {
                    'kernel_size': 3,
                    'stride': 2,
                    'filters': 96
                }
            }
        )

    @classmethod
    def mixed_4a(cls, input_shape):
        return ConvBranches(
            input_shape=input_shape, mode='.',
            branch0={
                'layout': 'cna cna',
                'c': {
                    'kernel_size': [1, 3],
                    'stride': [1, 1],
                    'filters': [64, 96]
                }
            },
            branch1={
                'layout': 'cna cna cna cna',
                'c': {
                    'kernel_size': [(1, 1),
                                    (1, 7),
                                    (7, 1),
                                    (3, 3)],
                    'stride': [1, 1, 1, 1],
                    'filters': [64, 64, 64, 96]
                }
            }
        )

    @classmethod
    def mixed_5a(cls, input_shape):
        return ConvBranches(
            input_shape=input_shape, mode='.',
            maxpool={
                'layout': 'p',
                'p': {
                    'kernel_size': 3,
                    'stride': 2,
                    'mode': 'max'
                }
            },
            conv={
                'layout': 'cna',
                'c': {
                    'kernel_size': 3,
                    'stride': 2,
                    'filters': 192
                }
            }
        )

    @classmethod
    def inception_a(cls, input_shape, **kwargs):
        return InceptionA(
            input_shape=input_shape,
            filters1x1=96,
            filters5x5=(64, 96),
            filters3x3dbl=(64, 96, 96),
            filters_pool=96,
            kernel_size=3,
            **kwargs
        )

    @classmethod
    def inception_b(cls, input_shape, **kwargs):
        return InceptionB(
            input_shape=input_shape,
            filters1x1=192,
            filters7x7=(192, 224, 256),
            filters7x7dbl=(192, 192, 224,
                           224, 256),
            filters_pool=128,
            **kwargs
        )

    @classmethod
    def inception_c(cls, input_shape, **kwargs):
        return InceptionC(
            input_shape=input_shape,
            filters1x1=256,
            filters3x3=(384, 256, 256),
            filters3x3dbl=(384, 448, 512, 256, 256),
            filters_pool=256,
            version=4,
            **kwargs
        )

    @classmethod
    def reduction_a(cls, input_shape, **kwargs):
        return ReductionA(
            input_shape=input_shape,
            filters3x3=384,
            filters3x3dbl=(192, 224, 256),
            **kwargs
        )

    @classmethod
    def reduction_b(cls, input_shape, **kwargs):
        return ReductionB(
            input_shape=input_shape,
            filters3x3=(192, 192),
            filters7x7x3=(256, 256, 320, 320),
            **kwargs
        )

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cna cna cna',
            'c': dict(kernel_size=[3, 3, 3],
                      stride=[2, 1, 1],
                      filters=[32, 32, 64]),
        }

        config['body'] = {
            'layout': (4 * 'a' + 'A'
                       + 7 * 'b' + 'B'
                       + 3 * 'c')
        }

        config['head'] = {
            'layout': '> fa',
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    @classmethod
    def build_body(cls, input_shape, config):
        shape = input_shape
        layout = config.get('layout')

        body = Sequential()
        body.add_module('Mixed_3a', cls.mixed_3a(shape))
        shape = body.output_shape
        body.add_module('Mixed_4a', cls.mixed_4a(shape))
        shape = body.output_shape
        body.add_module('Mixed_5a', cls.mixed_5a(shape))
        shape = body.output_shape
        modules = []
        for i, layer in enumerate(layout):
            if layer == 'a':
                x = cls.inception_a(shape)
            elif layer == 'b':
                x = cls.inception_b(shape)
            elif layer == 'c':
                x = cls.inception_c(shape)
            elif layer == 'A':
                x = cls.reduction_a(shape)
            elif layer == 'B':
                x = cls.reduction_b(shape)
            shape = x.output_shape
            modules.append(x)
        return Sequential(*body, *modules)
