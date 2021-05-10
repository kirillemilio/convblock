""" Contains implementation of InceptionV3 architecture. """

from ..layers.conv_block import ConvBlock
from ..layers.conv_block import ConvBranches
from ..bases import Sequential, Module
from .base_model import BaseModel
from ..blocks.inception import InceptionA, InceptionB, InceptionC
from ..blocks.inception import ReductionA, ReductionB


class InceptionV3(BaseModel):

    @classmethod
    def inception_a(cls, input_shape, filters_pool: int = 32, **kwargs):
        return InceptionA(
            input_shape=input_shape,
            filters1x1=64,
            filters5x5=(48, 64),
            filters3x3dbl=(64, 96, 96),
            filters_pool=filters_pool,
            kernel_size=3,
            **kwargs
        )

    @classmethod
    def inception_b(cls, input_shape, filters7x7: int = 128, **kwargs):
        return InceptionB(
            input_shape=input_shape,
            filters1x1=192,
            filters7x7=(filters7x7, filters7x7, 192),
            filters7x7dbl=(filters7x7, filters7x7,
                           filters7x7, filters7x7,
                           192),
            filters_pool=92,
            **kwargs
        )

    @classmethod
    def inception_c(cls, input_shape, **kwargs):
        return InceptionC(
            input_shape=input_shape,
            filters1x1=320,
            filters3x3=(384, 384, 384),
            filters3x3dbl=(448, 384, 384, 384),
            filters_pool=192,
            version=3,
            **kwargs
        )

    @classmethod
    def reduction_a(cls, input_shape, **kwargs):
        return ReductionA(
            input_shape=input_shape,
            filters3x3=384,
            filters3x3dbl=(64, 96, 96),
            **kwargs
        )

    @classmethod
    def reduction_b(cls, input_shape, **kwargs):
        return ReductionB(
            input_shape=input_shape,
            filters3x3=(192, 320),
            filters7x7x3=(192, 192, 192, 192),
            **kwargs
        )

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cna cna cna cna cna',
            'c': dict(kernel_size=[3, 3, 3, 1, 3],
                      stride=[2, 1, 1, 1, 1],
                      filters=[32, 32, 64, 80, 192]),
        }

        config['body'] = {
            'layout': (3 * 'a' + 'A'
                       + 4 * 'b' + 'B'
                       + 2 * 'c')
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
        modules = []

        filters_pool = {0: 32, 1: 64, 2: 64}
        filters7x7 = {0: 128, 1: 160, 2: 160, 3: 192}

        a_counter, b_counter = 0, 0
        for i, layer in enumerate(layout):
            if layer == 'a':
                x = cls.inception_a(
                    input_shape=shape,
                    filters_pool=filters_pool.get(a_counter, 64)
                )
                a_counter += 1
            elif layer == 'b':
                x = cls.inception_b(
                    input_shape=shape,
                    filters7x7=filters7x7.get(b_counter, 192)
                )
                b_counter += 1
            elif layer == 'c':
                x = cls.inception_c(shape)
            elif layer == 'A':
                x = cls.reduction_a(shape)
            elif layer == 'B':
                x = cls.reduction_b(shape)
            shape = x.output_shape
            modules.append(x)
        return Sequential(*body, *modules)
