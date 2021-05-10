""" Contains implementation of Squeezenet architecture. """

from ..blocks import FireBlock
from ..blocks import FireBlockWithBypass
from ..bases import Sequential
from .base_model import BaseModel


class SqueezeNet(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cna',
            'c': dict(kernel_size=7, stride=2, filters=32)
        }

        config['body'] = {
            'filters': [
                (16, 64, 64),
                (16, 64, 64),
                (32, 128, 128),

                (32, 128, 128),
                (48, 192, 192),
                (48, 192, 192),
                (64, 256, 256),

                (64, 256, 256)
            ],
            'downsample': [
                True, False, False,
                True, False, False, False,
                True
            ],
            'use_bypass': [False] * 8,
            'layout': 'cna',
            'pool_kernel': 3,
            'pool_mode': 'max'
        }

        config['head'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=10),
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    @classmethod
    def block(cls, input_shape, filters, layout='cna', pool_kernel=3,
              pool_mode='max', downsample=False, block=None,
              how='+', use_bypass=False):

        if use_bypass:
            return FireBlockWithBypass(
                input_shape=input_shape, layout=layout,
                s1x1=filters[0], e1x1=filters[1], e3x3=filters[2],
                pool_kernel=pool_kernel, pool_mode=pool_mode,
                downsample=downsample, block=block, how=how)
        else:
            return FireBlock(
                input_shape=input_shape, layout=layout,
                s1x1=filters[0], e1x1=filters[1], e3x3=filters[2],
                pool_kernel=pool_kernel, pool_mode=pool_mode,
                downsample=downsample, block=block, how=how)

    def build_body(self, input_shape, config):
        """ Body block of squeezenet model. """
        use_bypass = config.get('use_bypass')
        downsample = config.get('downsample')
        filters = config.get('filters')
        layout = config.get('layout')
        pool_kernel = config.get('pool_kernel')
        pool_mode = config.get('pool_mode')

        shape = input_shape
        body = Sequential()
        for i in range(len(filters)):
            x = self.block(shape, filters[i],
                           layout=layout,
                           pool_kernel=pool_kernel,
                           pool_mode=pool_mode,
                           downsample=downsample[i],
                           use_bypass=use_bypass[i],
                           block=self.conv_block)

            body.add_module("Block_{}".format(i), x)
            shape = x.output_shape
        return body


class SqueezeNetSimpleBypass(SqueezeNet):

    @classmethod
    def default_config(cls):
        config = SqueezeNet.default_config()
        config['body/use_bypass'] = [
            False, True, False,
            True, False, True, False,
            True
        ]
        return config


class SqueezeNetComplexBypass(SqueezeNet):

    @classmethod
    def default_config(cls):
        config = SqueezeNet.default_config()
        config['body/use_bypass'] = [True] * 8
        return config
