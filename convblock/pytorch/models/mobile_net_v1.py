""" Contains implementation of parametrized MobileNetV1 model. """

from ..layers import ConvBlock
from ..bases import Sequential
from .base_model import BaseModel


class MobileNetV1(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cna',
            'c': dict(kernel_size=3, stride=2, filters=32),
            'a': dict(activation='relu')
        }

        config['body'] = {

            'min_depth': 8,
            'compression': 1.0,

            'filters': (64, 128,
                        128, 256,
                        256, 512,
                        512, 512,
                        512, 512, 512,
                        1024, 1024),

            'downsample': (False, True,
                           False, True,
                           False, True,
                           False, False,
                           False, False, False,
                           True, False)
        }

        config['head'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=1028),
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation=('relu', 'linear'))
        }
        return config

    @classmethod
    def block(cls, input_shape, filters,
              layout='cna', downsample=False,
              block=None, **kwargs):
        block = ConvBlock if block is None else block
        return block(
            input_shape=input_shape, layout=layout * 2,
            c=dict(kernel_size=[3, 1], filters=(int(input_shape[0]), filters),
                   groups=[int(input_shape[0]), 1],
                   stride=[2, 1] if downsample else [1, 1])
        )

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        downsample = config.get('downsample')
        compression = config.get('compression', 1.0)
        min_depth = config.get('min_depth', 1.0)
        shape = input_shape
        body = Sequential()
        for i, (ifilters, idownsample) in enumerate(zip(filters, downsample)):
            iconfig = {
                'input_shape': shape,
                'filters': max(int(ifilters * compression), min_depth),
                'downsample': downsample[i],
                'layout': config.get('layout', default='cna'),
                'block': self.conv_block
            }
            x = self.block(**iconfig)

            body.add_module("Block-{}".format(i), x)
            shape = x.output_shape
        return body
