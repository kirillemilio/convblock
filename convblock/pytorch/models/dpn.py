""" Contains implementation of parametrized DPN model. """


from ..blocks import DPNBlock
from ..bases import Sequential
from ..bases import Cat
from ..layers import ConvBlock
from .base_model import BaseModel


class DPN(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=7, stride=2, filters=32),
            'p': dict(kernel_size=3, stride=2)
        }

        config['body'] = {
            'filters': (64, 128, 256, 512),
            'inc': (16, 32, 24, 128),
            'num_blocks': (3, 4, 20, 3),
            'factor': 96,
            'groups': 1
        }

        config['head'] = {
            'layout': '> fa',
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    def build_body(self, input_shape, config):
        num_blocks = config.get('num_blocks')
        filters = config.get('filters')
        inc = config.get('inc')
        groups = config.get('groups', 1)
        factor = config.get('factor', 1)

        layers = Sequential()
        shape = input_shape
        for i, (k, f, d) in enumerate(zip(num_blocks,
                                          filters,
                                          inc)):
            for j in range(k):
                x = DPNBlock(input_shape=shape,
                             filters=(f, f, factor * 2 ** i),
                             groups=groups, inc=d,
                             downsample=(i > 0) and (j == 0),
                             force_proj=(i == 0) and (j == 0))

                shape = x.output_shape
                layers.add_module(f'Block_{i}_{j}', x)

        layers = Cat(layers)
        return Sequential(layers, ConvBlock(
            input_shape=layers.output_shape,
            layout='na'))
