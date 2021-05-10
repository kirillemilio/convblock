""" Contains implementation of parametrized ShuffleNet model. """

from ..blocks import CSplitAndShuffleUnit
from ..bases import Sequential
from .base_model import BaseModel


class ShuffleNetV2(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=3, stride=2, filters=24),
            'p': dict(kernel_size=3, stride=2, mode='max')
        }

        config['body'] = {
            'share': 0.5,
            'filters': (116, 232, 464),
            'num_blocks': (4, 8, 4)
        }

        config['head'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=1024),
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation=('relu', 'linear'))
        }
        return config

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        num_blocks = config.get('num_blocks')
        share = float(config.get('share'))
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            for j, repeats in enumerate(range(num_blocks[i])):
                iconfig = {
                    'downsample': j == 0,
                    'filters': ifilters,
                    'share': share,
                    'block': self.conv_block
                }
                x = CSplitAndShuffleUnit(shape, **iconfig)

                body.add_module("Block_{}_{}".format(i, j), x)
                shape = x.output_shape
        return body
