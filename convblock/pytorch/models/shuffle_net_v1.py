""" Contains implementation of parametrized ShuffleNet model. """

from ..blocks import CShuffleUnit
from ..bases import Sequential
from .base_model import BaseModel


class ShuffleNetV1(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=3, stride=2, filters=24),
            'p': dict(kernel_size=3, stride=2, mode='max')
        }

        config['body'] = {
            'filters': (272, 544, 1088),
            'num_blocks': (4, 8, 4),
            'shuffle_unit': {
                'groups': 4,
                'factor': 4,
                'pool_kernel': 3,
                'pool_mode': 'avg',
                'post_activation': True,
                'layout': 'cna s cn cn'
            }
        }

        config['head'] = {
            'layout': '> fa',
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    def build_body(self, input_shape, config):
        shuffle_unit_config = config.get('shuffle_unit')
        filters = config.get('filters')
        num_blocks = config.get('num_blocks')
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            for j, repeats in enumerate(range(num_blocks[i])):
                iconfig = {
                    'downsample': j == 0,
                    'filters': ifilters,
                    'block': self.conv_block,
                    **shuffle_unit_config
                }
                if i == 0 and j == 0:
                    iconfig.update({'groups': 1})
                x = CShuffleUnit(shape, **iconfig)

                body.add_module("Block_{}_{}".format(i, j), x)
                shape = x.output_shape
        return body
