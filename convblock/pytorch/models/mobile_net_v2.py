""" Contains implementation of parametrized MobileNetV2 model. """
import numpy as np

from ..layers import ConvBlock
from ..bases import Sequential
from .base_model import BaseModel


class MobileNetV2(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['conv_block'] = {
            'a': dict(activation='relu6')
        }

        config['input'] = {
            'layout': 'cna',
            'c': dict(kernel_size=3, stride=2, filters=32),
            'a': dict(activation='relu')
        }

        config['body'] = {
            'min_depth': 8,
            'compression': 1.0,
            'layout': 'cna cna cn',
            'filters': (16, 24, 32, 64, 96, 160, 320),
            'num_blocks': (1, 2, 3, 4, 3, 3, 1),
            'factor': (1, 6, 6, 6, 6, 6, 6),
            'downsample': (False, True, True, True, False, True, False)
        }

        config['head'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=1028),
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation=('relu', 'linear'))
        }
        return config
            
    @staticmethod
    def _make_divisible(x: int, by: int = 8):
        return int(np.ceil(x * 1. / by) * by)

    @classmethod
    def block(cls,
              input_shape,
              filters: int,
              factor: int = 4,
              kernel_size: int = 3,
              dilation: int = 1,
              activation: str = 'relu6',
              se_reduction: int = 4,
              se_activation: str = 'sigmoid',
              downsample: bool = False,
              use_se: bool = False,
              post_activation: bool = False,
              block=None,
              **kwargs):
        conv_block = ConvBlock if block is None else block
        kernel_size, dilation = int(kernel_size), int(dilation)
        factor, se_reduction = int(factor), int(se_reduction)
        assert isinstance(activation, str)
        assert isinstance(se_activation, str)
        assert factor > 0
        assert se_reduction > 0
        use_res = input_shape[0] == filters and not downsample
        expand_filters = int(round(input_shape[0] * factor))
        if not use_se:
            return conv_block(
                input_shape=input_shape,
                layout='+ cna cna cn +' if use_res else 'cna cna cn',
                c=dict(kernel_size=[1, kernel_size, 1],
                       groups=[1, expand_filters, 1],
                       stride=[1, 2, 1] if downsample else [1, 1, 1],
                       filters=(expand_filters, expand_filters, filters)),
                **kwargs
            )
        else:
            se_filters = expand_filters // se_reduction
            return conv_block(
                input_shape=input_shape,
                layout='+ cna cna * p cna cna * cn +' + 'a' * post_activation,
                c=dict(kernel_size=[1, kernel_size, 1, 1, 1],
                       stride=[1, 2 if downsample else 1, 1, 1, 1],
                       groups=[1, expand_filters, 1, 1, 1],
                       filters=[expand_filters, expand_filters,
                                se_filters, expand_filters, filters]),
                p=dict(output_size=1, adaptive=True, mode='avg'),
                a=dict(activation=(3 * [activation] + [se_activation]
                                   + post_activation * [activation])),
                **kwargs
            )

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        downsample = config.get('downsample')
        factor = config.get('factor')
        num_repeats = config.get('num_repeats')
        compression = config.get('compression', 1.0)
        min_depth = config.get('min_depth', 1.0)
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            for j in range(num_repeats[i]):
                iconfig = {
                    'input_shape': shape,
                    'filters': max(self._make_divisible(ifilters * compression), min_depth),
                    'downsample': downsample[i] and (j == 0),
                    'factor': factor[i],
                    'layout': config.get('layout', 'cna cna cn'),
                    'block': self.conv_block
                }
                x = self.block(**iconfig)

                body.add_module("Block_{}_{}".format(i, j), x)
                shape = x.output_shape
        return body
