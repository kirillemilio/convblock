""" Contains implementation of parametrized MobileNetV3 model. """
import numpy as np

from .base_model import BaseModel
from .mobile_net_v2 import MobileNetV2
from ..layers import ConvBlock
from ..bases import Sequential


class MobileNetV3(MobileNetV2):

    @staticmethod
    def _make_divisible(x: int, by: int = 8):
        return int(np.ceil(x * 1. / by) * by)

    @classmethod
    def block(cls,
              input_shape,
              filters: int,
              exp_filters: int,
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
        filters, exp_filters = int(filters), int(exp_filters)
        se_reduction = int(se_reduction)
        assert isinstance(activation, str)
        assert isinstance(se_activation, str)
        assert se_reduction > 0
        assert exp_filters > 0
        use_res = input_shape[0] == filters and not downsample
        if not use_se:
            return conv_block(
                input_shape=input_shape,
                layout='+ cna cna cn +' if use_res else 'cna cna cn',
                c=dict(kernel_size=[1, kernel_size, 1],
                       groups=[1, exp_filters, 1],
                       stride=[1, 2, 1] if downsample else [1, 1, 1],
                       filters=(exp_filters, exp_filters, filters)),
                **kwargs
            )
        else:
            se_filters = exp_filters // se_reduction
            return conv_block(
                input_shape=input_shape,
                layout='+ cna cna * p cna cna * cn +' + 'a' * post_activation,
                c=dict(kernel_size=[1, kernel_size, 1, 1, 1],
                       groups=[1, exp_filters, 1, 1, 1],
                       stride=[1, 2 if downsample else 1, 1, 1, 1],
                       filters=[exp_filters, exp_filters,
                                se_filters, exp_filters, filters]),
                p=dict(output_size=1, adaptive=True, mode='avg'),
                a=dict(activation=(3 * [activation] + [se_activation]
                                   + post_activation * [activation])),
                **kwargs
            )

    def build_body(self, input_shape, config):
        filters = config.get('filters')
        exp_filters = config.get('exp_filters')
        use_se = config.get('use_se')
        kernel_size = config.get('kernel_size')
        downsample = config.get('downsample')
        activation = config.get('activation')
        compression = config.get('compression', 1.0)
        min_depth = config.get('min_depth', 1.0)
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            iconfig = {
                'input_shape': shape,
                'filters': max(self._make_divisible(ifilters * compression), min_depth),
                'exp_filters': max(self._make_divisible(exp_filters[i] * compression), min_depth),
                'downsample': downsample[i],
                'activation': activation[i],
                'kernel_size': kernel_size[i],
                'use_se': use_se[i],
                'block': self.conv_block
            }
            x = self.block(**iconfig)

            body.add_module(f"Block_{i}", x)
            shape = x.output_shape
        return body


class MobileNetV3Small(MobileNetV3):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['body'] = {
            'min_depth': 8,
            'compression': 1.0,
            'activation': [
                'relu',
                'relu', 'relu',
                'hswish', 'hswish', 'hswish',
                'hswish', 'hswish',
                'hswish', 'hswish', 'hswish'
            ],
            'kernel_size': [
                3,
                3, 3,
                5, 5, 5,
                5, 5,
                5, 5, 5
            ],
            'filters': [
                16,
                24, 24,
                40, 40, 40,
                48, 48,
                96, 96, 96
            ],
            'exp_filters': [
                16,
                72, 88,
                96, 240, 240,
                120, 144,
                288, 576, 576
            ],
            'downsample': [
                True,
                True, False,
                True, False, False,
                False, False,
                True, False, False
            ],
            'use_se': [
                True,
                False, False,
                True, True, True,
                True, True,
                True, True, True
            ]
        }
        config['head'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=1028),
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation=('relu', 'linear'))
        }
        return config


class MobileNetV3Large(MobileNetV3):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['body'] = {
            'min_depth': 8,
            'compression': 1.0,
            'activation': [
                'relu',
                'relu', 'relu',
                'relu', 'relu', 'relu',
                'hswish', 'hswish', 'hswish', 'hswish',
                'hswish', 'hswish',
                'hswish', 'hswish', 'hswish'
            ],
            'kernel_size': [
                3,
                3, 3,
                5, 5, 5,
                3, 3, 3, 3,
                3, 3,
                5, 5, 5
            ],
            'filters': [
                16,
                24, 24,
                40, 40, 40,
                80, 80, 80, 80,
                112, 122,
                160, 160, 160
            ],
            'exp_filters': [
                16,
                64, 72,
                72, 120, 120,
                240, 200, 184, 184,
                480, 672,
                672, 960, 960
            ],
            'downsample': [
                False,
                True, False,
                True, False, False,
                True, False, False, False,
                False, False,
                True, False, False
            ],
            'use_se': [
                False,
                False, False,
                True, True, True,
                False, False, False, False,
                True, True,
                True, True, True
            ]
        }
        config['head'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=1028),
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation=('relu', 'linear'))
        }
        return config
